from typing import Dict, Optional
import copy
import json
from pathlib import Path

import numpy as np
import torch
import zarr

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset


class GuideLowdimDataset(BaseLowdimDataset):
    """Lowdim dataset for guide-follow episodes stored as per-episode Zarr."""

    def __init__(
        self,
        data_dir: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        action_mode: str = "forward_heading",
        frame_stride: int = 1,
        n_lookahead: int = 10,
        k_lookahead: Optional[int] = None,
        lookahead_stride: Optional[int] = None,
        robot_frame: bool = True,
        robot_state: str = "vel",
        turn_speed: Optional[float] = None,
        heading_delta_limit: Optional[float] = None,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")
        self.frame_stride = max(1, int(frame_stride))
        self.n_lookahead = max(0, int(n_lookahead))
        self.robot_frame = bool(robot_frame)
        self.robot_state = str(robot_state)
        self.turn_speed = None if turn_speed is None else float(turn_speed)
        self.heading_delta_limit = (
            None if heading_delta_limit is None else float(heading_delta_limit)
        )
        stride = k_lookahead if k_lookahead is not None else lookahead_stride
        if stride is None:
            stride = 5
        self.lookahead_stride = max(1, int(stride))

        if self.robot_frame and action_mode == "position":
            raise ValueError(
                "robot_frame=True is incompatible with action_mode='position' "
                "(absolute position is not robot-centric). Use 'delta' or 'velocity'."
            )
        if (action_mode == "forward_heading") and (not self.robot_frame):
            raise ValueError(
                "action_mode='forward_heading' requires robot_frame=True "
                "(forward/heading is robot-centric)."
            )

        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        episode_dirs = sorted([p for p in self.data_dir.iterdir() if p.is_dir()])
        if not episode_dirs:
            raise RuntimeError(f"No episode directories found under {self.data_dir}")

        loaded = 0
        for ep_dir in episode_dirs:
            traj_path = ep_dir / "trajectory.zarr"
            if not traj_path.exists():
                continue
            episode = self._load_episode(traj_path, action_mode)
            if episode is None:
                continue
            self.replay_buffer.add_episode(episode)
            loaded += 1

        if loaded == 0:
            raise RuntimeError(f"No valid episodes found under {self.data_dir}")

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def _load_episode(self, traj_path: Path, action_mode: str) -> Optional[Dict[str, np.ndarray]]:
        try:
            root = zarr.open(str(traj_path), mode="r")
        except Exception as exc:
            print(f"[warn] failed to open {traj_path}: {exc}")
            return None

        if "robot_path" not in root or "human_path" not in root:
            print(f"[warn] missing robot_path/human_path in {traj_path}")
            return None

        robot = np.asarray(root["robot_path"][:], dtype=np.float32)
        human = np.asarray(root["human_path"][:], dtype=np.float32)
        timestamps = None
        if "timestamps" in root:
            timestamps = np.asarray(root["timestamps"][:], dtype=np.float32)
        length = min(len(robot), len(human))
        if timestamps is not None:
            length = min(length, len(timestamps))
        if length < 2:
            print(f"[warn] episode too short in {traj_path} (len={length})")
            return None
        robot = robot[:length]
        human = human[:length]
        if timestamps is not None:
            timestamps = timestamps[:length]

        headings_full = self._compute_headings(robot)

        if self.frame_stride > 1:
            indices = np.arange(0, length, self.frame_stride)
            robot = robot[indices]
            human = human[indices]
            headings = headings_full[indices]
            if timestamps is not None:
                timestamps = timestamps[indices]
            length = min(len(robot), len(human), len(headings))
            if timestamps is not None:
                length = min(length, len(timestamps))
            if length < 2:
                print(f"[warn] episode too short after stride in {traj_path} (len={length})")
                return None
            robot = robot[:length]
            human = human[:length]
            headings = headings[:length]
            if timestamps is not None:
                timestamps = timestamps[:length]
        else:
            headings = headings_full

        ref_path = self._load_reference_path(traj_path)
        ref_features = self._build_reference_features(robot, headings, ref_path)

        robot_obs, human_obs = robot, human
        if self.robot_frame:
            human_obs = self._to_robot_frame(points=human, origin=robot, headings=headings)
            robot_obs = self._build_robot_state(
                robot=robot,
                headings=headings,
                timestamps=timestamps,
                mode=self.robot_state,
            )

        obs = np.concatenate([robot_obs, human_obs, ref_features], axis=-1).astype(np.float32)
        action = self._build_action(
            robot=robot,
            root=root,
            length=length,
            action_mode=action_mode,
            timestamps=timestamps,
            headings=headings if self.robot_frame else None,
        )
        return {"obs": obs, "action": action}

    def _load_reference_path(self, traj_path: Path) -> Optional[np.ndarray]:
        meta_path = traj_path.parent / "metadata.json"
        if not meta_path.exists():
            print(f"[warn] metadata.json not found for {traj_path}")
            return None
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:
            print(f"[warn] failed to load metadata {meta_path}: {exc}")
            return None
        ref = np.asarray(meta.get("reference_path", []), dtype=np.float32)
        if ref.ndim != 2 or ref.shape[1] != 2:
            print(f"[warn] invalid reference_path in {meta_path}")
            return None
        return ref

    def _compute_headings(self, robot: np.ndarray) -> np.ndarray:
        if len(robot) < 2:
            return np.zeros((len(robot),), dtype=np.float32)
        diffs = np.diff(robot, axis=0)
        headings = np.zeros((len(robot),), dtype=np.float32)
        headings[:-1] = np.arctan2(diffs[:, 1], diffs[:, 0]).astype(np.float32)
        headings[-1] = headings[-2]
        for i in range(1, len(robot)):
            if np.linalg.norm(diffs[i - 1]) < 1e-6:
                headings[i] = headings[i - 1]
        return headings

    def _wrap_angle(self, angles: np.ndarray) -> np.ndarray:
        return (angles + np.pi) % (2 * np.pi) - np.pi

    def _build_robot_state(
        self,
        robot: np.ndarray,
        headings: np.ndarray,
        timestamps: Optional[np.ndarray],
        mode: str,
    ) -> np.ndarray:
        mode = str(mode).lower()
        if mode in ("zero", "zeros", "none"):
            return np.zeros_like(robot, dtype=np.float32)
        if mode not in ("vel", "velocity"):
            raise ValueError(f"Unsupported robot_state: {mode}")

        if len(robot) < 2:
            return np.zeros_like(robot, dtype=np.float32)

        diffs = np.zeros_like(robot, dtype=np.float32)
        diffs[1:] = (robot[1:] - robot[:-1]).astype(np.float32)
        diffs[0] = diffs[1]

        if timestamps is not None and len(timestamps) == len(robot):
            dt = np.zeros((len(robot),), dtype=np.float32)
            dt[1:] = np.diff(timestamps).astype(np.float32)
            dt[0] = dt[1]
            dt = np.where(dt <= 1e-6, 1e-6, dt)
            vel_world = diffs / dt[:, None]
        else:
            vel_world = diffs

        cos_h = np.cos(headings).astype(np.float32)
        sin_h = np.sin(headings).astype(np.float32)
        vx = cos_h * vel_world[:, 0] + sin_h * vel_world[:, 1]
        vy = -sin_h * vel_world[:, 0] + cos_h * vel_world[:, 1]
        return np.stack([vx, vy], axis=-1).astype(np.float32)

    def _to_robot_frame(
        self,
        points: np.ndarray,
        origin: np.ndarray,
        headings: np.ndarray,
    ) -> np.ndarray:
        rel = (points - origin).astype(np.float32)
        cos_h = np.cos(headings).astype(np.float32)
        sin_h = np.sin(headings).astype(np.float32)
        x = cos_h * rel[:, 0] + sin_h * rel[:, 1]
        y = -sin_h * rel[:, 0] + cos_h * rel[:, 1]
        return np.stack([x, y], axis=-1).astype(np.float32)

    def _build_reference_features(
        self,
        robot: np.ndarray,
        headings: np.ndarray,
        ref_path: Optional[np.ndarray],
    ) -> np.ndarray:
        length = len(robot)
        if self.n_lookahead <= 0:
            return np.zeros((length, 0), dtype=np.float32)
        if ref_path is None or len(ref_path) == 0:
            return np.zeros((length, self.n_lookahead * 2), dtype=np.float32)

        ref_path = ref_path.astype(np.float32)
        n_ref = len(ref_path)
        features = np.zeros((length, self.n_lookahead, 2), dtype=np.float32)
        for t in range(length):
            pos = robot[t]
            diffs = ref_path - pos
            idx = int(np.argmin(np.sum(diffs * diffs, axis=1)))
            indices = idx + np.arange(self.n_lookahead) * self.lookahead_stride
            indices = np.clip(indices, 0, n_ref - 1)
            points = ref_path[indices]
            rel = points - pos
            cos_h = float(np.cos(headings[t]))
            sin_h = float(np.sin(headings[t]))
            features[t, :, 0] = cos_h * rel[:, 0] + sin_h * rel[:, 1]
            features[t, :, 1] = -sin_h * rel[:, 0] + cos_h * rel[:, 1]
        return features.reshape(length, -1)

    def _build_action(
        self,
        robot: np.ndarray,
        root: zarr.Group,
        length: int,
        action_mode: str,
        timestamps: Optional[np.ndarray] = None,
        headings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if action_mode == "delta":
            delta = robot[1:] - robot[:-1]
            last = delta[-1:] if delta.shape[0] > 0 else np.zeros((1, 2), dtype=np.float32)
            action = np.concatenate([delta, last], axis=0).astype(np.float32)
        elif action_mode == "forward_heading":
            if headings is None:
                raise ValueError("action_mode='forward_heading' requires headings")
            delta = robot[1:] - robot[:-1]
            cos_h = np.cos(headings[:-1]).astype(np.float32)
            sin_h = np.sin(headings[:-1]).astype(np.float32)
            forward = cos_h * delta[:, 0] + sin_h * delta[:, 1]
            heading_delta = self._wrap_angle(headings[1:] - headings[:-1]).astype(np.float32)
            if self.heading_delta_limit is not None:
                limit = float(self.heading_delta_limit)
                heading_delta = np.clip(heading_delta, -limit, limit)
            elif (self.turn_speed is not None) and (timestamps is not None) and (len(timestamps) >= 2):
                dt = np.diff(timestamps[:len(headings)]).astype(np.float32)
                dt = np.where(dt <= 1e-6, 1e-6, dt)
                max_delta = float(self.turn_speed) * dt
                heading_delta = np.clip(heading_delta, -max_delta, max_delta)
            last_forward = forward[-1:] if forward.shape[0] > 0 else np.zeros((1,), dtype=np.float32)
            last_heading = (
                heading_delta[-1:] if heading_delta.shape[0] > 0 else np.zeros((1,), dtype=np.float32)
            )
            forward_full = np.concatenate([forward, last_forward], axis=0)
            heading_full = np.concatenate([heading_delta, last_heading], axis=0)
            action = np.stack([forward_full, heading_full], axis=-1).astype(np.float32)
        elif action_mode == "position":
            action = robot.astype(np.float32)
        elif action_mode == "velocity":
            if timestamps is None:
                if "timestamps" not in root:
                    raise ValueError("action_mode='velocity' requires timestamps in trajectory.zarr")
                timestamps = np.asarray(root["timestamps"][:length], dtype=np.float32)
            if timestamps is None:
                raise ValueError("action_mode='velocity' requires timestamps in trajectory.zarr")
            if len(timestamps) < 2:
                raise ValueError("timestamps too short for velocity action")
            dt = np.diff(timestamps)
            dt = np.where(dt <= 1e-6, 1e-6, dt)
            vel = (robot[1:] - robot[:-1]) / dt[:, None]
            last = vel[-1:] if vel.shape[0] > 0 else np.zeros((1, 2), dtype=np.float32)
            action = np.concatenate([vel, last], axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported action_mode: {action_mode}")

        if headings is not None:
            if len(headings) != length:
                raise ValueError("headings length mismatch")
            if action_mode in ("delta", "velocity"):
                # rotate per-timestep action into robot/body frame
                cos_h = np.cos(headings).astype(np.float32)
                sin_h = np.sin(headings).astype(np.float32)
                x = cos_h * action[:, 0] + sin_h * action[:, 1]
                y = -sin_h * action[:, 0] + cos_h * action[:, 1]
                action = np.stack([x, y], axis=-1).astype(np.float32)

        if action.shape[0] != length:
            action = action[:length]
        return action

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        data = {
            "obs": self.replay_buffer["obs"],
            "action": self.replay_buffer["action"],
        }
        if "range_eps" not in kwargs:
            # avoid blowing up dims that barely change
            kwargs["range_eps"] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(sample, torch.from_numpy)
        return torch_data
