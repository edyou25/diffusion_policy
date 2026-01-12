from typing import Dict, Optional
import copy
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
        action_mode: str = "delta",
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")

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
        length = min(len(robot), len(human))
        if length < 2:
            print(f"[warn] episode too short in {traj_path} (len={length})")
            return None
        robot = robot[:length]
        human = human[:length]

        obs = np.concatenate([robot, human], axis=-1).astype(np.float32)
        action = self._build_action(robot, root, length, action_mode)
        return {"obs": obs, "action": action}

    def _build_action(
        self,
        robot: np.ndarray,
        root: zarr.Group,
        length: int,
        action_mode: str,
    ) -> np.ndarray:
        if action_mode == "delta":
            delta = robot[1:] - robot[:-1]
            last = delta[-1:] if delta.shape[0] > 0 else np.zeros((1, 2), dtype=np.float32)
            action = np.concatenate([delta, last], axis=0).astype(np.float32)
        elif action_mode == "position":
            action = robot.astype(np.float32)
        elif action_mode == "velocity":
            if "timestamps" not in root:
                raise ValueError("action_mode='velocity' requires timestamps in trajectory.zarr")
            timestamps = np.asarray(root["timestamps"][:length], dtype=np.float32)
            if len(timestamps) < 2:
                raise ValueError("timestamps too short for velocity action")
            dt = np.diff(timestamps)
            dt = np.where(dt <= 1e-6, 1e-6, dt)
            vel = (robot[1:] - robot[:-1]) / dt[:, None]
            last = vel[-1:] if vel.shape[0] > 0 else np.zeros((1, 2), dtype=np.float32)
            action = np.concatenate([vel, last], axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported action_mode: {action_mode}")

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
