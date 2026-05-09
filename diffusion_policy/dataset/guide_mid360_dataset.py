from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import zarr

from diffusion_policy.common.guide_mid360 import (
    GuideMid360ObservationConfig,
    encode_mid360_scan_from_pointcloud,
)
from diffusion_policy.dataset.guide_lowdim_dataset import GuideLowdimDataset


class GuideMid360Dataset(GuideLowdimDataset):
    """Guide dataset with Mid360 point cloud encoded as a fixed-length range scan."""

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
        lidar_num_bins: int = 128,
        lidar_min_angle: float = -np.pi,
        lidar_max_angle: float = np.pi,
        lidar_min_range: float = 0.2,
        lidar_max_range: float = 8.0,
        lidar_ground_height: float = 0.10,
        lidar_max_height: float = 2.20,
        lidar_use_world_height: bool = True,
        lidar_cache: bool = True,
    ):
        self.mid360_obs_config = GuideMid360ObservationConfig(
            num_bins=int(lidar_num_bins),
            min_angle=float(lidar_min_angle),
            max_angle=float(lidar_max_angle),
            min_range=float(lidar_min_range),
            max_range=float(lidar_max_range),
            ground_height=float(lidar_ground_height),
            max_height=float(lidar_max_height),
            use_world_height=bool(lidar_use_world_height),
        )
        self.lidar_num_bins = int(lidar_num_bins)
        self.lidar_cache = bool(lidar_cache)

        super().__init__(
            data_dir=data_dir,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            action_mode=action_mode,
            frame_stride=frame_stride,
            n_lookahead=n_lookahead,
            k_lookahead=k_lookahead,
            lookahead_stride=lookahead_stride,
            robot_frame=robot_frame,
            robot_state=robot_state,
            turn_speed=turn_speed,
            heading_delta_limit=heading_delta_limit,
            n_obstacle_circles=0,
            n_obstacle_segments=0,
            obstacle_include_radius=False,
            obstacle_include_human_clearance=False,
            human_radius=0.3,
            segment_repr="closest_dir",
        )

    def _load_episode(self, traj_path: Path, action_mode: str) -> Optional[Dict[str, np.ndarray]]:
        try:
            root = zarr.open(str(traj_path), mode="r")
        except Exception as exc:
            print(f"[warn] failed to open {traj_path}: {exc}")
            return None

        if "robot_path" not in root or "human_path" not in root:
            print(f"[warn] missing robot_path/human_path in {traj_path}")
            return None
        if "point_cloud_values" not in root or "point_cloud_offsets" not in root or "point_cloud_sizes" not in root:
            print(f"[warn] missing Mid360 point cloud datasets in {traj_path}")
            return None
        if "mid360_pose" not in root:
            print(f"[warn] missing mid360_pose in {traj_path}")
            return None

        robot = np.asarray(root["robot_path"][:], dtype=np.float32)
        human = np.asarray(root["human_path"][:], dtype=np.float32)
        timestamps = None
        if "timestamps" in root:
            timestamps = np.asarray(root["timestamps"][:], dtype=np.float32)

        scan_features = self._load_or_build_mid360_scan_features(root=root, traj_path=traj_path)
        length = min(len(robot), len(human), len(scan_features))
        if timestamps is not None:
            length = min(length, len(timestamps))
        if length < 2:
            print(f"[warn] episode too short in {traj_path} (len={length})")
            return None
        robot = robot[:length]
        human = human[:length]
        scan_features = scan_features[:length]
        if timestamps is not None:
            timestamps = timestamps[:length]

        headings_full = self._compute_headings(robot)
        if self.frame_stride > 1:
            indices = np.arange(0, length, self.frame_stride)
            robot = robot[indices]
            human = human[indices]
            headings = headings_full[indices]
            scan_features = scan_features[indices]
            if timestamps is not None:
                timestamps = timestamps[indices]
            length = min(len(robot), len(human), len(headings), len(scan_features))
            if timestamps is not None:
                length = min(length, len(timestamps))
            if length < 2:
                print(f"[warn] episode too short after stride in {traj_path} (len={length})")
                return None
            robot = robot[:length]
            human = human[:length]
            headings = headings[:length]
            scan_features = scan_features[:length]
            if timestamps is not None:
                timestamps = timestamps[:length]
        else:
            headings = headings_full

        meta, meta_path = self._load_metadata(traj_path)
        ref_path = self._load_reference_path(meta, meta_path)
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

        obs = np.concatenate(
            [robot_obs, human_obs, ref_features, scan_features],
            axis=-1,
        ).astype(np.float32)
        action = self._build_action(
            robot=robot,
            root=root,
            length=length,
            action_mode=action_mode,
            timestamps=timestamps,
            headings=headings if self.robot_frame else None,
        )
        return {"obs": obs, "action": action}

    def _load_or_build_mid360_scan_features(
        self,
        root: zarr.Group,
        traj_path: Path,
    ) -> np.ndarray:
        cache_path = self.mid360_obs_config.cache_path(traj_path.parent)
        if self.lidar_cache and cache_path.exists():
            try:
                cached = np.load(cache_path)
                if cached.ndim == 2 and cached.shape[1] == self.mid360_obs_config.scan_dim:
                    return cached.astype(np.float32, copy=False)
            except Exception:
                pass

        field_names = list(root.attrs.get("point_cloud_fields", []))
        if not field_names:
            raise ValueError(f"Missing point_cloud_fields attribute in {traj_path}")

        offsets = np.asarray(root["point_cloud_offsets"][:], dtype=np.int64)
        sizes = np.asarray(root["point_cloud_sizes"][:], dtype=np.int64)
        poses = np.asarray(root["mid360_pose"][:], dtype=np.float64)
        values = root["point_cloud_values"]
        n_frames = min(len(offsets), len(sizes), len(poses))
        features = np.full(
            (n_frames, self.mid360_obs_config.scan_dim),
            self.mid360_obs_config.fill_value,
            dtype=np.float32,
        )

        for frame_idx in range(n_frames):
            count = int(sizes[frame_idx])
            if count <= 0:
                continue
            end = int(offsets[frame_idx])
            start = end - count
            frame = np.asarray(values[start:end], dtype=np.float32)
            features[frame_idx] = encode_mid360_scan_from_pointcloud(
                frame=frame,
                field_names=field_names,
                config=self.mid360_obs_config,
                mid360_pose=poses[frame_idx],
            )

        if self.lidar_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, features)
        return features
