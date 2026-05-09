from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class GuideMid360ObservationConfig:
    num_bins: int = 128
    min_angle: float = -np.pi
    max_angle: float = np.pi
    min_range: float = 0.2
    max_range: float = 8.0
    ground_height: float = 0.10
    max_height: float = 2.20
    use_world_height: bool = True
    empty_range: float | None = None

    @property
    def scan_dim(self) -> int:
        return int(self.num_bins)

    @property
    def fill_value(self) -> float:
        if self.empty_range is None:
            return float(self.max_range)
        return float(self.empty_range)

    def cache_key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def cache_path(self, episode_dir: str | Path) -> Path:
        episode_dir = Path(episode_dir)
        return episode_dir / f"mid360_scan_cache_{self.cache_key()}.npy"


def resolve_pointcloud_field_indices(field_names: Sequence[str]) -> dict[str, int]:
    return {str(name): idx for idx, name in enumerate(field_names)}


def encode_mid360_scan_from_pointcloud(
    frame: np.ndarray,
    field_names: Sequence[str],
    config: GuideMid360ObservationConfig,
    *,
    mid360_pose: np.ndarray | None = None,
) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    indices = resolve_pointcloud_field_indices(field_names)
    if "x" not in indices or "y" not in indices:
        raise ValueError(f"Point cloud fields must contain x/y, got {list(field_names)}")
    ix = indices["x"]
    iy = indices["y"]
    iz = indices.get("z")

    local_xy = frame[:, [ix, iy]] if frame.ndim == 2 and len(frame) > 0 else np.zeros((0, 2), dtype=np.float32)
    azimuth = np.arctan2(local_xy[:, 1], local_xy[:, 0]).astype(np.float32, copy=False)
    ranges = np.linalg.norm(local_xy, axis=-1).astype(np.float32, copy=False)

    height = None
    if iz is not None and frame.shape[1] > iz:
        local_z = frame[:, iz].astype(np.float32, copy=False)
        if config.use_world_height and mid360_pose is not None and len(mid360_pose) >= 7:
            local_xyz = frame[:, [ix, iy, iz]].astype(np.float32, copy=False)
            pose = np.asarray(mid360_pose, dtype=np.float64)
            rot = Rotation.from_quat(pose[3:7]).as_matrix().astype(np.float32)
            height = local_xyz @ rot[2, :].astype(np.float32) + np.float32(pose[2])
        else:
            height = local_z

    return encode_mid360_scan_from_local_points(
        local_xy=local_xy,
        config=config,
        ranges=ranges,
        azimuth=azimuth,
        height=height,
    )


def encode_mid360_scan_from_local_points(
    local_xy: np.ndarray,
    config: GuideMid360ObservationConfig,
    *,
    ranges: np.ndarray | None = None,
    azimuth: np.ndarray | None = None,
    height: np.ndarray | None = None,
) -> np.ndarray:
    local_xy = np.asarray(local_xy, dtype=np.float32)
    if local_xy.ndim != 2 or local_xy.shape[1] < 2:
        return np.full((config.scan_dim,), config.fill_value, dtype=np.float32)

    if ranges is None:
        ranges = np.linalg.norm(local_xy[:, :2], axis=-1).astype(np.float32, copy=False)
    else:
        ranges = np.asarray(ranges, dtype=np.float32)
    if azimuth is None:
        azimuth = np.arctan2(local_xy[:, 1], local_xy[:, 0]).astype(np.float32, copy=False)
    else:
        azimuth = np.asarray(azimuth, dtype=np.float32)

    valid = np.isfinite(local_xy[:, 0]) & np.isfinite(local_xy[:, 1])
    valid &= np.isfinite(ranges) & np.isfinite(azimuth)
    valid &= ranges >= float(config.min_range)
    valid &= ranges <= float(config.max_range)
    valid &= _angle_mask(azimuth, config.min_angle, config.max_angle)

    if height is not None:
        height = np.asarray(height, dtype=np.float32)
        valid &= np.isfinite(height)
        valid &= height >= float(config.ground_height)
        valid &= height <= float(config.max_height)

    scan = np.full((config.scan_dim,), config.fill_value, dtype=np.float32)
    if not np.any(valid):
        return scan

    bin_idx = azimuth_to_bin_indices(azimuth[valid], config)
    np.minimum.at(scan, bin_idx, ranges[valid].astype(np.float32, copy=False))
    return scan


def azimuth_to_bin_indices(
    azimuth: np.ndarray | Sequence[float],
    config: GuideMid360ObservationConfig,
) -> np.ndarray:
    azimuth = np.asarray(azimuth, dtype=np.float32)
    width = float(config.max_angle - config.min_angle)
    if width <= 0.0:
        raise ValueError("max_angle must be larger than min_angle")
    normalized = (azimuth - float(config.min_angle)) / width
    indices = np.floor(normalized * float(config.scan_dim)).astype(np.int64)
    return np.clip(indices, 0, int(config.scan_dim) - 1)


def filter_runtime_episode_dirs(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.is_dir() and not p.name.startswith("_")]


def _angle_mask(azimuth: np.ndarray, min_angle: float, max_angle: float) -> np.ndarray:
    width = float(max_angle - min_angle)
    if width >= (2.0 * np.pi - 1e-6):
        return np.ones_like(azimuth, dtype=bool)
    return (azimuth >= float(min_angle)) & (azimuth < float(max_angle))
