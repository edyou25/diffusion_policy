"""
Guide Mid360 runner for evaluating guide-follow policies on validation episodes.
"""

from typing import Dict, Optional
from pathlib import Path

import numpy as np
import torch
import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.guide_mid360_dataset import GuideMid360Dataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class GuideMid360Runner(BaseLowdimRunner):
    def __init__(
        self,
        output_dir,
        data_dir: str,
        n_test: int = 50,
        n_test_vis: int = 4,
        test_start_seed: int = 100000,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        pad_before: int = 1,
        pad_after: int = 7,
        val_ratio: float = 0.05,
        action_mode: str = "forward_heading",
        n_lookahead: int = 20,
        k_lookahead: int = 5,
        lookahead_stride: Optional[int] = None,
        frame_stride: int = 1,
        robot_frame: bool = True,
        lidar_num_bins: int = 128,
        lidar_min_angle: float = -np.pi,
        lidar_max_angle: float = np.pi,
        lidar_min_range: float = 0.2,
        lidar_max_range: float = 8.0,
        lidar_ground_height: float = 0.10,
        lidar_max_height: float = 2.20,
        lidar_use_world_height: bool = True,
        tqdm_interval_sec: float = 5.0,
        **kwargs,
    ):
        super().__init__(output_dir)

        self.val_dataset = GuideMid360Dataset(
            data_dir=data_dir,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            seed=42,
            val_ratio=val_ratio,
            max_train_episodes=None,
            action_mode=action_mode,
            n_lookahead=n_lookahead,
            k_lookahead=k_lookahead,
            lookahead_stride=lookahead_stride,
            frame_stride=frame_stride,
            robot_frame=robot_frame,
            lidar_num_bins=lidar_num_bins,
            lidar_min_angle=lidar_min_angle,
            lidar_max_angle=lidar_max_angle,
            lidar_min_range=lidar_min_range,
            lidar_max_range=lidar_max_range,
            lidar_ground_height=lidar_ground_height,
            lidar_max_height=lidar_max_height,
            lidar_use_world_height=lidar_use_world_height,
        )

        val_mask = ~self.val_dataset.train_mask
        val_episode_indices = np.where(val_mask)[0]

        np.random.seed(test_start_seed)
        n_available = len(val_episode_indices)
        n_test = min(n_test, n_available)
        selected_indices = np.random.choice(
            val_episode_indices,
            size=n_test,
            replace=False,
        )

        self.test_episode_indices = selected_indices
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_mode = action_mode
        self.robot_frame = bool(robot_frame)
        self.tqdm_interval_sec = tqdm_interval_sec

        self.media_dir = Path(output_dir) / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)

    def _compute_trajectory_error(
        self,
        pred_actions: np.ndarray,
        gt_actions: np.ndarray,
        initial_robot_pos: np.ndarray,
    ) -> Dict[str, float]:
        if self.action_mode == "delta":
            pred_traj = initial_robot_pos[None, :] + np.cumsum(pred_actions, axis=0)
            gt_traj = initial_robot_pos[None, :] + np.cumsum(gt_actions, axis=0)
        elif self.action_mode == "forward_heading":
            pred_traj = self._integrate_forward_heading(pred_actions, initial_robot_pos)
            gt_traj = self._integrate_forward_heading(gt_actions, initial_robot_pos)
        elif self.action_mode == "position":
            pred_traj = pred_actions
            gt_traj = gt_actions
        else:
            pred_traj = initial_robot_pos[None, :] + np.cumsum(pred_actions, axis=0)
            gt_traj = initial_robot_pos[None, :] + np.cumsum(gt_actions, axis=0)

        position_errors = np.linalg.norm(pred_traj - gt_traj, axis=-1)
        action_errors = np.linalg.norm(pred_actions - gt_actions, axis=-1)

        return {
            "mean_position_error": float(np.mean(position_errors)),
            "max_position_error": float(np.max(position_errors)),
            "final_position_error": float(position_errors[-1]),
            "mean_action_error": float(np.mean(action_errors)),
            "max_action_error": float(np.max(action_errors)),
        }

    def _integrate_forward_heading(
        self,
        actions: np.ndarray,
        initial_robot_pos: np.ndarray,
        initial_heading: float = 0.0,
    ) -> np.ndarray:
        pos = np.asarray(initial_robot_pos, dtype=np.float32).copy()
        heading = float(initial_heading)
        traj = np.zeros((len(actions), 2), dtype=np.float32)
        for i, act in enumerate(actions):
            forward = float(act[0])
            heading += float(act[1])
            pos = pos + np.array([np.cos(heading), np.sin(heading)], dtype=np.float32) * forward
            traj[i] = pos
        return traj

    def run(self, policy: BaseLowdimPolicy) -> Dict:
        device = policy.device
        dtype = policy.dtype
        policy.eval()

        all_metrics = []
        all_position_errors = []
        all_action_errors = []

        replay_buffer = self.val_dataset.replay_buffer

        with torch.no_grad():
            pbar = tqdm.tqdm(
                self.test_episode_indices[: self.n_test],
                desc="Evaluating Guide Mid360 Policy",
                mininterval=self.tqdm_interval_sec,
            )

            for ep_idx in pbar:
                episode = replay_buffer.get_episode(ep_idx)
                obs_episode = episode["obs"]
                action_episode = episode["action"]

                episode_errors = []
                episode_length = len(obs_episode)
                n_windows = min(10, max(1, episode_length - self.horizon))
                window_starts = np.linspace(
                    0,
                    max(0, episode_length - self.horizon - 1),
                    n_windows,
                    dtype=int,
                )

                for start_idx in window_starts:
                    end_idx = min(start_idx + self.horizon, episode_length)
                    window_length = end_idx - start_idx
                    if window_length < self.n_obs_steps:
                        continue

                    obs_window = obs_episode[start_idx : start_idx + self.n_obs_steps]
                    obs_tensor = torch.from_numpy(obs_window).to(
                        device=device, dtype=dtype
                    ).unsqueeze(0)

                    action_start = start_idx + self.n_obs_steps - 1
                    action_end = min(action_start + self.n_action_steps, episode_length)
                    gt_actions = action_episode[action_start:action_end]

                    obs_dict = {"obs": obs_tensor}
                    action_dict = policy.predict_action(obs_dict)
                    pred_actions = action_dict["action"].cpu().numpy()[0]

                    if len(pred_actions) > len(gt_actions):
                        pred_actions = pred_actions[: len(gt_actions)]
                    elif len(pred_actions) < len(gt_actions):
                        gt_actions = gt_actions[: len(pred_actions)]

                    initial_robot_pos = (
                        np.zeros((2,), dtype=np.float32)
                        if self.robot_frame
                        else obs_episode[action_start, :2]
                    )
                    metrics = self._compute_trajectory_error(
                        pred_actions=pred_actions,
                        gt_actions=gt_actions,
                        initial_robot_pos=initial_robot_pos,
                    )
                    episode_errors.append(metrics)

                if len(episode_errors) == 0:
                    continue

                mean_metrics = {
                    key: float(np.mean([m[key] for m in episode_errors]))
                    for key in episode_errors[0].keys()
                }
                all_metrics.append(mean_metrics)
                all_position_errors.append(mean_metrics["mean_position_error"])
                all_action_errors.append(mean_metrics["mean_action_error"])

                pbar.set_postfix(
                    pos_err=f"{mean_metrics['mean_position_error']:.3f}",
                    act_err=f"{mean_metrics['mean_action_error']:.3f}",
                )

        if len(all_metrics) == 0:
            return {"test_mean_score": 0.0}

        agg_metrics = {
            f"test_{key}": float(np.mean([m[key] for m in all_metrics]))
            for key in all_metrics[0].keys()
        }
        agg_metrics["test_mean_score"] = float(
            1.0 / (1.0 + np.mean(all_position_errors) + 0.5 * np.mean(all_action_errors))
        )
        return agg_metrics
