"""
Guide Lowdim Runner for evaluating guide-follow policies on validation dataset.
This runner evaluates the policy by comparing predicted robot trajectories 
with ground truth trajectories from the validation set.
"""

from typing import Dict, Optional
import numpy as np
import torch
import tqdm
from pathlib import Path

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.dataset.guide_lowdim_dataset import GuideLowdimDataset


class GuideLowdimRunner(BaseLowdimRunner):
    """
    Evaluates guide-follow policy on validation dataset.
    Computes metrics by comparing predicted actions with ground truth.
    """
    
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
        n_obstacle_circles: int = 0,
        n_obstacle_segments: int = 0,
        obstacle_include_radius: bool = True,
        tqdm_interval_sec: float = 5.0,
        **kwargs
    ):
        """
        Args:
            output_dir: Output directory for saving results
            data_dir: Path to guide dataset directory
            n_test: Number of test episodes to evaluate
            n_test_vis: Number of episodes to visualize (save trajectories)
            test_start_seed: Starting seed for test episodes
            horizon: Prediction horizon
            n_obs_steps: Number of observation steps
            n_action_steps: Number of action steps
            pad_before: Padding before sequence
            pad_after: Padding after sequence
            val_ratio: Validation ratio (should match training)
            action_mode: Action mode (delta, forward_heading, position, velocity)
            tqdm_interval_sec: Progress bar update interval
        """
        super().__init__(output_dir)
        
        # Load validation dataset
        self.val_dataset = GuideLowdimDataset(
            data_dir=data_dir,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            seed=42,  # Same seed as training
            val_ratio=val_ratio,
            max_train_episodes=None,  # Use all episodes for validation
            action_mode=action_mode,
            n_lookahead=n_lookahead,
            k_lookahead=k_lookahead,
            lookahead_stride=lookahead_stride,
            frame_stride=frame_stride,
            robot_frame=robot_frame,
            n_obstacle_circles=n_obstacle_circles,
            n_obstacle_segments=n_obstacle_segments,
            obstacle_include_radius=obstacle_include_radius,
        )
        
        # Get validation episodes from replay buffer
        val_mask = ~self.val_dataset.train_mask
        val_episode_indices = np.where(val_mask)[0]
        
        # Select test episodes
        np.random.seed(test_start_seed)
        n_available = len(val_episode_indices)
        n_test = min(n_test, n_available)
        selected_indices = np.random.choice(
            val_episode_indices, 
            size=n_test, 
            replace=False
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
        
        # Create output directory for visualizations
        self.media_dir = Path(output_dir) / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_trajectory_error(
        self, 
        pred_actions: np.ndarray, 
        gt_actions: np.ndarray,
        initial_robot_pos: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute trajectory errors between predicted and ground truth.
        
        Args:
            pred_actions: Predicted actions (T, 2)
            gt_actions: Ground truth actions (T, 2)
            initial_robot_pos: Initial robot position (2,)
            
        Returns:
            Dictionary of error metrics
        """
        # Reconstruct trajectories from actions
        if self.action_mode == "delta":
            # Cumulative sum for delta actions (no extra initial point)
            pred_traj = initial_robot_pos[None, :] + np.cumsum(pred_actions, axis=0)
            gt_traj = initial_robot_pos[None, :] + np.cumsum(gt_actions, axis=0)
        elif self.action_mode == "forward_heading":
            pred_traj = self._integrate_forward_heading(pred_actions, initial_robot_pos)
            gt_traj = self._integrate_forward_heading(gt_actions, initial_robot_pos)
        elif self.action_mode == "position":
            pred_traj = pred_actions
            gt_traj = gt_actions
        else:  # velocity
            # For velocity, we'd need timestamps, so use delta approximation
            pred_traj = initial_robot_pos[None, :] + np.cumsum(pred_actions, axis=0)
            gt_traj = initial_robot_pos[None, :] + np.cumsum(gt_actions, axis=0)
        
        # Compute errors
        position_errors = np.linalg.norm(pred_traj - gt_traj, axis=-1)
        action_errors = np.linalg.norm(pred_actions - gt_actions, axis=-1)
        
        return {
            'mean_position_error': float(np.mean(position_errors)),
            'max_position_error': float(np.max(position_errors)),
            'final_position_error': float(position_errors[-1]),
            'mean_action_error': float(np.mean(action_errors)),
            'max_action_error': float(np.max(action_errors)),
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
        """
        Evaluate policy on validation dataset.
        
        Args:
            policy: Policy to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        device = policy.device
        dtype = policy.dtype
        policy.eval()
        
        all_metrics = []
        all_position_errors = []
        all_action_errors = []
        
        # Get replay buffer for accessing full episodes
        replay_buffer = self.val_dataset.replay_buffer
        
        with torch.no_grad():
            pbar = tqdm.tqdm(
                self.test_episode_indices[:self.n_test],
                desc="Evaluating Guide Policy",
                mininterval=self.tqdm_interval_sec
            )
            
            for ep_idx in pbar:
                # Get episode data
                episode = replay_buffer.get_episode(ep_idx)
                obs_episode = episode['obs']  # (T, D) - [robot, human, lookahead...]
                action_episode = episode['action']  # (T, 2)
                
                # Evaluate on sliding windows
                episode_errors = []
                episode_length = len(obs_episode)
                
                # Sample multiple windows from this episode
                n_windows = min(10, max(1, episode_length - self.horizon))
                window_starts = np.linspace(
                    0, 
                    max(0, episode_length - self.horizon - 1),
                    n_windows,
                    dtype=int
                )
                
                for start_idx in window_starts:
                    end_idx = min(start_idx + self.horizon, episode_length)
                    window_length = end_idx - start_idx
                    
                    if window_length < self.n_obs_steps:
                        continue
                    
                    # Get observation window
                    obs_window = obs_episode[start_idx:start_idx+self.n_obs_steps]
                    obs_tensor = torch.from_numpy(obs_window).to(
                        device=device, dtype=dtype
                    ).unsqueeze(0)  # (1, n_obs_steps, 4)
                    
                    # Get ground truth actions for this window
                    action_start = start_idx + self.n_obs_steps - 1
                    action_end = min(action_start + self.n_action_steps, episode_length)
                    gt_actions = action_episode[action_start:action_end]
                    
                    # Predict actions
                    obs_dict = {'obs': obs_tensor}
                    action_dict = policy.predict_action(obs_dict)
                    pred_actions = action_dict['action'].cpu().numpy()[0]  # (n_action_steps, 2)
                    
                    # Truncate if needed
                    if len(pred_actions) > len(gt_actions):
                        pred_actions = pred_actions[:len(gt_actions)]
                    elif len(pred_actions) < len(gt_actions):
                        gt_actions = gt_actions[:len(pred_actions)]
                    
                    # Compute errors
                    initial_robot_pos = np.zeros((2,), dtype=np.float32) if self.robot_frame else obs_episode[action_start, :2]
                    errors = self._compute_trajectory_error(
                        pred_actions, gt_actions, initial_robot_pos
                    )
                    episode_errors.append(errors)
                
                if len(episode_errors) > 0:
                    # Aggregate errors for this episode
                    ep_metrics = {
                        k: np.mean([e[k] for e in episode_errors])
                        for k in episode_errors[0].keys()
                    }
                    all_metrics.append(ep_metrics)
                    all_position_errors.append(ep_metrics['mean_position_error'])
                    all_action_errors.append(ep_metrics['mean_action_error'])
        
        # Aggregate metrics across all episodes
        if len(all_metrics) == 0:
            return {
                'test/mean_score': 0.0,
                'test/mean_position_error': float('inf'),
                'test/mean_action_error': float('inf'),
            }
        
        # Compute aggregate statistics
        mean_position_error = np.mean(all_position_errors)
        mean_action_error = np.mean(all_action_errors)
        
        # Convert error to score (lower error = higher score)
        # Use inverse of normalized error as score
        # Assuming reasonable error range, we use exponential decay
        max_reasonable_error = 1.0  # Adjust based on your data scale
        position_score = np.exp(-mean_position_error / max_reasonable_error)
        
        result = {
            'test/mean_score': float(position_score),
            'test/mean_position_error': float(mean_position_error),
            'test/mean_action_error': float(mean_action_error),
            'test/max_position_error': float(np.max(all_position_errors)),
            'test/min_position_error': float(np.min(all_position_errors)),
            'test/std_position_error': float(np.std(all_position_errors)),
        }
        
        return result
