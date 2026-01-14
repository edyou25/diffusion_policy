import os
import sys
import warnings

import numpy as np

# Ignore pkg_resources deprecation warnings (from llvmlite)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.dataset.guide_lowdim_dataset import GuideLowdimDataset


def test():
    data_dir = os.environ.get("GUIDE_DATA_DIR", "/home/yyf/IROS2026/FollowDataset/data")
    if not os.path.exists(data_dir):
        print(f"[warn] data_dir not found: {data_dir}")
        return

    print("=" * 60)
    print("Test dataset - coordinate frame check")
    print("=" * 60)

    # Test robot-centric frame (default)
    dataset = GuideLowdimDataset(
        data_dir=data_dir,
        horizon=16,
        pad_before=1,
        pad_after=7,
        val_ratio=0.05,
        max_train_episodes=10,
        robot_frame=True,  # explicitly use robot-centric frame
    )

    print(f"\nrobot_frame: {dataset.robot_frame}")
    print(f"robot_state: {dataset.robot_state}")
    print(f"n_lookahead: {dataset.n_lookahead}")
    print(f"train size: {len(dataset)}")

    sample = dataset[0]
    obs = sample["obs"].numpy()
    action = sample["action"].numpy()

    print(f"obs shape: {obs.shape}")
    print(f"action shape: {action.shape}")

    # Check robot-centric observation structure
    robot_obs = obs[:, :2]  # robot observation (first 2 dims)
    human_obs = obs[:, 2:4]  # human observation (3rd-4th dims)
    lookahead_obs = obs[:, 4:] if obs.shape[1] > 4 else None  # lookahead features (remaining dims)

    print("\n[Coordinate frame check]")
    robot_state_mode = dataset.robot_state.lower()
    
    if robot_state_mode in ("zero", "zeros", "none"):
        # Robot state is zero (robot at origin)
        print(f"robot obs range: [{robot_obs.min():.6f}, {robot_obs.max():.6f}]")
        if abs(robot_obs.max()) < 1e-5 and abs(robot_obs.min()) < 1e-5:
            print("OK: robot obs is all zeros, consistent with robot at origin in robot frame.")
        else:
            print("WARN: robot obs is not zero, but robot_state='zero' was expected.")
    elif robot_state_mode in ("vel", "velocity"):
        # Robot state is velocity in robot frame
        print(f"robot obs (velocity) range: [{robot_obs.min():.4f}, {robot_obs.max():.4f}]")
        speed = np.linalg.norm(robot_obs, axis=1)
        print(f"robot speed range: [{speed.min():.4f}, {speed.max():.4f}] m/s")
        print(f"robot speed mean: {speed.mean():.4f} m/s")
        print("OK: robot obs encodes velocity in robot frame (vx, vy).")
    else:
        print(f"robot obs range: [{robot_obs.min():.6f}, {robot_obs.max():.6f}]")
        print(f"WARN: unknown robot_state mode: {robot_state_mode}")

    print(f"\nhuman obs range: [{human_obs.min():.2f}, {human_obs.max():.2f}]")
    print("  (human position is relative to robot and rotated into robot heading)")

    # Check lookahead/reference path features
    if lookahead_obs is not None and lookahead_obs.size > 0:
        print(f"\n[Lookahead features check]")
        print(f"lookahead features shape: {lookahead_obs.shape}")
        print(f"expected shape: (horizon, {dataset.n_lookahead * 2})")
        
        if lookahead_obs.shape[1] == dataset.n_lookahead * 2:
            # Reshape to (horizon, n_lookahead, 2)
            lookahead_reshaped = lookahead_obs.reshape(-1, dataset.n_lookahead, 2)
            
            # Check first timestep
            first_lookahead = lookahead_reshaped[0]  # (n_lookahead, 2)
            print(f"first timestep lookahead range: [{first_lookahead.min():.2f}, {first_lookahead.max():.2f}]")
            
            # Check distances from robot (should be reasonable, not too large)
            distances = np.linalg.norm(lookahead_reshaped, axis=2)  # (horizon, n_lookahead)
            print(f"lookahead distances range: [{distances.min():.2f}, {distances.max():.2f}] m")
            print(f"lookahead distances mean: {distances.mean():.2f} m")
            
            # Check if lookahead points are in robot frame
            # In robot frame, the first point should typically be ahead (positive x)
            # and points should generally be in front/forward direction
            first_point = lookahead_reshaped[0, 0]  # first lookahead point
            print(f"first lookahead point: [{first_point[0]:.2f}, {first_point[1]:.2f}]")
            
            # Check if most points are in forward direction (positive x in robot frame)
            forward_points = np.sum(lookahead_reshaped[:, :, 0] > 0)  # x > 0 means forward
            total_points = lookahead_reshaped.shape[0] * lookahead_reshaped.shape[1]
            forward_ratio = forward_points / total_points
            print(f"forward points ratio: {forward_ratio:.2%} (x > 0 in robot frame)")
            
            if forward_ratio > 0.5:
                print("OK: most lookahead points are in forward direction (robot frame).")
            else:
                print("WARN: many lookahead points are behind robot, may indicate coordinate issue.")
            
            # Check consistency: lookahead points should be relative to robot
            # The first point should be closest to robot (smallest distance)
            first_dist = np.linalg.norm(first_lookahead[0])
            last_dist = np.linalg.norm(first_lookahead[-1])
            if first_dist < last_dist:
                print("OK: lookahead points are ordered by distance (closest first).")
            else:
                print("WARN: lookahead points ordering may be incorrect.")
        else:
            print(f"WARN: lookahead features shape mismatch!")
    else:
        print(f"\n[Lookahead features check]")
        print("No lookahead features (n_lookahead=0 or obs_dim <= 4)")

    val_dataset = dataset.get_validation_dataset()
    print(f"\nval size: {len(val_dataset)}")

    normalizer = dataset.get_normalizer()
    print(f"normalizer keys: {list(normalizer.params_dict.keys())}")

    print("\n" + "=" * 60)
    print("Conclusion: dataset uses robot-centric frame (robot_frame=True)")
    print("=" * 60)


if __name__ == "__main__":
    test()
