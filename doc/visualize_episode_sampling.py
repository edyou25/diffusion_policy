#!/usr/bin/env python3
"""
可视化数据集采样：展示16帧action序列与完整episode的关系

展示：
    - 完整episode的action轨迹
    - 从episode中采样出的16帧片段
    - 它们之间的对应关系
"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 添加项目根目录到路径
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer


def visualize_episode_sampling():
    """可视化episode采样过程"""
    
    root = pathlib.Path(__file__).parent.parent
    zarr_path = root / "data/pusht/pusht_cchi_v7_replay.zarr"
    
    print("=" * 60)
    print("加载数据...")
    
    # 1. 加载完整episode
    replay_buffer = ReplayBuffer.copy_from_path(
        str(zarr_path),
        keys=['state', 'action']
    )
    
    # 2. 创建数据集（会进行采样）
    dataset = PushTImageDataset(
        zarr_path=str(zarr_path),
        horizon=16,
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.02,
        max_train_episodes=90,
    )
    
    print(f"总 Episodes: {replay_buffer.n_episodes}")
    print(f"数据集大小: {len(dataset)} 个样本")
    
    # 3. 选择一个episode和对应的采样样本
    episode_idx = 0
    sample_idx = 0  # 选择第一个样本（通常来自第一个episode）
    
    # 获取完整episode
    episode = replay_buffer.get_episode(episode_idx)
    full_actions = episode['action']  # (T_full, 2)
    full_states = episode['state']  # (T_full, 5)
    full_agent_pos = full_states[:, :2]  # (T_full, 2)
    
    T_full = len(full_actions)
    print(f"\nEpisode {episode_idx}:")
    print(f"  完整长度: {T_full} 帧")
    
    # 获取采样样本
    sample = dataset[sample_idx]
    sampled_actions = sample['action'].numpy()  # (16, 2)
    sampled_agent_pos = sample['obs']['agent_pos'].numpy()  # (16, 2)
    
    T_sampled = len(sampled_actions)
    print(f"  采样长度: {T_sampled} 帧")
    
    # 4. 计算轨迹（通过累积action）
    # 完整episode轨迹
    full_trajectory = np.zeros_like(full_agent_pos)
    full_trajectory[0] = full_agent_pos[0]
    for t in range(1, T_full):
        full_trajectory[t] = full_trajectory[t-1] + full_actions[t-1]
    
    # 采样序列轨迹
    sampled_trajectory = np.zeros_like(sampled_agent_pos)
    sampled_trajectory[0] = sampled_agent_pos[0]
    for t in range(1, T_sampled):
        sampled_trajectory[t] = sampled_trajectory[t-1] + sampled_actions[t-1]
    
    # 5. 找到采样序列在完整episode中的位置
    # 通过匹配起始位置来找到对应关系
    start_pos_sampled = sampled_agent_pos[0]
    distances = np.linalg.norm(full_agent_pos - start_pos_sampled, axis=1)
    start_idx_in_episode = np.argmin(distances)
    
    print(f"  采样序列起始位置在episode中的索引: {start_idx_in_episode}")
    
    # 6. 创建可视化
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # ===== 上图：完整episode轨迹 =====
    
    # 绘制完整轨迹
    ax1.plot(full_trajectory[:, 0], full_trajectory[:, 1], 
            'o-', color='lightgray', linewidth=1.5, markersize=3, 
            alpha=0.6, label='Full Episode Trajectory')
    
    # 标记起点和终点
    ax1.plot(full_trajectory[0, 0], full_trajectory[0, 1], 
            'go', markersize=10, label='Episode Start', zorder=5)
    ax1.plot(full_trajectory[-1, 0], full_trajectory[-1, 1], 
            'rx', markersize=12, label='Episode End', zorder=5)
    
    # 高亮采样区域
    end_idx_in_episode = min(start_idx_in_episode + T_sampled - 1, T_full - 1)
    sampled_region_traj = full_trajectory[start_idx_in_episode:end_idx_in_episode+1]
    
    if len(sampled_region_traj) > 0:
        ax1.plot(sampled_region_traj[:, 0], sampled_region_traj[:, 1],
                'o-', color='orange', linewidth=3, markersize=6,
                label=f'Sampled Region (frames {start_idx_in_episode}-{end_idx_in_episode})', zorder=4)
        
        # 标记采样区域起点
        ax1.plot(sampled_region_traj[0, 0], sampled_region_traj[0, 1],
                'bs', markersize=10, label='Sampling Start', zorder=6)
    
    ax1.set_title(f'Full Episode Trajectory (Total Length: {T_full} frames)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Coordinate', fontsize=12)
    ax1.set_ylabel('Y Coordinate', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # ===== 下图：采样序列轨迹 =====
    
    # 绘制采样序列轨迹
    ax2.plot(sampled_trajectory[:, 0], sampled_trajectory[:, 1],
            'o-', color='red', linewidth=2.5, markersize=6,
            label=f'Sampled Sequence (length: {T_sampled} frames)', zorder=3)
    
    # 标记起点和终点
    ax2.plot(sampled_trajectory[0, 0], sampled_trajectory[0, 1],
            'go', markersize=10, label='Start', zorder=5)
    ax2.plot(sampled_trajectory[-1, 0], sampled_trajectory[-1, 1],
            'rx', markersize=12, label='End', zorder=5)
    
    # 添加帧号标注
    for i in [0, T_sampled//4, T_sampled//2, 3*T_sampled//4, T_sampled-1]:
        if i < len(sampled_trajectory):
            ax2.annotate(f't={i}', 
                        (sampled_trajectory[i, 0], sampled_trajectory[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                                             facecolor='yellow', alpha=0.7))
    
    ax2.set_title(f'Dataset Sampled Sequence (horizon={T_sampled})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Coordinate', fontsize=12)
    ax2.set_ylabel('Y Coordinate', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 设置相同的坐标范围以便对比
    all_x = np.concatenate([full_trajectory[:, 0], sampled_trajectory[:, 0]])
    all_y = np.concatenate([full_trajectory[:, 1], sampled_trajectory[:, 1]])
    x_margin = (all_x.max() - all_x.min()) * 0.1
    y_margin = (all_y.max() - all_y.min()) * 0.1
    
    ax1.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
    ax1.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)
    ax2.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
    ax2.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)
    
    # ===== 时间轴视图：展示采样关系 =====
    ax3.barh(0, T_full, height=0.3, color='lightgray', alpha=0.5, label='Full Episode')
    ax3.barh(0, T_sampled, left=start_idx_in_episode, height=0.3, 
            color='orange', alpha=0.8, label=f'Sampled Sequence (horizon={T_sampled})')
    
    # 标记关键点
    ax3.plot([0], [0], 'go', markersize=10, label='Episode Start', zorder=5)
    ax3.plot([T_full-1], [0], 'rx', markersize=10, label='Episode End', zorder=5)
    ax3.plot([start_idx_in_episode], [0], 'bs', markersize=10, label='Sampling Start', zorder=5)
    ax3.plot([end_idx_in_episode], [0], 'rs', markersize=10, label='Sampling End', zorder=5)
    
    # 添加文本标注
    ax3.text(T_full/2, 0.5, f'Full Episode: {T_full} frames', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.text(start_idx_in_episode + T_sampled/2, -0.5, 
            f'Sampled Sequence: {T_sampled} frames\n(frames {start_idx_in_episode}-{end_idx_in_episode})',
            ha='center', va='top', fontsize=10, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax3.set_xlim(-5, T_full + 5)
    ax3.set_ylim(-1, 1)
    ax3.set_xlabel('Timestep (Frame Index)', fontsize=12)
    ax3.set_title('Sampling Relationship Timeline', fontsize=14, fontweight='bold')
    ax3.set_yticks([])
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = root / "data/vis_pusht/episode_sampling_relationship.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 保存图片: {output_path}")
    
    # 显示图片
    plt.show()
    
    # 7. 打印详细信息
    print("\n" + "=" * 60)
    print("采样关系说明:")
    print(f"  - 完整Episode: {T_full} 帧")
    print(f"  - 采样序列: {T_sampled} 帧 (horizon=16)")
    print(f"  - 采样区域: episode中的第 {start_idx_in_episode} 到 {end_idx_in_episode} 帧")
    print(f"  - 采样比例: {T_sampled/T_full*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    visualize_episode_sampling()

