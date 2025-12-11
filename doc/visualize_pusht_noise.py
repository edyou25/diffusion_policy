#!/usr/bin/env python3
"""
简化版：可视化 PushT 数据集的编码和加噪声过程

功能：
    - 加载数据集
    - 归一化数据
    - 编码观测（图像 → 特征）
    - 加噪声（不同时间步）
    - 静态可视化对比图

用法:
    python doc/visualize_pusht_noise.py
"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch

# 添加项目根目录到路径
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import OmegaConf
import hydra


def visualize_noise_process():
    """主函数：展示编码和加噪声过程"""
    
    # 1. 加载配置和数据集
    root = pathlib.Path(__file__).parent.parent
    zarr_path = root / "data/pusht/pusht_cchi_v7_replay.zarr"
    
    print("=" * 60)
    print("加载数据集...")
    dataset = PushTImageDataset(
        zarr_path=str(zarr_path),
        horizon=16,
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.02,
        max_train_episodes=90,
    )
    print(f"数据集大小: {len(dataset)} 个样本")
    
    # 2. 获取归一化器
    print("\n获取归一化器...")
    normalizer = dataset.get_normalizer()
    
    # 3. 初始化噪声调度器
    print("初始化噪声调度器...")
    config_path = root / "image_pusht_diffusion_policy_cnn.yaml"
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        noise_scheduler = hydra.utils.instantiate(cfg.policy.noise_scheduler)
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )
    
    num_train_timesteps = noise_scheduler.config.num_train_timesteps
    noise_scheduler.set_timesteps(num_train_timesteps)
    print(f"训练时间步数: {num_train_timesteps}")
    
    # 4. 初始化观测编码器（可选，如果无法加载则跳过编码步骤）
    print("\n初始化观测编码器...")
    obs_encoder = None
    try:
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            # 实例化整个策略类，然后提取观测编码器
            policy = hydra.utils.instantiate(cfg.policy)
            obs_encoder = policy.obs_encoder
            obs_encoder.eval()
            if torch.cuda.is_available():
                obs_encoder = obs_encoder.cuda()
            print(f"✅ 观测编码器: {obs_encoder.__class__.__name__}")
        else:
            print("⚠️  配置文件不存在，跳过观测编码")
    except Exception as e:
        obs_encoder = None
        print(f"⚠️  无法加载观测编码器（将跳过编码步骤）: {str(e)[:100]}")
    
    # 5. 选择一个样本
    sample_idx = 0
    sample = dataset[sample_idx]
    print(f"\n处理样本 {sample_idx}...")
    
    # 提取数据
    img = sample["obs"]["image"].numpy()  # (T, 3, 96, 96)
    agent_pos = sample["obs"]["agent_pos"].numpy()  # (T, 2)
    action = sample["action"].numpy()  # (T, 2)
    T = img.shape[0]
    
    print(f"  序列长度: {T}")
    print(f"  图像形状: {img.shape}")
    print(f"  动作形状: {action.shape}")
    
    # 6. 归一化
    print("\n归一化数据...")
    # 分别归一化 obs 和 action
    nobs = normalizer.normalize(sample['obs'])
    naction = normalizer['action'].normalize(sample['action'])  # (T, 2)
    
    # 转换为 numpy 用于打印和后续处理
    if isinstance(naction, torch.Tensor):
        naction_np = naction.cpu().numpy()
    else:
        naction_np = naction
    
    print(f"  归一化后动作范围: [{naction_np.min():.3f}, {naction_np.max():.3f}]")
    
    # 7. 编码观测（如果可用）
    obs_features = None
    if obs_encoder is not None:
        print("\n编码观测...")
        with torch.no_grad():
            # 准备输入（使用归一化后的数据，已经是 Tensor）
            img_tensor = nobs['image']  # (T, 3, 96, 96)
            agent_pos_tensor = nobs['agent_pos']  # (T, 2)
            
            # 确保是 float 类型
            if not isinstance(img_tensor, torch.Tensor):
                img_tensor = torch.from_numpy(img_tensor).float()
            else:
                img_tensor = img_tensor.float()
            
            if not isinstance(agent_pos_tensor, torch.Tensor):
                agent_pos_tensor = torch.from_numpy(agent_pos_tensor).float()
            else:
                agent_pos_tensor = agent_pos_tensor.float()
            
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
                agent_pos_tensor = agent_pos_tensor.cuda()
            
            # 编码（reshape 为 batch）
            img_batch = img_tensor.reshape(-1, 3, 96, 96)  # (T, 3, 96, 96)
            agent_pos_batch = agent_pos_tensor.reshape(-1, 2)  # (T, 2)
            
            obs_dict = {
                'image': img_batch,
                'agent_pos': agent_pos_batch
            }
            
            obs_features = obs_encoder(obs_dict)  # (T, D)
            obs_features = obs_features.cpu().numpy()
        
        print(f"  观测特征形状: {obs_features.shape}")
        print(f"  特征范围: [{obs_features.min():.3f}, {obs_features.max():.3f}]")
    
    # 8. 加噪声（不同时间步）
    print("\n加噪声（不同时间步）...")
    timesteps_list = [0, 10, 25, 50, 75, 99]
    timesteps_list = [t for t in timesteps_list if t < num_train_timesteps]
    
    # 确保 naction 是 Tensor
    if isinstance(naction, torch.Tensor):
        naction_torch = naction.float()
    else:
        naction_torch = torch.from_numpy(naction).float()
    
    if torch.cuda.is_available():
        naction_torch = naction_torch.cuda()
    
    noisy_actions = {}
    noise_levels = {}
    
    for timestep in timesteps_list:
        # 生成噪声
        noise = torch.randn_like(naction_torch)
        timesteps_tensor = torch.full((1,), timestep, dtype=torch.long, device=naction_torch.device)
        
        # 加噪声
        noisy_naction = noise_scheduler.add_noise(
            naction_torch.unsqueeze(0).float(),
            noise.unsqueeze(0).float(),
            timesteps_tensor.long()
        ).squeeze(0)
        
        # 反归一化用于可视化
        noisy_action_np = noisy_naction.cpu().numpy()
        noisy_action_unnorm = normalizer['action'].unnormalize(
            torch.from_numpy(noisy_action_np).float()
        ).numpy()
        
        noisy_actions[timestep] = noisy_action_unnorm
        
        # 计算噪声水平
        alpha_prod = noise_scheduler.alphas_cumprod[timestep]
        if isinstance(alpha_prod, torch.Tensor):
            alpha_prod = alpha_prod.cpu().item()
        noise_level = (1 - alpha_prod) * 100
        noise_levels[timestep] = noise_level
        
        print(f"  时间步 {timestep:3d}: 噪声水平 {noise_level:.1f}%")
    
    # 9. 可视化
    print("\n生成可视化...")
    
    # 计算原始轨迹和加噪声后的轨迹
    original_trajectory = np.zeros_like(agent_pos)
    original_trajectory[0] = agent_pos[0]
    for t in range(1, T):
        original_trajectory[t] = original_trajectory[t-1] + action[t-1]
    
    # 创建对比图：每个时间步一个子图，每个子图中同时显示原始和加噪声后的轨迹
    num_timesteps = len(timesteps_list)
    cols = min(3, num_timesteps)  # 每行最多3个
    rows = (num_timesteps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    if num_timesteps == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, timestep in enumerate(timesteps_list):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # 计算加噪声后的轨迹
        noisy_trajectory = np.zeros_like(agent_pos)
        noisy_trajectory[0] = agent_pos[0]  # 起点相同
        for t in range(1, T):
            noisy_trajectory[t] = noisy_trajectory[t-1] + noisy_actions[timestep][t-1]
        
        # 在同一图中绘制原始轨迹和加噪声后的轨迹
        # 原始轨迹
        ax.plot(original_trajectory[:, 0], original_trajectory[:, 1], 
               'o-', color='orange', linewidth=2.5, markersize=5, 
               alpha=0.8, label='Original Trajectory', zorder=2)
        ax.plot(original_trajectory[0, 0], original_trajectory[0, 1], 
               'go', markersize=10, label='Original Start', zorder=4)
        ax.plot(original_trajectory[-1, 0], original_trajectory[-1, 1], 
               'bx', markersize=12, label='Original End', zorder=4)
        
        # 加噪声后的轨迹
        ax.plot(noisy_trajectory[:, 0], noisy_trajectory[:, 1], 
               's-', color='red', linewidth=2.5, markersize=5, 
               alpha=0.8, label='Noisy Trajectory', zorder=3)
        ax.plot(noisy_trajectory[0, 0], noisy_trajectory[0, 1], 
               'g^', markersize=10, label='Noisy Start', zorder=5)
        ax.plot(noisy_trajectory[-1, 0], noisy_trajectory[-1, 1], 
               'r+', markersize=12, linewidth=2, label='Noisy End', zorder=5)
        
        # 设置标题和标签
        noise_level = noise_levels[timestep]
        ax.set_title(f'Timestep {timestep} (Noise: {noise_level:.1f}%)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=10)
        ax.set_ylabel('Y Coordinate', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.set_aspect('equal')
    
    # 隐藏多余的子图
    for idx in range(num_timesteps, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = root / "data/vis_pusht/noise_process_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 保存对比图: {output_path}")
    
    # 显示图片
    plt.show()
    
    # 10. 打印统计信息
    print("\n" + "=" * 60)
    print("统计信息:")
    print(f"  原始动作范围: [{action.min():.2f}, {action.max():.2f}]")
    print(f"  归一化动作范围: [{naction.min():.3f}, {naction.max():.3f}]")
    if obs_features is not None:
        print(f"  观测特征维度: {obs_features.shape[1]}")
        print(f"  观测特征范围: [{obs_features.min():.3f}, {obs_features.max():.3f}]")
    print("=" * 60)


if __name__ == "__main__":
    visualize_noise_process()
