#!/usr/bin/env python3
"""
可视化 PushT 数据集 - 在图像上叠加完整轨迹，连续播放

特色功能：
    - 🎯 完整轨迹叠加：灰色路径显示整个episode轨迹
    - 🟢 起点标记：绿色圆点
    - 🔵 终点标记：蓝色叉号
    - 🟠 已走路径：橙色加粗线条
    - 🔴 当前位置：红色圆点
    - 🔵 动作向量：蓝色箭头

用法:
    # ⭐ 窗口播放完整episode（100+帧专家演示）推荐！
    python doc/visualize_pusht_dataset.py --full_episodes --play --num_samples 3
    
    # 慢速播放，清晰看到轨迹演进
    python doc/visualize_pusht_dataset.py --full_episodes --play --num_samples 2 --fps 5
    
    # 保存完整episode为GIF（带轨迹叠加）
    python doc/visualize_pusht_dataset.py --full_episodes --num_samples 3
    
    # 窗口播放数据集采样（16帧片段）
    python doc/visualize_pusht_dataset.py --play --num_samples 3

播放控制（--play模式）:
    空格: 暂停/继续
    ESC/q: 退出当前播放
    ←→ 或 a/d: 前后翻帧
    r: 重新播放
"""

import sys
import pathlib
import argparse
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow

# 添加项目根目录到路径
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
import cv2
import time


def play_frames_window(frames, fps=10, window_name="PushT Visualization"):
    """
    在窗口中播放帧序列
    
    按键控制:
        空格: 暂停/继续
        ESC/q: 退出
        左箭头: 上一帧
        右箭头: 下一帧
        r: 重新播放
    
    Args:
        frames: 帧序列 numpy array (T, H, W, 3)
        fps: 播放帧率
        window_name: 窗口名称
    """
    T = len(frames)
    frame_idx = 0
    paused = False
    delay = int(1000 / fps)  # 毫秒
    
    print(f"\n▶ 播放 {T} 帧 @ {fps} fps")
    print("  空格: 暂停/继续 | ESC/q: 退出 | ←→: 前后翻帧 | r: 重播")
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 384, 384)  # 4倍放大显示
    
    while True:
        # 显示当前帧
        frame_bgr = cv2.cvtColor(frames[frame_idx], cv2.COLOR_RGB2BGR)
        
        # 添加帧号指示
        display_frame = frame_bgr.copy()
        text = f"Frame: {frame_idx+1}/{T}"
        if paused:
            text += " [PAUSED]"
        # cv2.putText(display_frame, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 
        #            0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow(window_name, display_frame)
        
        # 等待按键
        key = cv2.waitKey(delay if not paused else 0) & 0xFF
        
        # 处理按键
        if key == 27 or key == ord('q'):  # ESC or 'q'
            print("  ⏹ 停止播放")
            break
        elif key == ord(' '):  # 空格
            paused = not paused
            status = "暂停" if paused else "继续"
            print(f"  ⏸ {status}")
        elif key == 83 or key == ord('d'):  # 右箭头 or 'd'
            frame_idx = min(frame_idx + 1, T - 1)
            print(f"  → 第 {frame_idx+1}/{T} 帧")
        elif key == 81 or key == ord('a'):  # 左箭头 or 'a'
            frame_idx = max(frame_idx - 1, 0)
            print(f"  ← 第 {frame_idx+1}/{T} 帧")
        elif key == ord('r'):  # 'r' 重播
            frame_idx = 0
            paused = False
            print("  🔄 重新播放")
        else:
            # 自动播放
            if not paused:
                frame_idx += 1
                if frame_idx >= T:
                    frame_idx = 0  # 循环播放
    
    cv2.destroyWindow(window_name)


def visualize_sample(sample, sample_idx, output_dir, save_gif=True, save_frames=False, 
                     show_state=True, show_trajectory=True, fps=10, play=False):
    """
    可视化一个数据样本
    
    Args:
        sample: 数据集返回的样本字典
        sample_idx: 样本索引
        output_dir: 输出目录
        save_gif: 是否保存为 GIF
        save_frames: 是否保存单独的帧
        show_state: 是否在图像上绘制 state 信息
        show_trajectory: 是否显示轨迹
        fps: GIF 帧率
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取数据
    img = sample["obs"]["image"].numpy()       # (T, 3, 96, 96)
    agent_pos = sample["obs"]["agent_pos"].numpy()  # (T, 2)
    action = sample["action"].numpy()          # (T, 2)
    
    T = img.shape[0]
    
    # 转换图像格式: (T, 3, 96, 96) -> (T, 96, 96, 3)
    frames = np.moveaxis(img, 1, -1)
    frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8).copy()  # copy 以便修改
    
    print(f"\n样本 {sample_idx}:")
    print(f"  序列长度: {T}")
    print(f"  图像形状: {frames.shape}")
    print(f"  Agent位置范围: [{agent_pos.min(axis=0)}, {agent_pos.max(axis=0)}]")
    print(f"  动作范围: [{action.min(axis=0)}, {action.max(axis=0)}]")
    
    # 1. 保存或播放纯图像
    if play:
        print(f"\n播放原始图像:")
        play_frames_window(frames, fps=fps, window_name=f"Sample {sample_idx} - Raw")
    elif save_gif:
        gif_path = output_dir / f"sample_{sample_idx:03d}_raw.gif"
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)  # type: ignore
        print(f"  保存原始 GIF: {gif_path}")
    
    # 2. 创建带 state 标注的帧
    frames_annotated = None
    if show_state:
        frames_annotated = []
        
        for t in range(T):
            frame = frames[t].copy()
            
            # 坐标转换：从 [0, 512] 映射到 [0, 96]
            agent_x = int(agent_pos[t, 0] * 96 / 512)
            agent_y = int(agent_pos[t, 1] * 96 / 512)
            
            # 绘制完整轨迹（如果启用）
            if show_trajectory:
                # 绘制整个episode的完整轨迹（半透明灰色）
                for i in range(T - 1):
                    x1 = int(agent_pos[i, 0] * 96 / 512)
                    y1 = int(agent_pos[i, 1] * 96 / 512)
                    x2 = int(agent_pos[i+1, 0] * 96 / 512)
                    y2 = int(agent_pos[i+1, 1] * 96 / 512)
                    cv2.line(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)  # 灰色完整轨迹
                
                # 标记起点（绿色）
                start_x = int(agent_pos[0, 0] * 96 / 512)
                start_y = int(agent_pos[0, 1] * 96 / 512)
                cv2.circle(frame, (start_x, start_y), 3, (0, 255, 0), -1)
                
                # 标记终点（蓝色）
                end_x = int(agent_pos[-1, 0] * 96 / 512)
                end_y = int(agent_pos[-1, 1] * 96 / 512)
                cv2.drawMarker(frame, (end_x, end_y), (0, 0, 255), 
                             cv2.MARKER_CROSS, 6, 2)
                
                # 绘制已经走过的轨迹（橙色加粗）
                if t > 0:
                    for i in range(t):
                        x1 = int(agent_pos[i, 0] * 96 / 512)
                        y1 = int(agent_pos[i, 1] * 96 / 512)
                        x2 = int(agent_pos[i+1, 0] * 96 / 512)
                        y2 = int(agent_pos[i+1, 1] * 96 / 512)
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # 橙色已走轨迹
            
            # 绘制 agent 当前位置（红色圆点）
            cv2.circle(frame, (agent_x, agent_y), 4, (255, 0, 0), -1)
            cv2.circle(frame, (agent_x, agent_y), 5, (255, 255, 255), 1)  # 白色边框
            
            # 绘制动作向量（蓝色箭头）
            if t < len(action):
                # 动作是位移，缩放后绘制
                action_scale = 0.1  # 调整箭头长度
                dx = int(action[t, 0] * action_scale)
                dy = int(action[t, 1] * action_scale)
                end_x = agent_x + dx
                end_y = agent_y + dy
                cv2.arrowedLine(frame, (agent_x, agent_y), (end_x, end_y), 
                               (0, 0, 255), 2, tipLength=0.3)
            
            # 添加文本信息
            text = f"t={t}/{T-1}"
            cv2.putText(frame, text, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, text, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (0, 0, 0), 2, cv2.LINE_AA)  # 黑色描边
            
            # 显示 agent 位置
            pos_text = f"pos:({agent_pos[t,0]:.0f},{agent_pos[t,1]:.0f})"
            cv2.putText(frame, pos_text, (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.3, (255, 255, 255), 1, cv2.LINE_AA)
            
            # 显示动作
            if t < len(action):
                act_text = f"act:({action[t,0]:.0f},{action[t,1]:.0f})"
                cv2.putText(frame, act_text, (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.3, (255, 255, 255), 1, cv2.LINE_AA)
            
            frames_annotated.append(frame)
        
        # 播放或保存带标注的帧
        if play:
            print(f"\n播放标注图像:")
            play_frames_window(np.array(frames_annotated), fps=fps, 
                             window_name=f"Sample {sample_idx} - Annotated")
        elif save_gif:
            gif_path_annotated = output_dir / f"sample_{sample_idx:03d}_annotated.gif"
            imageio.mimsave(gif_path_annotated, frames_annotated, fps=fps, loop=0)  # type: ignore
            print(f"  保存标注 GIF: {gif_path_annotated}")
    
    # 3. 创建对比图（前8帧）
    if show_state and frames_annotated is not None:
        num_display = min(8, T)
        fig, axes = plt.subplots(2, num_display, figsize=(num_display*2, 4))
        if num_display == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        
        for t in range(num_display):
            # 上排：原始图像
            ax = axes[0, t]
            ax.imshow(frames[t])
            ax.set_title(f't={t}', fontsize=10)
            ax.axis('off')
            
            # 下排：带标注的图像
            ax = axes[1, t]
            ax.imshow(frames_annotated[t])
            ax.set_title(f'annotated', fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        comparison_path = output_dir / f"sample_{sample_idx:03d}_comparison.png"
        plt.savefig(comparison_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  保存对比图: {comparison_path}")
    
    # 4. 保存单独的帧（可选）
    if save_frames:
        frames_dir = output_dir / f"sample_{sample_idx:03d}_frames"
        frames_dir.mkdir(exist_ok=True)
        for t in range(T):
            # 保存原始帧
            frame_path = frames_dir / f"frame_{t:03d}_raw.png"
            imageio.imwrite(frame_path, frames[t])
            # 保存标注帧
            if show_state and frames_annotated is not None:
                frame_path_ann = frames_dir / f"frame_{t:03d}_annotated.png"
                imageio.imwrite(frame_path_ann, frames_annotated[t])
        print(f"  保存 {T} 帧到: {frames_dir}")


def visualize_full_episode(replay_buffer, episode_idx, output_dir, 
                           show_state=True, show_trajectory=True, fps=10, 
                           play=False, save_gif=True):
    """
    可视化完整的 episode（不是采样的16帧片段）
    
    Args:
        replay_buffer: ReplayBuffer 对象
        episode_idx: episode 索引
        output_dir: 输出目录
        show_state: 是否显示 state 标注
        show_trajectory: 是否显示轨迹
        fps: GIF 帧率
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取 episode 数据
    episode = replay_buffer.get_episode(episode_idx)
    
    # 提取数据
    img = episode['img']  # (T, 96, 96, 3)
    state = episode['state']  # (T, 5)
    action = episode['action']  # (T, 2)
    
    T = len(img)
    agent_pos = state[:, :2]  # (T, 2)
    
    # 转换图像格式
    frames = (img).astype(np.uint8).copy()
    
    print(f"\nEpisode {episode_idx}:")
    print(f"  完整长度: {T} 帧")
    print(f"  图像形状: {frames.shape}")
    print(f"  Agent位置范围: [{agent_pos.min(axis=0)}, {agent_pos.max(axis=0)}]")
    print(f"  动作范围: [{action.min(axis=0)}, {action.max(axis=0)}]")
    
    # 1. 播放或保存原始帧
    if play:
        print(f"\n播放原始图像 ({T}帧):")
        play_frames_window(frames, fps=fps, window_name=f"Episode {episode_idx} - Raw")
    elif save_gif:
        gif_path = output_dir / f"episode_{episode_idx:03d}_raw.gif"
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)  # type: ignore
        print(f"  保存原始 GIF ({T}帧): {gif_path}")
    
    # 2. 创建带标注的帧
    frames_annotated = None
    if show_state:
        frames_annotated = []
        
        for t in range(T):
            frame = frames[t].copy()
            
            # 坐标转换：从 [0, 512] 映射到 [0, 96]
            agent_x = int(agent_pos[t, 0] * 96 / 512)
            agent_y = int(agent_pos[t, 1] * 96 / 512)
            
            # 绘制完整轨迹（如果启用）
            if show_trajectory:
                # 绘制整个episode的完整轨迹（半透明灰色）
                for i in range(T - 1):
                    x1 = int(agent_pos[i, 0] * 96 / 512)
                    y1 = int(agent_pos[i, 1] * 96 / 512)
                    x2 = int(agent_pos[i+1, 0] * 96 / 512)
                    y2 = int(agent_pos[i+1, 1] * 96 / 512)
                    cv2.line(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)  # 灰色完整轨迹
                
                # 标记起点（绿色）
                start_x = int(agent_pos[0, 0] * 96 / 512)
                start_y = int(agent_pos[0, 1] * 96 / 512)
                cv2.circle(frame, (start_x, start_y), 3, (0, 255, 0), -1)
                
                # 标记终点（蓝色）
                end_x = int(agent_pos[-1, 0] * 96 / 512)
                end_y = int(agent_pos[-1, 1] * 96 / 512)
                cv2.drawMarker(frame, (end_x, end_y), (0, 0, 255), 
                             cv2.MARKER_CROSS, 6, 2)
                
                # 绘制已经走过的轨迹（橙色加粗）
                if t > 0:
                    for i in range(t):
                        x1 = int(agent_pos[i, 0] * 96 / 512)
                        y1 = int(agent_pos[i, 1] * 96 / 512)
                        x2 = int(agent_pos[i+1, 0] * 96 / 512)
                        y2 = int(agent_pos[i+1, 1] * 96 / 512)
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # 橙色已走轨迹
            
            # 绘制 agent 当前位置（红色圆点）
            cv2.circle(frame, (agent_x, agent_y), 4, (255, 0, 0), -1)
            cv2.circle(frame, (agent_x, agent_y), 5, (255, 255, 255), 1)
            
            # 绘制动作向量（蓝色箭头）
            if t < len(action):
                action_scale = 0.1
                dx = int(action[t, 0] * action_scale)
                dy = int(action[t, 1] * action_scale)
                end_x = agent_x + dx
                end_y = agent_y + dy
                cv2.arrowedLine(frame, (agent_x, agent_y), (end_x, end_y), 
                               (0, 0, 255), 2, tipLength=0.3)
            
            # 添加文本信息
            text = f"t={t}/{T-1}"
            cv2.putText(frame, text, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
            pos_text = f"pos:({agent_pos[t,0]:.0f},{agent_pos[t,1]:.0f})"
            cv2.putText(frame, pos_text, (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.3, (255, 255, 255), 1, cv2.LINE_AA)
            
            if t < len(action):
                act_text = f"act:({action[t,0]:.0f},{action[t,1]:.0f})"
                cv2.putText(frame, act_text, (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.3, (255, 255, 255), 1, cv2.LINE_AA)
            
            frames_annotated.append(frame)
        
        # 播放或保存带标注的帧
        if play:
            print(f"\n播放标注图像 ({T}帧):")
            play_frames_window(np.array(frames_annotated), fps=fps, 
                             window_name=f"Episode {episode_idx} - Annotated")
        elif save_gif:
            gif_path_annotated = output_dir / f"episode_{episode_idx:03d}_annotated.gif"
            imageio.mimsave(gif_path_annotated, frames_annotated, fps=fps, loop=0)  # type: ignore
            print(f"  保存标注 GIF ({T}帧): {gif_path_annotated}")
    
    # 3. 创建关键帧对比图（可选）
    if show_state and frames_annotated is not None and not play:
        # 只在保存模式下生成，播放模式不需要
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        
        # 显示关键帧
        key_frames = [
            ('Start', 0),
            ('1/4', T//4),
            ('1/2', T//2),
            ('3/4', T*3//4),
            ('End', T-1)
        ]
        
        for i, (label, idx) in enumerate(key_frames):
            ax = axes[i]
            ax.imshow(frames_annotated[idx])
            ax.set_title(f'{label}\nt={idx}', fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        keyframes_path = output_dir / f"episode_{episode_idx:03d}_keyframes.png"
        plt.savefig(keyframes_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  保存关键帧: {keyframes_path}")


def plot_dataset_statistics(dataset, output_dir):
    """绘制数据集统计信息"""
    output_dir = pathlib.Path(output_dir)
    
    print("\n收集数据集统计信息...")
    
    # 采样一部分数据来分析
    num_samples = min(100, len(dataset))
    all_actions = []
    all_agent_pos = []
    
    for i in range(num_samples):
        sample = dataset[i]
        all_actions.append(sample["action"].numpy())
        all_agent_pos.append(sample["obs"]["agent_pos"].numpy())
    
    all_actions = np.concatenate(all_actions, axis=0)  # (N, 2)
    all_agent_pos = np.concatenate(all_agent_pos, axis=0)  # (N, 2)
    
    # 创建统计图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 动作分布 - X
    axes[0, 0].hist(all_actions[:, 0], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Action X')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title(f'动作 X 分布\n(mean={all_actions[:,0].mean():.3f}, std={all_actions[:,0].std():.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 动作分布 - Y
    axes[0, 1].hist(all_actions[:, 1], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Action Y')
    axes[0, 1].set_ylabel('频数')
    axes[0, 1].set_title(f'动作 Y 分布\n(mean={all_actions[:,1].mean():.3f}, std={all_actions[:,1].std():.3f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Agent 位置分布
    axes[1, 0].scatter(all_agent_pos[:, 0], all_agent_pos[:, 1], 
                       alpha=0.1, s=1, color='red')
    axes[1, 0].set_xlabel('Agent X')
    axes[1, 0].set_ylabel('Agent Y')
    axes[1, 0].set_title('Agent 位置分布')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')
    
    # 4. 动作向量场
    # 下采样以避免过于密集
    step = max(1, len(all_actions) // 500)
    axes[1, 1].quiver(
        all_agent_pos[::step, 0], 
        all_agent_pos[::step, 1],
        all_actions[::step, 0], 
        all_actions[::step, 1],
        alpha=0.3, scale=20
    )
    axes[1, 1].set_xlabel('Agent X')
    axes[1, 1].set_ylabel('Agent Y')
    axes[1, 1].set_title('动作向量场')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    stats_path = output_dir / "dataset_statistics.png"
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n保存统计图表: {stats_path}")
    
    # 打印数值统计
    print("\n数据集统计信息:")
    print(f"  样本数: {len(dataset)}")
    print(f"  动作 X: mean={all_actions[:,0].mean():.4f}, std={all_actions[:,0].std():.4f}, "
          f"min={all_actions[:,0].min():.4f}, max={all_actions[:,0].max():.4f}")
    print(f"  动作 Y: mean={all_actions[:,1].mean():.4f}, std={all_actions[:,1].std():.4f}, "
          f"min={all_actions[:,1].min():.4f}, max={all_actions[:,1].max():.4f}")
    print(f"  Agent X: mean={all_agent_pos[:,0].mean():.4f}, std={all_agent_pos[:,0].std():.4f}, "
          f"min={all_agent_pos[:,0].min():.4f}, max={all_agent_pos[:,0].max():.4f}")
    print(f"  Agent Y: mean={all_agent_pos[:,1].mean():.4f}, std={all_agent_pos[:,1].std():.4f}, "
          f"min={all_agent_pos[:,1].min():.4f}, max={all_agent_pos[:,1].max():.4f}")


def main():
    parser = argparse.ArgumentParser(description="可视化 PushT 数据集")
    parser.add_argument(
        '--zarr_path', 
        type=str, 
        default='data/pusht/pusht_cchi_v7_replay.zarr',
        help='Zarr 数据集路径'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='data/vis_pusht',
        help='输出目录'
    )
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=1,
        help='可视化的样本数量'
    )
    parser.add_argument(
        '--random', 
        action='store_true',
        help='随机选择样本（而不是前N个）'
    )
    parser.add_argument(
        '--save_frames', 
        action='store_true',
        help='保存单独的帧图像'
    )
    parser.add_argument(
        '--stats', 
        action='store_true',
        help='生成数据集统计信息'
    )
    parser.add_argument(
        '--validation', 
        action='store_true',
        help='可视化验证集而非训练集'
    )
    parser.add_argument(
        '--fps', 
        type=int, 
        default=10,
        help='GIF 帧率（默认：10）'
    )
    parser.add_argument(
        '--no_trajectory', 
        action='store_true',
        help='不显示轨迹线'
    )
    parser.add_argument(
        '--play', 
        action='store_true',
        default=True,
        help='在窗口中播放而不是保存为GIF（支持暂停、翻帧等交互）'
    )
    parser.add_argument(
        '--no_save_gif',
        action='store_true',
        help='不保存GIF文件（仅与--play一起使用时有效）'
    )
    parser.add_argument(
        '--full_episodes',
        action='store_true',
        default=True,
        help='可视化完整的episode（100+帧）而不是采样的16帧片段'
    )
    
    args = parser.parse_args()
    
    # 构建完整路径
    root = pathlib.Path(__file__).parent.parent
    zarr_path = root / args.zarr_path
    output_dir = root / args.output_dir
    
    if not zarr_path.exists():
        print(f"错误: 数据集路径不存在: {zarr_path}")
        return
    
    print(f"加载数据集: {zarr_path}")
    
    # 根据模式选择不同的可视化方式
    if args.full_episodes:
        # ===== 完整 Episode 模式 =====
        print("\n📺 完整 Episode 模式（播放整个专家演示）")
        
        # 直接从 ReplayBuffer 加载
        replay_buffer = ReplayBuffer.copy_from_path(
            str(zarr_path), 
            keys=['img', 'state', 'action']
        )
        
        total_episodes = replay_buffer.n_episodes
        print(f"总 Episodes: {total_episodes}")
        
        # 选择要可视化的 episode 索引
        if args.random:
            import random
            indices = random.sample(range(total_episodes), min(args.num_samples, total_episodes))
        else:
            indices = list(range(min(args.num_samples, total_episodes)))
        indices = [123,111,202]
        print(f"将可视化 {len(indices)} 个完整 episodes: {indices}")
        
        # 可视化每个 episode
        for i, episode_idx in enumerate(indices):
            print(f"\n{'='*50}")
            print(f"Episode {i+1}/{len(indices)}: 索引 {episode_idx}")
            print(f"{'='*50}")
            
            visualize_full_episode(
                replay_buffer,
                episode_idx,
                output_dir,
                show_state=True,
                show_trajectory=not args.no_trajectory,
                fps=args.fps,
                play=args.play,
                save_gif=(not args.no_save_gif) and (not args.play)
            )
            
            # 播放模式下，询问是否继续
            if args.play and i < len(indices) - 1:
                response = input(f"\n继续播放下一个episode? (y/n, 默认y): ").strip().lower()
                if response == 'n':
                    print("停止播放")
                    break
    
    else:
        # ===== 数据集采样模式（16帧片段）=====
        print("\n📺 数据集采样模式（16帧片段）")
        
        # 创建数据集（使用与配置文件相同的参数）
        dataset = PushTImageDataset(
            zarr_path=str(zarr_path),
            horizon=16,
            pad_before=1,
            pad_after=7,
            seed=42,
            val_ratio=0.02,
            max_train_episodes=90,
        )
        
        # 切换到验证集（如果指定）
        if args.validation:
            print("使用验证集")
            dataset = dataset.get_validation_dataset()
        else:
            print("使用训练集")
        
        print(f"数据集大小: {len(dataset)} 个样本")
        
        # 选择要可视化的样本索引
        if args.random:
            import random
            indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
        else:
            indices = list(range(min(args.num_samples, len(dataset))))
        
        print(f"将可视化 {len(indices)} 个样本: {indices}")
        
        # 可视化每个样本
        for i, idx in enumerate(indices):
            print(f"\n{'='*50}")
            print(f"处理样本 {i+1}/{len(indices)}: 索引 {idx}")
            print(f"{'='*50}")
            sample = dataset[idx]
            visualize_sample(
                sample, 
                idx, 
                output_dir, 
                save_gif=(not args.no_save_gif) and (not args.play), 
                save_frames=args.save_frames,
                show_state=True,
                show_trajectory=not args.no_trajectory,
                fps=args.fps,
                play=args.play
            )
            
            # 播放模式下，询问是否继续
            if args.play and i < len(indices) - 1:
                response = input(f"\n继续播放下一个样本? (y/n, 默认y): ").strip().lower()
                if response == 'n':
                    print("停止播放")
                    break
    
    # 生成统计信息（如果指定，仅在数据集采样模式下）
    if args.stats and not args.full_episodes:
        plot_dataset_statistics(dataset, output_dir)  # type: ignore
    elif args.stats and args.full_episodes:
        print("\n⚠️  统计信息功能仅在数据集采样模式下可用（不使用 --full_episodes）")
    
    if args.play:
        print(f"\n✅ 播放完成！")
    else:
        print(f"\n✅ 完成！所有可视化已保存到: {output_dir}")
        print(f"\n提示:")
        print(f"  - 原始 GIF: {output_dir}/sample_*_raw.gif")
        print(f"  - 标注 GIF (含state+轨迹): {output_dir}/sample_*_annotated.gif")
        print(f"  - 对比图: {output_dir}/sample_*_comparison.png")
        if args.save_frames:
            print(f"  - 单帧图像: {output_dir}/sample_*_frames/")
        if args.stats:
            print(f"  - 统计图表: {output_dir}/dataset_statistics.png")


if __name__ == "__main__":
    main()

