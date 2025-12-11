#!/usr/bin/env python3
"""
训练日志可视化脚本

用法:
    python visualize_training_logs.py --log-file <logs.json.txt路径> [选项]

示例:
    python visualize_training_logs.py --log-file ../data/outputs/2025.12.03/18.08.49_train_diffusion_unet_hybrid_pusht_image/logs.json.txt
    python visualize_training_logs.py --log-file ../data/outputs/2025.12.03/18.08.49_train_diffusion_unet_hybrid_pusht_image/logs.json.txt --smooth 100 --save-dir ./plots
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# 设置字体（使用英文避免字体问题）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def load_logs(log_file: str) -> List[Dict]:
    """
    加载 JSON Lines 格式的日志文件
    
    Args:
        log_file: 日志文件路径
        
    Returns:
        日志条目列表
    """
    logs = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                log_entry = json.loads(line)
                logs.append(log_entry)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行解析失败: {e}", file=sys.stderr)
                continue
    return logs


def smooth_data(data: np.ndarray, window_size: int = 100) -> np.ndarray:
    """
    使用移动平均平滑数据
    
    Args:
        data: 原始数据
        window_size: 窗口大小
        
    Returns:
        平滑后的数据
    """
    if len(data) < window_size:
        return data
    
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='same')
    # 处理边界
    half_window = window_size // 2
    smoothed[:half_window] = data[:half_window]
    smoothed[-half_window:] = data[-half_window:]
    return smoothed


def extract_metrics(logs: List[Dict]) -> Dict[str, np.ndarray]:
    """
    从日志中提取指标
    
    Args:
        logs: 日志条目列表
        
    Returns:
        包含各种指标的字典
    """
    metrics = {
        'global_step': [],
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'train_action_mse_error': [],
    }
    
    # 检查是否有其他指标
    all_keys = set()
    for log in logs:
        all_keys.update(log.keys())
    
    # 提取已知指标
    for log in logs:
        metrics['global_step'].append(log.get('global_step', 0))
        metrics['epoch'].append(log.get('epoch', 0))
        metrics['train_loss'].append(log.get('train_loss', None))
        metrics['val_loss'].append(log.get('val_loss', None))
        metrics['lr'].append(log.get('lr', None))
        metrics['train_action_mse_error'].append(log.get('train_action_mse_error', None))
    
    # 转换为 numpy 数组
    result = {}
    for key, values in metrics.items():
        arr = np.array(values, dtype=float)
        # 将 None 替换为 NaN
        mask = np.array([v is None for v in values])
        arr[mask] = np.nan
        result[key] = arr
    
    return result


def plot_training_curves(
    metrics: Dict[str, np.ndarray],
    smooth_window: int = 0,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: tuple = (16, 10)
):
    """
    绘制训练曲线
    
    Args:
        metrics: 指标字典
        smooth_window: 平滑窗口大小（0表示不平滑）
        save_path: 保存路径（可选）
        show_plot: 是否显示图表
        figsize: 图表大小
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    global_step = metrics['global_step']
    epoch = metrics['epoch']
    
    # 1. 训练损失
    ax1 = fig.add_subplot(gs[0, :])
    train_loss = metrics['train_loss']
    train_loss_valid = train_loss[~np.isnan(train_loss)]
    
    if len(train_loss_valid) > 0:
        if smooth_window > 0:
            train_loss_smooth = smooth_data(train_loss, smooth_window)
            ax1.plot(global_step, train_loss, alpha=0.3, color='blue', label='Raw', linewidth=0.5)
            ax1.plot(global_step, train_loss_smooth, color='blue', label=f'Smoothed (window={smooth_window})', linewidth=2)
        else:
            ax1.plot(global_step, train_loss, color='blue', label='Train Loss', linewidth=1)
        
        ax1.set_xlabel('Global Step', fontsize=12)
        ax1.set_ylabel('Train Loss', fontsize=12)
        ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # 使用对数刻度，因为损失可能变化很大
    
    # 2. 验证损失（如果有）
    ax2 = fig.add_subplot(gs[1, 0])
    val_loss = metrics['val_loss']
    val_loss_valid = val_loss[~np.isnan(val_loss)]
    
    if len(val_loss_valid) > 0:
        val_steps = global_step[~np.isnan(val_loss)]
        val_loss_clean = val_loss[~np.isnan(val_loss)]
        
        if smooth_window > 0 and len(val_loss_clean) > smooth_window:
            val_loss_smooth = smooth_data(val_loss_clean, min(smooth_window, len(val_loss_clean)))
            ax2.plot(val_steps, val_loss_clean, alpha=0.5, color='orange', label='Raw', marker='o', markersize=3)
            ax2.plot(val_steps, val_loss_smooth, color='orange', label='Smoothed', linewidth=2)
        else:
            ax2.plot(val_steps, val_loss_clean, color='orange', label='Val Loss', marker='o', markersize=4)
        
        ax2.set_xlabel('Global Step', fontsize=12)
        ax2.set_ylabel('Validation Loss', fontsize=12)
        ax2.set_title('Validation Loss Curve', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    else:
        ax2.text(0.5, 0.5, 'No validation loss data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Validation Loss Curve', fontsize=14, fontweight='bold')
    
    # 3. 训练损失 vs 验证损失对比
    ax3 = fig.add_subplot(gs[1, 1])
    if len(train_loss_valid) > 0 and len(val_loss_valid) > 0:
        # 使用 epoch 作为 x 轴
        epochs_unique = np.unique(epoch[~np.isnan(val_loss)])
        val_loss_by_epoch = []
        train_loss_by_epoch = []
        
        for e in epochs_unique:
            epoch_mask = (epoch == e) & (~np.isnan(val_loss))
            if np.any(epoch_mask):
                val_loss_by_epoch.append(np.nanmean(val_loss[epoch_mask]))
            else:
                val_loss_by_epoch.append(np.nan)
            
            epoch_mask_train = (epoch == e) & (~np.isnan(train_loss))
            if np.any(epoch_mask_train):
                train_loss_by_epoch.append(np.nanmean(train_loss[epoch_mask_train]))
            else:
                train_loss_by_epoch.append(np.nan)
        
        val_loss_by_epoch = np.array(val_loss_by_epoch)
        train_loss_by_epoch = np.array(train_loss_by_epoch)
        
        ax3.plot(epochs_unique, train_loss_by_epoch, 'o-', label='Train Loss', color='blue', markersize=4)
        ax3.plot(epochs_unique, val_loss_by_epoch, 's-', label='Val Loss', color='orange', markersize=4)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12)
        ax3.set_title('Train vs Validation Loss', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    else:
        ax3.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Train vs Validation Loss', fontsize=14, fontweight='bold')
    
    # 4. 学习率曲线
    ax4 = fig.add_subplot(gs[2, 0])
    lr = metrics['lr']
    lr_valid = lr[~np.isnan(lr)]
    
    if len(lr_valid) > 0:
        ax4.plot(global_step, lr, color='green', label='Learning Rate', linewidth=1)
        ax4.set_xlabel('Global Step', fontsize=12)
        ax4.set_ylabel('Learning Rate', fontsize=12)
        ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    
    # 5. 训练动作MSE误差（如果有）
    ax5 = fig.add_subplot(gs[2, 1])
    mse_error = metrics['train_action_mse_error']
    mse_valid = mse_error[~np.isnan(mse_error)]
    
    if len(mse_valid) > 0:
        mse_steps = global_step[~np.isnan(mse_error)]
        mse_clean = mse_error[~np.isnan(mse_error)]
        
        if smooth_window > 0 and len(mse_clean) > smooth_window:
            mse_smooth = smooth_data(mse_clean, min(smooth_window, len(mse_clean)))
            ax5.plot(mse_steps, mse_clean, alpha=0.5, color='red', label='Raw', marker='o', markersize=3)
            ax5.plot(mse_steps, mse_smooth, color='red', label='Smoothed', linewidth=2)
        else:
            ax5.plot(mse_steps, mse_clean, color='red', label='Action MSE Error', marker='o', markersize=4)
        
        ax5.set_xlabel('Global Step', fontsize=12)
        ax5.set_ylabel('Action MSE Error', fontsize=12)
        ax5.set_title('Training Action MSE Error', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No MSE error data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Training Action MSE Error', fontsize=14, fontweight='bold')
    
    # 添加总体信息
    fig.suptitle(
        f'Training Logs Visualization\n'
        f'Total Steps: {int(global_step[-1]):,} | '
        f'Total Epochs: {int(epoch[-1]):,} | '
        f'Final Train Loss: {train_loss_valid[-1]:.6f}',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def print_statistics(metrics: Dict[str, np.ndarray]):
    """
    打印统计信息
    
    Args:
        metrics: 指标字典
    """
    print("\n" + "="*60)
    print("Training Statistics")
    print("="*60)
    
    global_step = metrics['global_step']
    epoch = metrics['epoch']
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']
    lr = metrics['lr']
    
    train_loss_valid = train_loss[~np.isnan(train_loss)]
    val_loss_valid = val_loss[~np.isnan(val_loss)]
    lr_valid = lr[~np.isnan(lr)]
    
    print(f"\nOverall Info:")
    print(f"  Total Steps: {int(global_step[-1]):,}")
    print(f"  Total Epochs: {int(epoch[-1]):,}")
    print(f"  Total Data Points: {len(global_step):,}")
    
    if len(train_loss_valid) > 0:
        print(f"\nTraining Loss:")
        print(f"  Initial: {train_loss_valid[0]:.6f}")
        print(f"  Final: {train_loss_valid[-1]:.6f}")
        print(f"  Min: {np.nanmin(train_loss_valid):.6f} (at step: {int(global_step[np.nanargmin(train_loss_valid)]):,})")
        print(f"  Max: {np.nanmax(train_loss_valid):.6f}")
        print(f"  Mean: {np.nanmean(train_loss_valid):.6f}")
        print(f"  Reduction: {(1 - train_loss_valid[-1] / train_loss_valid[0]) * 100:.2f}%")
    
    if len(val_loss_valid) > 0:
        print(f"\nValidation Loss:")
        print(f"  Initial: {val_loss_valid[0]:.6f}")
        print(f"  Final: {val_loss_valid[-1]:.6f}")
        print(f"  Min: {np.nanmin(val_loss_valid):.6f} (at step: {int(global_step[~np.isnan(val_loss)][np.nanargmin(val_loss_valid)]):,})")
        print(f"  Max: {np.nanmax(val_loss_valid):.6f}")
        print(f"  Mean: {np.nanmean(val_loss_valid):.6f}")
        print(f"  Validation Points: {len(val_loss_valid)}")
    
    if len(lr_valid) > 0:
        print(f"\nLearning Rate:")
        print(f"  Initial: {lr_valid[0]:.2e}")
        print(f"  Final: {lr_valid[-1]:.2e}")
        print(f"  Max: {np.nanmax(lr_valid):.2e}")
        print(f"  Min: {np.nanmin(lr_valid):.2e}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize training logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--log-file',
        type=str,
        required=True,
        help='Path to log file (JSON Lines format)'
    )
    parser.add_argument(
        '--smooth',
        type=int,
        default=0,
        help='Smoothing window size (0 means no smoothing, default: 0)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Directory to save plots (default: don\'t save)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Don\'t show plot (only save)'
    )
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Don\'t print statistics'
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"Error: File not found: {log_file}", file=sys.stderr)
        sys.exit(1)
    
    # 加载日志
    print(f"Loading log file: {log_file}")
    logs = load_logs(str(log_file))
    print(f"Loaded {len(logs):,} log entries")
    
    if len(logs) == 0:
        print("Error: Log file is empty", file=sys.stderr)
        sys.exit(1)
    
    # 提取指标
    print("Extracting metrics...")
    metrics = extract_metrics(logs)
    
    # 打印统计信息
    if not args.no_stats:
        print_statistics(metrics)
    
    # 确定保存路径
    save_path = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"training_curves_{log_file.stem}.png"
    elif not args.no_show:
        # 如果没有指定保存目录但需要显示，也保存一份到日志文件同目录
        save_path = log_file.parent / f"training_curves_{log_file.stem}.png"
    
    # 绘制图表
    print("Plotting curves...")
    plot_training_curves(
        metrics,
        smooth_window=args.smooth,
        save_path=str(save_path) if save_path else None,
        show_plot=not args.no_show
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()

