#!/usr/bin/env python3
"""
训练日志可视化脚本 - 使用 Plotly（更美观的交互式图表）

用法:
    python visualize_training_logs_plotly.py --log-file <logs.json.txt路径> [选项]

示例:
    python visualize_training_logs_plotly.py --log-file ../data/outputs/2025.12.03/18.08.49_train_diffusion_unet_hybrid_pusht_image/logs.json.txt
    python doc/visualize_training_logs_plotly.py --log-file data/outputs/2025.12.03/18.08.49_train_diffusion_unet_hybrid_pusht_image/logs.json.txt --smooth 100 --max-points 5000 --no-show --save-html ../data/plots/training_curves_plotly_fixed.html
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
                print(f"Warning: Line {line_num} parse failed: {e}", file=sys.stderr)
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


def downsample_data(x: np.ndarray, y: np.ndarray, max_points: int = 10000) -> tuple:
    """
    对数据进行降采样，保留最多 max_points 个点
    
    Args:
        x: x轴数据
        y: y轴数据
        max_points: 最大点数（如果为None或大于数据长度，不降采样）
        
    Returns:
        (x_downsampled, y_downsampled) 降采样后的数据
    """
    if max_points is None or len(x) <= max_points:
        return x, y
    
    # 均匀采样，确保包含首尾点
    if max_points == 1:
        return x[[0]], y[[0]]
    elif max_points == 2:
        return x[[0, -1]], y[[0, -1]]
    else:
        indices = np.linspace(0, len(x) - 1, max_points, dtype=int)
        # 确保包含最后一个点
        indices[-1] = len(x) - 1
        return x[indices], y[indices]


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
    max_points: int = 10000,
    save_html: Optional[str] = None,
    save_png: Optional[str] = None,
    show_plot: bool = True,
    height: int = 900
):
    """
    使用 Plotly 绘制训练曲线
    
    Args:
        metrics: 指标字典
        smooth_window: 平滑窗口大小（0表示不平滑）
        max_points: 每个曲线最大数据点数（用于降采样，默认10000）
        save_html: 保存HTML路径（可选）
        save_png: 保存PNG路径（可选）
        show_plot: 是否显示图表
        height: 图表高度
    """
    global_step = metrics['global_step']
    epoch = metrics['epoch']
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']
    lr = metrics['lr']
    mse_error = metrics['train_action_mse_error']
    
    # 创建子图
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Training Loss',
            'Validation Loss',
            'Train vs Validation Loss (by Epoch)',
            'Learning Rate Schedule',
            'Training Action MSE Error',
            'Loss Comparison'
        ),
        specs=[
            [{"colspan": 2}, None],  # 第一行：训练损失占两列
            [{"type": "scatter"}, {"type": "scatter"}],  # 第二行：验证损失、对比
            [{"type": "scatter"}, {"type": "scatter"}]   # 第三行：学习率、MSE
        ],
        vertical_spacing=0.12,  # 增加垂直间距，避免axis和title重合
        horizontal_spacing=0.12,  # 增加水平间距
        row_heights=[0.35, 0.32, 0.33]  # 设置行高比例
    )
    
    # 1. 训练损失（第一行，占两列）
    train_loss_valid = train_loss[~np.isnan(train_loss)]
    if len(train_loss_valid) > 0:
        # 降采样
        step_ds, loss_ds = downsample_data(global_step, train_loss, max_points)
        
        if smooth_window > 0:
            train_loss_smooth = smooth_data(train_loss, smooth_window)
            # 对平滑后的数据也进行降采样
            step_ds_smooth, loss_smooth_ds = downsample_data(global_step, train_loss_smooth, max_points)
            
            # 原始值（半透明）
            fig.add_trace(
                go.Scatter(
                    x=step_ds,
                    y=loss_ds,
                    mode='lines',
                    name='Raw',
                    line=dict(color='rgba(31, 119, 180, 0.2)', width=1),
                    hovertemplate='Step: %{x:,}<br>Loss: %{y:.6f}<extra></extra>'
                ),
                row=1, col=1
            )
            # 平滑值
            fig.add_trace(
                go.Scatter(
                    x=step_ds_smooth,
                    y=loss_smooth_ds,
                    mode='lines',
                    name=f'Smoothed (window={smooth_window})',
                    line=dict(color='rgb(31, 119, 180)', width=2),
                    hovertemplate='Step: %{x:,}<br>Smoothed Loss: %{y:.6f}<extra></extra>'
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=step_ds,
                    y=loss_ds,
                    mode='lines',
                    name='Train Loss',
                    line=dict(color='rgb(31, 119, 180)', width=1.5),
                    hovertemplate='Step: %{x:,}<br>Loss: %{y:.6f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        fig.update_xaxes(title_text="Global Step", row=1, col=1)
        fig.update_yaxes(title_text="Train Loss", type="log", row=1, col=1)
    
    # 2. 验证损失
    val_loss_valid = val_loss[~np.isnan(val_loss)]
    if len(val_loss_valid) > 0:
        val_steps = global_step[~np.isnan(val_loss)]
        val_loss_clean = val_loss[~np.isnan(val_loss)]
        
        # 验证损失通常点数较少，但也要降采样以防万一
        val_steps_ds, val_loss_ds = downsample_data(val_steps, val_loss_clean, max_points)
        
        if smooth_window > 0 and len(val_loss_clean) > smooth_window:
            val_loss_smooth = smooth_data(val_loss_clean, min(smooth_window, len(val_loss_clean)))
            val_steps_smooth_ds, val_loss_smooth_ds = downsample_data(val_steps, val_loss_smooth, max_points)
            
            fig.add_trace(
                go.Scatter(
                    x=val_steps_ds,
                    y=val_loss_ds,
                    mode='markers',
                    name='Val Loss (Raw)',
                    marker=dict(color='rgba(255, 127, 14, 0.5)', size=4),
                    hovertemplate='Step: %{x:,}<br>Loss: %{y:.6f}<extra></extra>'
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=val_steps_smooth_ds,
                    y=val_loss_smooth_ds,
                    mode='lines',
                    name='Val Loss (Smoothed)',
                    line=dict(color='rgb(255, 127, 14)', width=2),
                    hovertemplate='Step: %{x:,}<br>Smoothed Loss: %{y:.6f}<extra></extra>'
                ),
                row=2, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=val_steps_ds,
                    y=val_loss_ds,
                    mode='markers+lines',
                    name='Val Loss',
                    marker=dict(color='rgb(255, 127, 14)', size=5),
                    line=dict(color='rgb(255, 127, 14)', width=1.5),
                    hovertemplate='Step: %{x:,}<br>Loss: %{y:.6f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text="Global Step", row=2, col=1)
        fig.update_yaxes(title_text="Validation Loss", type="log", row=2, col=1)
    
    # 3. 训练 vs 验证损失对比（按epoch）
    if len(train_loss_valid) > 0 and len(val_loss_valid) > 0:
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
        
        fig.add_trace(
            go.Scatter(
                x=epochs_unique,
                y=train_loss_by_epoch,
                mode='lines+markers',
                name='Train Loss',
                line=dict(color='rgb(31, 119, 180)', width=2),
                marker=dict(size=6),
                hovertemplate='Epoch: %{x}<br>Loss: %{y:.6f}<extra></extra>'
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=epochs_unique,
                y=val_loss_by_epoch,
                mode='lines+markers',
                name='Val Loss',
                line=dict(color='rgb(255, 127, 14)', width=2),
                marker=dict(size=6, symbol='square'),
                hovertemplate='Epoch: %{x}<br>Loss: %{y:.6f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Loss", type="log", row=2, col=2)
    
    # 4. 学习率曲线
    lr_valid = lr[~np.isnan(lr)]
    if len(lr_valid) > 0:
        # 降采样学习率数据
        step_lr_ds, lr_ds = downsample_data(global_step, lr, max_points)
        
        fig.add_trace(
            go.Scatter(
                x=step_lr_ds,
                y=lr_ds,
                mode='lines',
                name='Learning Rate',
                line=dict(color='rgb(44, 160, 44)', width=2),
                hovertemplate='Step: %{x:,}<br>LR: %{y:.2e}<extra></extra>'
            ),
            row=3, col=1
        )
        fig.update_xaxes(title_text="Global Step", row=3, col=1)
        fig.update_yaxes(title_text="Learning Rate", type="log", row=3, col=1)
    
    # 5. 训练动作MSE误差
    mse_valid = mse_error[~np.isnan(mse_error)]
    if len(mse_valid) > 0:
        mse_steps = global_step[~np.isnan(mse_error)]
        mse_clean = mse_error[~np.isnan(mse_error)]
        
        # 降采样MSE数据
        mse_steps_ds, mse_clean_ds = downsample_data(mse_steps, mse_clean, max_points)
        
        if smooth_window > 0 and len(mse_clean) > smooth_window:
            mse_smooth = smooth_data(mse_clean, min(smooth_window, len(mse_clean)))
            mse_steps_smooth_ds, mse_smooth_ds = downsample_data(mse_steps, mse_smooth, max_points)
            
            fig.add_trace(
                go.Scatter(
                    x=mse_steps_ds,
                    y=mse_clean_ds,
                    mode='markers',
                    name='MSE Error (Raw)',
                    marker=dict(color='rgba(214, 39, 40, 0.5)', size=4),
                    hovertemplate='Step: %{x:,}<br>MSE: %{y:.2f}<extra></extra>'
                ),
                row=3, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=mse_steps_smooth_ds,
                    y=mse_smooth_ds,
                    mode='lines',
                    name='MSE Error (Smoothed)',
                    line=dict(color='rgb(214, 39, 40)', width=2),
                    hovertemplate='Step: %{x:,}<br>Smoothed MSE: %{y:.2f}<extra></extra>'
                ),
                row=3, col=2
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=mse_steps_ds,
                    y=mse_clean_ds,
                    mode='markers+lines',
                    name='Action MSE Error',
                    marker=dict(color='rgb(214, 39, 40)', size=5),
                    line=dict(color='rgb(214, 39, 40)', width=1.5),
                    hovertemplate='Step: %{x:,}<br>MSE: %{y:.2f}<extra></extra>'
                ),
                row=3, col=2
            )
        
        fig.update_xaxes(title_text="Global Step", row=3, col=2)
        fig.update_yaxes(title_text="Action MSE Error", row=3, col=2)
    
    # 更新子图标题样式，增加字体大小和间距
    fig.update_annotations(
        font_size=14,
        yshift=10  # 增加标题上移，避免和axis重合
    )
    
    # 更新所有坐标轴，增加标题和标签的间距
    fig.update_xaxes(title_font_size=12, title_standoff=15)
    fig.update_yaxes(title_font_size=12, title_standoff=20)
    
    # 更新布局
    final_loss = train_loss_valid[-1] if len(train_loss_valid) > 0 else float('nan')
    final_loss_str = f"{final_loss:.6f}" if not np.isnan(final_loss) else "N/A"
    fig.update_layout(
        height=height,
        title_text=(
            "Training Logs Visualization<br>"
            f"<sub>Total Steps: {int(global_step[-1]):,} | "
            f"Total Epochs: {int(epoch[-1]):,} | "
            f"Final Train Loss: {final_loss_str}</sub>"
        ),
        title_x=0.5,
        title_font_size=18,
        title_pad=dict(t=20, b=10),  # 增加title的上下padding
        margin=dict(t=100, b=50, l=50, r=150),  # 增加右边距，为legend留出空间
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',  # 使用白色主题，更现代
        legend=dict(
            orientation="v",  # 垂直排列，避免一行太长
            yanchor="top",
            y=0.98,  # 放在图表顶部右侧
            xanchor="right",
            x=1.02,  # 稍微超出右边界
            font_size=10,
            bgcolor="rgba(255,255,255,0.8)",  # 半透明背景
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            itemwidth=30,
            tracegroupgap=5  # legend items之间的间距
        )
    )
    
    # 保存
    if save_html:
        fig.write_html(save_html)
        print(f"HTML saved to: {save_html}")
    
    if save_png:
        fig.write_image(save_png, width=1600, height=height, scale=2)
        print(f"PNG saved to: {save_png}")
    
    if show_plot:
        fig.show()
    
    return fig


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
        description='Visualize training logs with Plotly (beautiful interactive charts)',
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
        '--save-html',
        type=str,
        default=None,
        help='Path to save HTML file (default: auto-generate in log file directory)'
    )
    parser.add_argument(
        '--save-png',
        type=str,
        default=None,
        help='Path to save PNG file (requires kaleido, default: don\'t save)'
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
    parser.add_argument(
        '--height',
        type=int,
        default=900,
        help='Chart height in pixels (default: 900)'
    )
    parser.add_argument(
        '--max-points',
        type=int,
        default=10000,
        help='Maximum number of data points per curve (for downsampling, default: 10000). '
             'Set to 0 to disable downsampling.'
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
    save_html = args.save_html
    if save_html is None and not args.no_show:
        # 自动生成HTML路径
        save_html = str(log_file.parent / f"training_curves_{log_file.stem}.html")
    
    save_png = args.save_png
    
    # 绘制图表
    print("Plotting curves...")
    max_points = args.max_points if args.max_points > 0 else None
    if max_points is None:
        max_points = len(metrics['global_step'])  # 不降采样
    
    plot_training_curves(
        metrics,
        smooth_window=args.smooth,
        max_points=max_points,
        save_html=save_html,
        save_png=save_png,
        show_plot=not args.no_show,
        height=args.height
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()

