#!/usr/bin/env python3
"""
训练日志可视化脚本 - 使用 Plotly（交互式图表）

用法:
  python doc/visualize_training_logs_plotly.py --exp-dir <experiment目录> [选项]

示例:
  python doc/visualize_training_logs_plotly.py \
    --exp-dir data/experiments/low_dim/square_ph/diffusion_policy_cnn/data/experiments/low_dim/square_ph/diffusion_policy_cnn \
    --smooth 200 --max-points 6000 --save-html training_curves_plotly.html
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def smooth_series(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size == 0:
        return y
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(y, kernel, mode="same")


def downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or x.size <= max_points:
        return x, y
    stride = max(1, math.ceil(x.size / max_points))
    return x[::stride], y[::stride]


def collect_run_metrics(log_path: Path, keys: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    series: Dict[str, Dict[str, List[float]]] = {k: {"x": [], "y": []} for k in keys}
    for rec in iter_jsonl(log_path):
        if "global_step" not in rec:
            continue
        step = rec["global_step"]
        for key in keys:
            if key in rec:
                series[key]["x"].append(float(step))
                series[key]["y"].append(float(rec[key]))
    return {k: (np.array(v["x"]), np.array(v["y"])) for k, v in series.items()}


def collect_metrics_agg(log_path: Path, keys: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    series: Dict[str, Dict[str, List[float]]] = {k: {"x": [], "y": []} for k in keys}
    for rec in iter_jsonl(log_path):
        if "epoch" not in rec:
            continue
        epoch = rec["epoch"]
        for key in keys:
            if key in rec:
                series[key]["x"].append(float(epoch))
                series[key]["y"].append(float(rec[key]))
    return {k: (np.array(v["x"]), np.array(v["y"])) for k, v in series.items()}


def add_series(
    fig: go.Figure,
    x: np.ndarray,
    y: np.ndarray,
    row: int,
    col: int,
    name: str,
    color: Optional[str] = None,
    dash: Optional[str] = None,
):
    line = dict(color=color, dash=dash) if color or dash else None
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", name=name, line=line),
        row=row,
        col=col,
    )


def main():
    parser = argparse.ArgumentParser(description="Plotly 可视化训练日志")
    parser.add_argument("--exp-dir", type=Path, required=True, default="/home/yyf/IROS2026/diffusion_policy/data/experiments/low_dim/square_ph/diffusion_policy_cnn/", help="experiment 目录路径")
    parser.add_argument("--smooth", type=int, default=1, help="平滑窗口大小（帧数）")
    parser.add_argument("--max-points", type=int, default=5000, help="每条曲线最大点数")
    parser.add_argument("--save-html", type=Path, default=None, help="保存 HTML 路径")
    parser.add_argument("--no-show", action="store_true", help="不弹出图窗")
    args = parser.parse_args()

    exp_dir = args.exp_dir
    metrics_file = exp_dir / "metrics" / "logs.json.txt"
    train_dirs = sorted([p for p in exp_dir.glob("train_*") if p.is_dir()])

    if not metrics_file.exists() and not train_dirs:
        raise SystemExit(f"No logs found under {exp_dir}")

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{}, None]],
        subplot_titles=(
            "Train vs Val Loss (per run)",
            "Learning Rate (per run)",
            "Test Mean Score (metrics/logs.json.txt)",
        ),
        vertical_spacing=0.12,
    )

    # Train logs per run
    train_keys = ["train_loss", "val_loss", "lr"]
    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for idx, run_dir in enumerate(train_dirs):
        run_log = run_dir / "logs.json.txt"
        if not run_log.exists():
            continue
        run_name = run_dir.name
        series = collect_run_metrics(run_log, train_keys)
        for k, (x, y) in series.items():
            if x.size == 0:
                continue
            y_s = smooth_series(y, args.smooth)
            x_s, y_s = downsample(x, y_s, args.max_points)
            color = color_cycle[idx % len(color_cycle)]
            label = f"{run_name} · {k}"
            if k == "train_loss":
                add_series(fig, x_s, y_s, 1, 1, label, color=color)
            elif k == "val_loss":
                add_series(fig, x_s, y_s, 1, 1, label, color=color, dash="dot")
            elif k == "lr":
                add_series(fig, x_s, y_s, 1, 2, label, color=color)

    # Aggregated metrics
    if metrics_file.exists():
        metric_keys = [
            "max/test_mean_score",
            "k_min_train_loss/test_mean_score",
            "min_val_loss/test_mean_score",
            "last/test_mean_score",
        ]
        metric_series = collect_metrics_agg(metrics_file, metric_keys)
        metric_colors = {
            "max/test_mean_score": "#1f77b4",
            "k_min_train_loss/test_mean_score": "#ff7f0e",
            "min_val_loss/test_mean_score": "#2ca02c",
            "last/test_mean_score": "#d62728",
        }
        for key, (x, y) in metric_series.items():
            if x.size == 0:
                continue
            y_s = smooth_series(y, max(1, args.smooth // 5))
            x_s, y_s = downsample(x, y_s, max(200, args.max_points // 10))
            add_series(fig, x_s, y_s, 2, 1, key, color=metric_colors.get(key))

    fig.update_layout(
        title=f"Training Logs Visualization<br><sub>{exp_dir.name}</sub>",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=60, r=40, t=120, b=120),
        height=900,
    )

    fig.update_xaxes(title_text="Global Step", row=1, col=1)
    fig.update_xaxes(title_text="Global Step", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)

    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="LR", row=1, col=2)
    fig.update_yaxes(title_text="Test Mean Score", row=2, col=1)

    if args.save_html:
        args.save_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(args.save_html))
        print(f"Saved: {args.save_html}")

    if not args.no_show:
        fig.show()


if __name__ == "__main__":
    main()
