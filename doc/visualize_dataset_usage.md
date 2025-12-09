# PushT 数据集可视化工具使用说明（更新版）

## ✨ 新功能

- ✅ **连续播放所有帧**（而不是只显示前8帧）
- ✅ **在图像上绘制 state 信息**（agent 位置、动作向量）
- ✅ **显示运动轨迹**（过去10帧的轨迹线）
- ✅ **可调帧率**（默认 10fps）

## 基本用法

### 1. 快速开始（生成前 3 个样本的动画）
```bash
cd /home/yyf/diffusion_policy
python doc/visualize_pusht_dataset.py --num_samples 3
```

输出目录：`data/vis_pusht/`

### 2. 调整帧率和显示选项
```bash
# 高帧率（15 fps），更流畅
python doc/visualize_pusht_dataset.py --num_samples 5 --fps 15

# 不显示轨迹线（只显示当前位置）
python doc/visualize_pusht_dataset.py --num_samples 3 --no_trajectory

# 低帧率（5 fps），更慢速观察
python doc/visualize_pusht_dataset.py --num_samples 2 --fps 5
```

### 3. 随机采样和统计
```bash
# 随机选择 5 个样本 + 生成统计图表
python doc/visualize_pusht_dataset.py --num_samples 5 --random --stats
```

### 4. 保存每帧为单独图像
```bash
python doc/visualize_pusht_dataset.py --num_samples 2 --save_frames
```

这会创建 `sample_XXX_frames/` 目录，包含：
- `frame_000_raw.png` - 原始帧
- `frame_000_annotated.png` - 带标注的帧

### 5. 可视化验证集
```bash
python doc/visualize_pusht_dataset.py --validation --num_samples 3
```

## 完整参数说明

```
--zarr_path PATH          Zarr 数据集路径（默认：data/pusht/pusht_cchi_v7_replay.zarr）
--output_dir DIR          输出目录（默认：data/vis_pusht）
--num_samples N           可视化的样本数量（默认：5）
--random                  随机选择样本而不是前 N 个
--save_frames             保存每帧为单独的图像文件
--stats                   生成数据集统计信息图表
--validation              使用验证集而非训练集
--fps N                   GIF 帧率（默认：10）
--no_trajectory           不显示运动轨迹线
```

## 输出文件说明

对每个样本，会生成：

### 1. **`sample_XXX_raw.gif`**  
- 纯图像序列 GIF，无任何标注
- **所有 16 帧连续播放**
- 展示原始图像数据

### 2. **`sample_XXX_annotated.gif`** ⭐ 主要输出
- 带完整标注的动画 GIF
- **包含的标注信息：**
  - 🔴 **Agent 位置**：红色圆点（白色边框）
  - 🔵 **动作向量**：蓝色箭头，从 agent 指向目标方向
  - 🔴 **运动轨迹**：淡红色线条，显示过去10帧的路径
  - 📊 **文本信息**：
    - `t=X/15`：当前帧 / 总帧数
    - `pos:(x, y)`：agent 当前位置
    - `act:(dx, dy)`：当前动作指令

### 3. **`sample_XXX_comparison.png`**
- 对比图（前8帧）
- 上排：原始图像
- 下排：带标注的图像
- 方便快速查看标注效果

### 4. **`sample_XXX_frames/`**（可选，需 `--save_frames`）
- 每一帧单独的 PNG 文件
- `frame_000_raw.png`：原始帧
- `frame_000_annotated.png`：标注帧
- 适合逐帧分析或制作自定义视频

### 5. **`dataset_statistics.png`**（可选，需 `--stats`）
- 动作 X/Y 分布直方图
- Agent 位置散点图
- 动作向量场

## 标注说明

### Agent 位置（红色圆点）
- 从 `state[:, :2]` 提取
- 坐标从 [0, 512] 映射到 [0, 96] 图像坐标

### 动作向量（蓝色箭头）
- 表示控制指令 `(delta_x, delta_y)`
- 箭头长度 = 动作幅度 × 0.1（缩放以适应显示）
- 箭头方向 = 动作方向

### 运动轨迹（淡红色线条）
- 显示过去 10 帧的运动路径
- 颜色渐变：越近的轨迹越亮
- 帮助理解 agent 的运动模式

## 使用场景

### 场景 1：快速检查数据是否正常
```bash
python doc/visualize_pusht_dataset.py --num_samples 3
```
查看 `sample_*_annotated.gif`，确认：
- 图像序列是否连贯
- Agent 位置是否合理
- 动作是否与运动一致

### 场景 2：分析特定样本的运动模式
```bash
# 慢速播放（5fps），显示轨迹
python doc/visualize_pusht_dataset.py --num_samples 1 --fps 5 --save_frames
```
逐帧查看图像，分析：
- Agent 如何接近 block
- 轨迹是否平滑
- 动作指令是否合理

### 场景 3：对比不同样本的策略
```bash
python doc/visualize_pusht_dataset.py --num_samples 10 --random
```
查看多个样本，比较：
- 不同初始条件下的策略
- 成功和失败的案例
- 运动轨迹的多样性

### 场景 4：数据集整体分析
```bash
python doc/visualize_pusht_dataset.py --num_samples 5 --stats
```
结合 GIF 和统计图表，了解：
- 动作分布是否平衡
- Agent 活动范围
- 策略的整体特征

### 场景 5：验证集 vs 训练集
```bash
# 训练集
python doc/visualize_pusht_dataset.py --num_samples 3 --output_dir data/vis_train

# 验证集
python doc/visualize_pusht_dataset.py --validation --num_samples 3 --output_dir data/vis_val
```
对比两个目录，检查数据划分是否合理。

## 技术细节

### 数据流程
```
PushTImageDataset[idx]
  ↓
{
  'obs': {
    'image': (16, 3, 96, 96),      # 归一化到 [0,1]
    'agent_pos': (16, 2)            # 像素坐标 [0, 512]
  },
  'action': (16, 2)                 # 控制指令
}
  ↓
可视化处理
  ↓
- 图像 × 255 → uint8
- 坐标映射：512 → 96
- 绘制标注（cv2）
  ↓
保存为 GIF（imageio）
```

### 坐标系统
- **原始坐标**：[0, 512] × [0, 512]（模拟器坐标）
- **图像坐标**：[0, 96] × [0, 96]（显示坐标）
- **转换公式**：`img_coord = orig_coord * 96 / 512`

### 绘制参数
```python
agent_circle_radius = 4          # agent 圆点半径
action_arrow_scale = 0.1         # 动作箭头缩放
trajectory_history = 10          # 显示过去10帧轨迹
trajectory_fade = True           # 轨迹渐变透明
```

### GIF 参数
- 默认帧率：10 fps
- 循环播放：loop=0（无限循环）
- 可调范围：5-30 fps

## 常见问题

**Q: 为什么标注 GIF 比原始 GIF 大？**  
A: 因为添加了图形和文本，压缩率降低。可以调低 fps 减小文件大小。

**Q: Agent 位置红点看不见？**  
A: 可能坐标映射有问题。检查 `agent_pos` 的范围是否在 [0, 512]。

**Q: 动作箭头太短/太长？**  
A: 编辑脚本第 89 行，修改 `action_scale` 参数（默认 0.1）。

**Q: 如何修改轨迹长度？**  
A: 编辑脚本第 79 行，修改 `range(max(0, t-10), t)` 中的 10 为其他值。

**Q: 如何更改标注颜色？**  
A: 修改 cv2 绘图函数的颜色参数（BGR 格式）：
- Agent：`(255, 0, 0)` 红色
- 箭头：`(0, 0, 255)` 蓝色
- 轨迹：`(int(255*alpha), 0, 0)` 渐变红

**Q: 播放速度太快/太慢？**  
A: 使用 `--fps` 参数调整，推荐范围：
- 慢速观察：5-8 fps
- 正常速度：10-12 fps
- 快速浏览：15-20 fps

**Q: 如何在图像上显示 block 位置？**  
A: 目前 `PushTImageDataset` 只返回 `agent_pos`（state 的前2维）。如果需要显示 block，需要：
1. 修改数据集加载，提取完整的 state（5维）
2. 在脚本中绘制 block 位置和角度
3. state 格式：`(agent_x, agent_y, block_x, block_y, block_theta)`

## 性能优化

### 减少处理时间
```bash
# 只生成必要的输出
python doc/visualize_pusht_dataset.py --num_samples 3  # 不用 --save_frames

# 降低帧率
python doc/visualize_pusht_dataset.py --num_samples 5 --fps 8
```

### 批量处理
```bash
# 分批处理大量样本
for i in 0 10 20 30; do
    python doc/visualize_pusht_dataset.py --num_samples 10 \
        --output_dir data/vis_pusht_batch_$i
done
```

## 代码位置
- 脚本：`/home/yyf/diffusion_policy/doc/visualize_pusht_dataset.py`
- 数据集类：`/home/yyf/diffusion_policy/diffusion_policy/dataset/pusht_image_dataset.py`
- 数据文件：`/home/yyf/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr`
