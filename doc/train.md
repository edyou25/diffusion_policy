# Diffusion Policy 训练时间估算

## 📊 实际测量数据

### 第一个Epoch (Epoch 0) 实际耗时
- **开始时间**: 2025-12-03 15:00:11
- **结束时间**: 2025-12-03 15:21:23  
- **总耗时**: **约21分钟**

### 训练详情
```
- 训练batches: 141个
- Batch大小: 64
- 学习步数: 167 steps (global_step: 0→166)
- 包含操作:
  ✓ 141个batch的前向+反向传播
  ✓ 验证集评估
  ✓ Rollout环境评估（生成6个视频）
  ✓ 扩散采样评估
```

---

## ⏱️ 时间分解分析

### 单个Epoch时间构成

#### 1. 纯训练时间（每个epoch都有）
- **141个batch × 约5-8秒/batch** ≈ **12-19分钟**
- 包括：前向传播、反向传播、优化器更新、EMA更新

#### 2. 验证时间（每个epoch都有，val_every: 1）
- **验证集评估** ≈ **10-20秒**
- 约3个batch的验证数据

#### 3. 采样评估（每5个epoch，sample_every: 5）
- **扩散采样测试** ≈ **5-10秒**
- 使用一个训练batch测试动作预测质量

#### 4. Rollout评估（每50个epoch，rollout_every: 50）
- **环境rollout + 视频生成** ≈ **2-5分钟**
- 运行50个测试episode（n_test: 50）
- 生成6个可视化视频（n_train_vis: 2, n_test_vis: 4）

#### 5. 检查点保存（每50个epoch，checkpoint_every: 50）
- **保存检查点** ≈ **5-15秒**
- 模型参数约251MB + 优化器状态

---

## 📈 训练时长估算

### 场景1：普通Epoch（无rollout）
```
训练时间:     14分钟
验证时间:     15秒
采样评估:     7秒  (每5个epoch)
────────────────────
总计:         ~15分钟/epoch
```

### 场景2：带Rollout的Epoch（每50个）
```
训练时间:     14分钟
验证时间:     15秒
采样评估:     7秒
Rollout:      3.5分钟
检查点保存:   10秒
────────────────────
总计:         ~18分钟/epoch
```

---

## 🎯 总训练时间估算

### 配置参数
- **总Epochs**: 3050
- **Rollout频率**: 每50个epoch
- **带Rollout的Epochs**: 3050 ÷ 50 = 61个
- **普通Epochs**: 3050 - 61 = 2989个

### 计算
```
普通Epochs:  2989 × 15分钟  = 44,835分钟  (747.25小时)
Rollout:        61 × 18分钟  =  1,098分钟   (18.3小时)
────────────────────────────────────────────────────────
总计:                         45,933分钟   (765.55小时)
                                           ≈ 32天
```

### 考虑实际情况（更保守估计）

实测第一个epoch（包含rollout）用了21分钟，说明：
- GPU可能运行较慢（RTX 5070兼容性问题）
- 或者首次运行有额外开销

**按21分钟/epoch（保守）计算**：
```
3050 epochs × 21分钟 = 64,050分钟
                     = 1067.5小时
                     ≈ 44.5天
                     ≈ 1.5个月
```

**按实际混合估算**：
```
- 普通epoch: 15分钟
- 带rollout: 21分钟
- 加权平均: (2989×15 + 61×21) / 3050 ≈ 15.1分钟

总时间: 3050 × 15.1分钟 = 46,055分钟
                        = 767.58小时
                        ≈ 32天
```

---

## 📊 训练进度里程碑

### 关键时间节点

| Epoch | 进度 | 预计完成时间（从现在开始） | 说明 |
|-------|------|--------------------------|------|
| 50 | 1.6% | ~17小时 | 第一次检查点保存 |
| 100 | 3.3% | ~1.4天 | |
| 500 | 16.4% | ~7.5天 | 初步效果可见 |
| 1000 | 32.8% | ~15天 | 中期检查 |
| 1500 | 49.2% | ~23天 | 接近一半 |
| 2000 | 65.6% | ~30天 | |
| 2500 | 82.0% | ~38天 | 后期训练 |
| 3050 | 100% | ~46天 | 训练完成 |

### 每日进度
```
- 每天可完成: ~68个epochs
- 每天进度: ~2.2%
```

---

## 💡 优化建议

### 1. 减少Rollout频率（可节省10-15%时间）
```yaml
training:
  rollout_every: 100  # 从50改为100
```
节省时间: 约3-7天

### 2. 减少验证频率
```yaml
training:
  val_every: 5  # 从1改为5（每5个epoch验证一次）
```
节省时间: 约5-10%

### 3. 调整检查点策略
```yaml
checkpoint:
  save_last_ckpt: true
  topk:
    k: 3  # 从5改为3，只保留最好的3个
```
节省存储空间，对时间影响较小

### 4. 升级PyTorch（最重要！）
```bash
# 安装支持RTX 5070的PyTorch 2.5+
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```
**潜在提速: 20-50%** （可能将训练时间缩短到20-30天）

### 5. 早停策略
如果性能达到满意水平，可以提前停止：
```yaml
training:
  num_epochs: 1500  # 减少到1500个epoch
```
时间减半: ~15-20天

### 6. 使用更大的Batch Size（如果GPU内存允许）
```yaml
dataloader:
  batch_size: 128  # 从64增加到128
```
可能提速: 10-20%

---

## 🎮 GPU性能影响

### 当前状态（RTX 5070 + PyTorch 1.12.1）
- ⚠️ CUDA sm_120 不被支持
- 可能通过兼容层运行
- **性能损失估计: 20-40%**

### 理想状态（升级PyTorch后）
- ✅ 原生sm_120支持
- 完整性能利用
- **预计训练时间: 20-30天**

---

## 📈 实时监控

### 查看当前进度
```bash
# 查看最新日志
tail -5 data/outputs/2025.12.03/15.00.11_train_diffusion_unet_hybrid_pusht_image/logs.json.txt

# 计算当前进度
# 当前epoch / 3050 × 100%
```

### 估算剩余时间
```python
import json

# 读取日志
with open('logs.json.txt', 'r') as f:
    logs = [json.loads(line) for line in f if line.strip()]

# 获取当前epoch
current_epoch = logs[-1]['epoch']
total_epochs = 3050

# 估算每个epoch时间（分钟）
time_per_epoch = 15.1

# 剩余时间
remaining_epochs = total_epochs - current_epoch - 1
remaining_minutes = remaining_epochs * time_per_epoch
remaining_hours = remaining_minutes / 60
remaining_days = remaining_hours / 24

print(f"当前Epoch: {current_epoch}/{total_epochs}")
print(f"进度: {current_epoch/total_epochs*100:.2f}%")
print(f"预计剩余时间: {remaining_days:.1f}天 ({remaining_hours:.1f}小时)")
```

---

## 🎯 总结

### 最可能的训练时长
```
🕐 保守估计: 45-50天  (当前RTX 5070兼容性问题)
🕐 理想估计: 25-35天  (升级PyTorch后)
🕐 优化后:    15-25天  (升级+减少epoch数)
```

### 建议策略
1. **短期**: 先训练500-1000个epoch观察效果（约7-15天）
2. **中期**: 如果效果良好，考虑在1500-2000 epoch时提前停止
3. **长期**: 升级PyTorch以获得完整GPU性能

### 实际论文复现
根据论文，Push-T任务在较早的epoch（~500-1000）就能达到不错的效果，不一定需要训练满3050个epoch。

---

## 📞 实时状态检查

### 快速检查命令
```bash
# 检查训练是否还在运行
ps aux | grep train.py

# 查看当前epoch
tail -1 logs.json.txt | python -c "import sys, json; print('Epoch:', json.load(sys.stdin)['epoch'])"

# 查看GPU使用
watch -n 1 nvidia-smi

# 查看WandB
# https://wandb.ai/edward-you-the-hong-kong-polytechnic-university/diffusion_policy_debug
```

