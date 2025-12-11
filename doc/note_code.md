# playload（checkpoint）

- **`state_dicts` 里：**
  - 一定有：`model` 的参数、`optimizer` 的参数。
  - 如果开启 EMA（`cfg.training.use_ema = True`），还会多存一个：`ema_model` 的参数。
  - 它们都是通过各自的 `state_dict()/load_state_dict()` 来保存和恢复的。

- **`pickles` 里：**
  - 总是有：`global_step`（当前总步数）、`epoch`（当前轮数）、`_output_dir`（输出目录）。
  - 这三个是直接用 `dill` 整体序列化/反序列化，把整个对象恢复回来。


# 什么是 LR Scheduler（学习率调度器）

- **核心定义**：  
  **LR Scheduler（学习率调度器）就是一个按照预先设定的规则，在训练过程中自动改变学习率的“策略/函数”。**

- **为什么要用它**：  
  - 刚开始训练时，可能希望 **学习率大一点**，收敛得快（或者先 warmup，从小到大缓一缓）。  
  - 训练到后期，希望 **学习率逐渐变小**，让模型在“收尾”阶段别乱动太大步，更稳定地收敛。  
  - 有时候还会设计周期性变化（cosine 退火之类），在局部最优附近多探索。

- **它具体做什么**：  
  - 你一开始给优化器（比如 AdamW）一个基础 `lr`。  
  - 每过一个 step / epoch，你调用一次 `lr_scheduler.step()`。  
  - Scheduler 根据当前的 step / epoch，**算出一个新的学习率**，改到优化器里去。  
  - 优化器后续的梯度更新就会用这个新的学习率。

- **在你这个项目里的用法**（结合 `train_diffusion_unet_hybrid_workspace.py`）：  
  - 用的是 `get_scheduler(name="cosine", ...)`（在 yaml 里配置的），底层来自 HuggingFace diffusers。  
  - 根据：
    - 总训练步数 `num_training_steps`  
    - warmup 步数 `num_warmup_steps`  
    - 当前已经训练到的 `global_step`  
    生成一个 **“先 warmup、再按余弦曲线缓慢下降”** 的学习率随 step 变化的函数。

- **一句话总结**：  
  **优化器决定“怎么走”（用什么公式更新参数），LR Scheduler 决定“每一步走多大”（学习率随时间怎么变）。**

# Step、Episode、Batch、Epoch 的关系

### 基本定义

1. **Episode（回合/轨迹）**
   - 一条完整的专家演示轨迹
   - 例如：机器人从起点到完成任务的完整过程
   - 你的数据：206 个 episodes

2. **Batch（批次）**
   - 一次送入模型的数据量
   - `batch_size = 64`：每次处理 64 个样本
   - 每个 batch 包含多个 episode 的片段

3. **Step（步数）**
   - 处理一个 batch 并更新一次参数 = 1 step
   - `global_step`：跨 epoch 累计的总步数
   - 每个 epoch 有 168 个 steps（168 个 batches）

4. **Epoch（轮次）**
   - 完整遍历一次训练集 = 1 epoch
   - 你的配置：3050 个 epochs

### 层级关系

```
Episode（206个）
    ↓ 切分成片段
Batch（每个epoch 168个，每个batch包含64个样本）
    ↓ 逐个处理
Step（每个batch = 1 step，每个epoch = 168 steps）
    ↓ 累积
Epoch（3050个）
```

### 数量关系（你的项目）

```
1 Epoch = 168 Batches = 168 Steps
1 Batch = 64 个样本（来自不同episodes的片段）
总 Steps = 168 × 3050 = 512,400 steps
总 Episodes = 206 个（原始数据）
```

### 训练流程示例

```python
for epoch in range(3050):           # 3050个epoch
    for batch in train_dataloader:  # 每个epoch 168个batch
        # 处理1个batch = 1个step
        loss = model(batch)          # batch包含64个样本
        loss.backward()
        optimizer.step()             # 更新参数
        global_step += 1             # step计数+1
```

### 记忆口诀

- **Episode**：原始数据单位（一条完整轨迹）
- **Batch**：训练单位（一次处理的数据量）
- **Step**：更新单位（处理一个 batch = 1 step）
- **Epoch**：遍历单位（完整过一遍数据 = 1 epoch）

**关系**：多个 Episodes → 切分成 Batches → 每个 Batch 产生 1 Step → 多个 Steps 组成 1 Epoch