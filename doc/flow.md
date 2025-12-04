# Diffusion Policy 代码执行流程详解

## 总体架构

```
train.py (入口)
    ↓
Hydra配置加载 (image_pusht_diffusion_policy_cnn.yaml)
    ↓
TrainDiffusionUnetHybridWorkspace (工作空间)
    ├── __init__ (初始化)
    │   ├── 设置随机种子
    │   ├── 实例化Policy模型
    │   ├── 创建EMA模型副本
    │   └── 配置优化器
    └── run (训练循环)
        ├── 加载数据集
        ├── 配置训练组件
        ├── 初始化WandB
        ├── 迁移模型到GPU
        └── 开始训练循环 (epoch循环)
```

---

## 第一阶段：程序入口 (train.py)

### 1.1 文件位置
```
/home/yyf/diffusion_policy/train.py
```

### 1.2 核心代码
```python
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()
```

### 1.3 执行步骤

**步骤1: Hydra装饰器处理**
- 读取配置文件：`diffusion_policy/config/image_pusht_diffusion_policy_cnn.yaml`
- 合并命令行参数（如 `training.seed=42`）
- 创建输出目录（根据 `hydra.run.dir` 参数）

**步骤2: 解析配置**
```python
OmegaConf.resolve(cfg)
```
- 解析所有 `${...}` 变量引用
- 解析 `${eval:'...'}` 表达式
- 解析 `${now:%Y.%m.%d}` 时间戳

**步骤3: 动态加载Workspace类**
```python
cls = hydra.utils.get_class(cfg._target_)
# cfg._target_ = "diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace.TrainDiffusionUnetHybridWorkspace"
```

**步骤4: 实例化并运行**
```python
workspace = cls(cfg)      # 调用 __init__
workspace.run()           # 调用 run 方法
```

---

## 第二阶段：工作空间初始化 (__init__)

### 2.1 文件位置
```
/home/yyf/diffusion_policy/diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py
```

### 2.2 初始化流程

```python
def __init__(self, cfg: OmegaConf, output_dir=None):
    super().__init__(cfg, output_dir=output_dir)
    
    # 1. 设置随机种子
    seed = cfg.training.seed  # 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 2. 实例化Policy模型
    self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
    
    # 3. 创建EMA模型（指数移动平均）
    self.ema_model = None
    if cfg.training.use_ema:
        self.ema_model = copy.deepcopy(self.model)
    
    # 4. 配置优化器
    self.optimizer = hydra.utils.instantiate(
        cfg.optimizer, params=self.model.parameters())
    
    # 5. 初始化训练状态
    self.global_step = 0
    self.epoch = 0
```

### 2.3 关键组件说明

**Policy模型实例化**
```yaml
# 配置文件中的定义
policy:
  _target_: diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy
  horizon: 16
  n_obs_steps: 2
  n_action_steps: 8
  # ... 其他参数
```

实际执行：
```python
self.model = DiffusionUnetHybridImagePolicy(
    shape_meta={...},
    noise_scheduler=DDPMScheduler(...),
    horizon=16,
    n_obs_steps=2,
    n_action_steps=8,
    # ...
)
```

**模型包含**：
- 视觉编码器（ResNet18）：处理图像观测
- 条件U-Net 1D：扩散模型主体
- 噪声调度器（DDPM）：控制扩散过程

---

## 第三阶段：训练准备 (run方法 - Part 1)

### 3.1 恢复检查点（如果存在）

```python
if cfg.training.resume:
    lastest_ckpt_path = self.get_checkpoint_path()
    if lastest_ckpt_path.is_file():
        print(f"Resuming from checkpoint {lastest_ckpt_path}")
        self.load_checkpoint(path=lastest_ckpt_path)
```

**检查点内容**：
- 模型参数（model.state_dict）
- 优化器状态（optimizer.state_dict）
- EMA模型参数
- 当前epoch和global_step

### 3.2 加载数据集

```python
# 实例化数据集
dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
# cfg.task.dataset._target_ = "diffusion_policy.dataset.pusht_image_dataset.PushTImageDataset"

# 创建数据加载器
train_dataloader = DataLoader(dataset, **cfg.dataloader)
# cfg.dataloader: {batch_size: 64, num_workers: 8, ...}

# 获取归一化器
normalizer = dataset.get_normalizer()
```

**数据集结构**：
```python
dataset[i] = {
    'obs': {
        'image': Tensor[2, 3, 96, 96],    # 过去2帧图像
        'agent_pos': Tensor[2, 2]          # 过去2个位置
    },
    'action': Tensor[16, 2]                # 未来16步动作
}
```

**数据来源**：
- Zarr格式文件：`data/pusht/pusht_cchi_v7_replay.zarr`
- 包含206个演示轨迹
- 共25650个时间步

### 3.3 配置验证数据集

```python
val_dataset = dataset.get_validation_dataset()
val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
```

**数据划分**：
- 训练集：90个episode（配置中的max_train_episodes）
- 验证集：剩余episode的2%（val_ratio=0.02）

### 3.4 设置归一化器

```python
self.model.set_normalizer(normalizer)
if cfg.training.use_ema:
    self.ema_model.set_normalizer(normalizer)
```

**归一化器作用**：
- 将观测和动作标准化到合适的范围
- 训练时：`normalized_data = (data - mean) / std`
- 推理时：`data = normalized_data * std + mean`

### 3.5 配置学习率调度器

```python
lr_scheduler = get_scheduler(
    cfg.training.lr_scheduler,           # "cosine"
    optimizer=self.optimizer,
    num_warmup_steps=cfg.training.lr_warmup_steps,  # 500
    num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,
    last_epoch=self.global_step-1
)
```

**学习率策略**：
- Warmup阶段：前500步线性增加
- Cosine衰减：之后余弦退火

### 3.6 初始化EMA模型管理器

```python
ema: EMAModel = None
if cfg.training.use_ema:
    ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
```

**EMA原理**：
- 维护模型参数的指数移动平均
- `θ_ema = β * θ_ema + (1-β) * θ`
- 用于推理，通常比训练模型更稳定

### 3.7 配置环境运行器

```python
env_runner: BaseImageRunner = hydra.utils.instantiate(
    cfg.task.env_runner,
    output_dir=self.output_dir
)
```

**环境运行器作用**：
- 在仿真环境中评估策略
- 生成rollout视频
- 计算成功率等指标

### 3.8 初始化WandB日志

```python
wandb_run = wandb.init(
    dir=str(self.output_dir),
    config=OmegaConf.to_container(cfg, resolve=True),
    **cfg.logging
)
```

**日志配置**：
```yaml
logging:
  mode: online
  project: diffusion_policy_debug
  name: 2023.01.16-20.20.06_train_diffusion_unet_hybrid_pusht_image
```

### 3.9 配置检查点管理器

```python
topk_manager = TopKCheckpointManager(
    save_dir=os.path.join(self.output_dir, 'checkpoints'),
    **cfg.checkpoint.topk
)
```

**TopK策略**：
- 保留性能最好的K个检查点（K=5）
- 监控指标：`test_mean_score`
- 文件名格式：`epoch=0250-test_mean_score=0.956.ckpt`

### 3.10 模型迁移到GPU

```python
device = torch.device(cfg.training.device)  # "cuda:0"
self.model.to(device)
if self.ema_model is not None:
    self.ema_model.to(device)
optimizer_to(self.optimizer, device)
```

**此时**：
- 你会看到PyTorch CUDA警告（RTX 5070兼容性问题）
- 但模型仍会尝试在GPU上运行

---

## 第四阶段：主训练循环 (run方法 - Part 2)

### 4.1 循环结构

```python
log_path = os.path.join(self.output_dir, 'logs.json.txt')
with JsonLogger(log_path) as json_logger:
    for local_epoch_idx in range(cfg.training.num_epochs):  # 3050个epoch
        # 每个epoch的训练和评估
```

### 4.2 单个Epoch训练流程

**4.2.1 训练阶段**
```python
train_losses = list()
with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}") as tepoch:
    for batch_idx, batch in enumerate(tepoch):
        # 步骤1: 数据迁移到GPU
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        
        # 步骤2: 前向传播计算损失
        raw_loss = self.model.compute_loss(batch)
        loss = raw_loss / cfg.training.gradient_accumulate_every
        
        # 步骤3: 反向传播
        loss.backward()
        
        # 步骤4: 优化器更新（梯度累积）
        if self.global_step % cfg.training.gradient_accumulate_every == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            lr_scheduler.step()
        
        # 步骤5: 更新EMA模型
        if cfg.training.use_ema:
            ema.step(self.model)
        
        # 步骤6: 记录日志
        raw_loss_cpu = raw_loss.item()
        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
        train_losses.append(raw_loss_cpu)
        
        step_log = {
            'train_loss': raw_loss_cpu,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'lr': lr_scheduler.get_last_lr()[0]
        }
        
        wandb_run.log(step_log, step=self.global_step)
        json_logger.log(step_log)
        self.global_step += 1
```

**compute_loss内部流程**：
```python
def compute_loss(self, batch):
    # 1. 提取观测和动作
    nobs = self.normalizer.normalize(batch['obs'])
    naction = self.normalizer['action'].normalize(batch['action'])
    
    # 2. 编码观测（图像 + 低维）
    obs_features = self.obs_encoder(nobs)
    
    # 3. 随机采样噪声时间步
    timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (B,))
    
    # 4. 给动作加噪声（前向扩散）
    noise = torch.randn_like(naction)
    noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)
    
    # 5. 预测噪声（反向扩散）
    noise_pred = self.model(
        noisy_actions,
        timesteps,
        global_cond=obs_features
    )
    
    # 6. 计算损失（预测噪声 vs 真实噪声）
    loss = F.mse_loss(noise_pred, noise)
    
    return loss
```

**4.2.2 评估阶段（每50个epoch）**

```python
# 1. 切换到评估模式
policy = self.ema_model if cfg.training.use_ema else self.model
policy.eval()

# 2. 运行环境Rollout
if (self.epoch % cfg.training.rollout_every) == 0:  # 每50个epoch
    runner_log = env_runner.run(policy)
    step_log.update(runner_log)
```

**Rollout流程**：
```python
def run(self, policy):
    # 对于每个测试episode
    for episode in range(n_test):
        obs = env.reset()
        for step in range(max_steps):
            # 使用策略预测动作
            action = policy.predict_action(obs)
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            
            if done:
                break
    
    # 计算指标
    success_rate = successes / n_test
    return {'test/mean_score': success_rate}
```

**4.2.3 验证损失计算**

```python
if (self.epoch % cfg.training.val_every) == 0:  # 每个epoch
    with torch.no_grad():
        val_losses = list()
        for batch in val_dataloader:
            batch = dict_apply(batch, lambda x: x.to(device))
            loss = self.model.compute_loss(batch)
            val_losses.append(loss)
        
        val_loss = torch.mean(torch.tensor(val_losses)).item()
        step_log['val_loss'] = val_loss
```

**4.2.4 扩散采样评估**

```python
if (self.epoch % cfg.training.sample_every) == 0:  # 每5个epoch
    with torch.no_grad():
        obs_dict = train_sampling_batch['obs']
        gt_action = train_sampling_batch['action']
        
        # 使用扩散模型预测动作
        result = policy.predict_action(obs_dict)
        pred_action = result['action_pred']
        
        # 计算MSE误差
        mse = F.mse_loss(pred_action, gt_action)
        step_log['train_action_mse_error'] = mse.item()
```

**4.2.5 保存检查点**

```python
if (self.epoch % cfg.training.checkpoint_every) == 0:  # 每50个epoch
    # 保存最新检查点
    if cfg.checkpoint.save_last_ckpt:
        self.save_checkpoint()  # 保存到 latest.ckpt
    
    # 保存TopK检查点
    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
    if topk_ckpt_path is not None:
        self.save_checkpoint(path=topk_ckpt_path)
```

---

## 第五阶段：推理/预测流程

### 5.1 predict_action方法

```python
def predict_action(self, obs_dict):
    # 1. 归一化观测
    nobs = self.normalizer.normalize(obs_dict)
    
    # 2. 编码观测
    obs_features = self.obs_encoder(nobs)
    
    # 3. 初始化随机噪声
    noisy_action = torch.randn((B, horizon, action_dim))
    
    # 4. 迭代去噪（DDPM采样）
    for t in reversed(range(num_inference_steps)):  # 100步
        # 预测噪声
        noise_pred = self.model(
            noisy_action,
            timestep=t,
            global_cond=obs_features
        )
        
        # 去噪一步
        noisy_action = self.noise_scheduler.step(
            noise_pred, t, noisy_action
        ).prev_sample
    
    # 5. 反归一化
    action = self.normalizer['action'].unnormalize(noisy_action)
    
    # 6. 返回动作序列
    return {'action_pred': action[:, :n_action_steps]}  # 只返回前8步
```

---

## 关键时间节点和性能分析

### 初始化阶段（约10-30秒）
1. Hydra配置解析：~1秒
2. 模型实例化：~5秒
3. 数据集加载：~5-10秒
4. WandB初始化：~3-5秒
5. 模型迁移到GPU：~3-5秒

### 单个Batch训练（约0.5-2秒）
- 数据迁移：~10ms
- 前向传播：~200-500ms
- 反向传播：~200-500ms
- 优化器更新：~50ms
- 日志记录：~10ms

### 单个Epoch（约2-5分钟）
- 训练141个batch：~2-4分钟
- 验证：~10-20秒
- （每50个epoch）Rollout：~1-3分钟
- （每50个epoch）保存检查点：~5-10秒

### 完整训练（约150-250小时）
- 3050个epoch × 3-5分钟 = 150-250小时（6-10天）
- 考虑rollout和评估：总时间可能更长

---

## 数据流图

```
输入：演示数据 (Zarr)
    ↓
[数据集加载]
    ├─ 序列采样 (滑动窗口)
    ├─ 填充处理 (pad_before=1, pad_after=7)
    └─ 归一化
    ↓
[Batch] obs: {image: [B,To,3,96,96], agent_pos: [B,To,2]}
        action: [B,T,Da]
    ↓
[视觉编码器]
    ├─ ResNet18处理图像
    └─ 拼接低维观测
    ↓
obs_features: [B, feature_dim]
    ↓
[扩散模型训练]
    ├─ 加噪声：action + noise → noisy_action
    ├─ U-Net预测噪声：noisy_action + obs_features → noise_pred
    └─ 计算损失：MSE(noise_pred, noise)
    ↓
[反向传播] → [优化器更新] → [EMA更新]
    ↓
[周期性评估]
    ├─ 验证损失
    ├─ 环境Rollout
    └─ 采样质量
    ↓
输出：训练好的策略模型
```

---

## 配置文件层次结构

```yaml
image_pusht_diffusion_policy_cnn.yaml (主配置)
├── _target_: TrainDiffusionUnetHybridWorkspace (工作空间类)
├── task: (引用 task/pusht_image.yaml)
│   ├── dataset: PushTImageDataset
│   ├── env_runner: PushTImageRunner
│   └── shape_meta: (观测和动作维度)
├── policy: (策略配置)
│   ├── _target_: DiffusionUnetHybridImagePolicy
│   ├── noise_scheduler: DDPMScheduler
│   └── network parameters
├── training: (训练参数)
│   ├── num_epochs: 3050
│   ├── device: cuda:0
│   ├── lr_scheduler: cosine
│   └── checkpointing策略
├── dataloader: (数据加载配置)
│   ├── batch_size: 64
│   └── num_workers: 8
├── optimizer: AdamW
├── ema: EMAModel配置
├── logging: WandB配置
└── checkpoint: 检查点策略
```

---

## 常见问题和关键点

### Q1: 为什么训练开始后看不到输出？
**A:** 有几个可能的原因：
1. **数据加载阶段**：首次加载数据集需要时间（10-30秒）
2. **模型初始化**：第一次前向传播会触发CUDA初始化（可能卡住）
3. **tqdm进度条设置**：`mininterval=1.0`表示至少1秒才更新一次
4. **缓冲问题**：虽然代码设置了行缓冲，但某些环境下仍可能有延迟

**解决方案**：
- 添加print语句在关键位置
- 使用`PYTHONUNBUFFERED=1`环境变量
- 查看`train.log`文件
- 检查GPU利用率（`nvidia-smi`）

### Q2: RTX 5070警告影响训练吗？
**A:** 
- **理论上**：sm_120不被支持，可能导致性能下降或错误
- **实际上**：PyTorch可能通过兼容层运行，但不保证稳定
- **建议**：升级到PyTorch 2.5+以获得完整支持

### Q3: 训练需要多长时间？
**A:** 
- Push-T任务：约6-10天（3050 epochs）
- 每个epoch：2-5分钟
- 可以减少epoch数进行快速实验

### Q4: 如何判断训练是否正常？
**检查点**：
1. ✓ 数据加载成功（看到batch数量）
2. ✓ 模型迁移到GPU
3. ✓ 第一个epoch开始
4. ✓ 损失下降
5. ✓ GPU利用率 > 50%
6. ✓ WandB有日志上传

---

## 文件输出结构

```
data/outputs/2025.12.03/15.00.11_train_diffusion_unet_hybrid_pusht_image/
├── checkpoints/
│   ├── latest.ckpt (最新检查点)
│   ├── epoch=0050-test_mean_score=0.234.ckpt
│   ├── epoch=0100-test_mean_score=0.456.ckpt
│   └── ... (TopK检查点)
├── wandb/ (WandB日志)
├── .hydra/ (Hydra配置)
│   ├── config.yaml (完整配置)
│   └── overrides.yaml (命令行覆盖)
├── train.log (训练日志)
├── logs.json.txt (JSON格式训练指标)
└── media/ (rollout视频)
```

---

## 调试技巧

### 1. 添加断点调试
```python
import pdb; pdb.set_trace()
```

### 2. 打印关键变量
```python
print(f"Epoch {self.epoch}, Step {self.global_step}, Loss {loss.item()}")
```

### 3. 使用debug模式
```yaml
training:
  debug: true  # 将训练缩短到2个epoch，3个batch
```

### 4. 监控GPU
```bash
watch -n 1 nvidia-smi
```

### 5. 查看WandB
访问：https://wandb.ai/your-username/diffusion_policy_debug

---

## 总结

整个训练流程可以概括为：

1. **入口**：`train.py` → Hydra加载配置
2. **初始化**：`TrainDiffusionUnetHybridWorkspace.__init__()` → 创建模型和优化器
3. **准备**：`run()` 方法前半部分 → 加载数据、配置组件
4. **训练**：`run()` 方法后半部分 → epoch循环
   - 每个batch：前向→反向→优化
   - 每个epoch：训练→验证→（rollout）→保存
5. **输出**：检查点、日志、rollout视频

核心是**扩散模型训练**：通过预测噪声来学习动作分布。


