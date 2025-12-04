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