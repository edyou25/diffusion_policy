# Diffusion Policy 代码仓库文件结构说明

本文档详细说明代码仓库中每个文件和目录的作用。

## 根目录文件

```sh
.
├── train.py                          # 训练入口脚本，使用Hydra进行配置管理
├── eval.py                           # 仿真环境评估脚本
├── eval_real_robot.py                # 真实机器人评估脚本
├── demo_pusht.py                     # Push-T任务演示脚本（仿真）
├── demo_real_robot.py                # 真实机器人演示数据收集脚本
├── ray_train_multirun.py             # 使用Ray进行多种子并行训练的脚本
├── ray_exec.py                       # Ray执行工具
├── multirun_metrics.py               # 多运行实验的指标聚合工具
├── setup.py                          # Python包安装配置
├── pyrightconfig.json                # Pyright类型检查配置
├── LICENSE                           # MIT许可证
├── README.md                         # 项目说明文档（英文）
├── image_pusht_diffusion_policy_cnn.yaml  # Push-T图像任务示例配置
├── conda_environment.yaml            # Conda环境配置（Linux仿真）
├── conda_environment_real.yaml       # Conda环境配置（真实机器人）
└── conda_environment_macos.yaml      # Conda环境配置（macOS开发）
```

## diffusion_policy/ 核心包目录

### codecs/ 编解码器
```sh
codecs/
└── imagecodecs_numcodecs.py          # 图像压缩编解码器（Jpeg2000），用于ReplayBuffer
```

### common/ 通用工具模块
```sh
common/
├── checkpoint_util.py                # 检查点保存/加载工具函数
├── cv2_util.py                       # OpenCV工具函数
├── env_util.py                       # 环境相关工具函数
├── json_logger.py                    # JSON格式日志记录器
├── nested_dict_util.py               # 嵌套字典操作工具
├── normalize_util.py                 # 归一化工具函数
├── pose_trajectory_interpolator.py   # 位姿轨迹插值器（用于真实机器人）
├── precise_sleep.py                  # 精确睡眠函数（用于实时控制）
├── pymunk_override.py                # PyMunk物理引擎覆盖
├── pymunk_util.py                    # PyMunk工具函数
├── pytorch_util.py                   # PyTorch工具函数
├── replay_buffer.py                  # 重放缓冲区（Zarr格式存储演示数据）
├── robomimic_config_util.py          # Robomimic配置工具
├── robomimic_util.py                 # Robomimic工具函数
├── sampler.py                        # 序列采样器（处理To和Ta的填充）
└── timestamp_accumulator.py          # 时间戳累加器（用于多传感器同步）
```

### config/ 配置文件目录
```sh
config/
├── task/                             # 任务配置文件目录
│   ├── blockpush_lowdim_seed.yaml   # 方块推任务（低维观测）
│   ├── blockpush_lowdim_seed_abs.yaml  # 方块推任务（低维观测，绝对动作）
│   ├── can_image.yaml                # Can任务（图像观测）
│   ├── can_image_abs.yaml            # Can任务（图像观测，绝对动作）
│   ├── can_lowdim.yaml               # Can任务（低维观测）
│   ├── can_lowdim_abs.yaml           # Can任务（低维观测，绝对动作）
│   ├── kitchen_lowdim.yaml           # Kitchen任务（低维观测）
│   ├── kitchen_lowdim_abs.yaml       # Kitchen任务（低维观测，绝对动作）
│   ├── lift_image.yaml               # Lift任务（图像观测）
│   ├── lift_image_abs.yaml           # Lift任务（图像观测，绝对动作）
│   ├── lift_lowdim.yaml              # Lift任务（低维观测）
│   ├── lift_lowdim_abs.yaml          # Lift任务（低维观测，绝对动作）
│   ├── pusht_image.yaml              # Push-T任务（图像观测）
│   ├── pusht_lowdim.yaml             # Push-T任务（低维观测）
│   ├── real_pusht_image.yaml         # 真实机器人Push-T任务（图像观测）
│   ├── square_image.yaml              # Square任务（图像观测）
│   ├── square_image_abs.yaml         # Square任务（图像观测，绝对动作）
│   ├── square_lowdim.yaml            # Square任务（低维观测）
│   ├── square_lowdim_abs.yaml        # Square任务（低维观测，绝对动作）
│   ├── tool_hang_image.yaml          # Tool Hang任务（图像观测）
│   ├── tool_hang_image_abs.yaml      # Tool Hang任务（图像观测，绝对动作）
│   ├── tool_hang_lowdim.yaml         # Tool Hang任务（低维观测）
│   ├── tool_hang_lowdim_abs.yaml     # Tool Hang任务（低维观测，绝对动作）
│   ├── transport_image.yaml           # Transport任务（图像观测）
│   ├── transport_image_abs.yaml      # Transport任务（图像观测，绝对动作）
│   ├── transport_lowdim.yaml         # Transport任务（低维观测）
│   └── transport_lowdim_abs.yaml     # Transport任务（低维观测，绝对动作）
│
└── train_*_workspace.yaml            # 各种训练工作空间配置文件
    ├── train_bet_lowdim_workspace.yaml              # BET方法训练配置（低维）
    ├── train_diffusion_transformer_hybrid_workspace.yaml      # Transformer扩散策略（混合）
    ├── train_diffusion_transformer_lowdim_kitchen_workspace.yaml  # Transformer扩散策略（Kitchen任务）
    ├── train_diffusion_transformer_lowdim_pusht_workspace.yaml    # Transformer扩散策略（Push-T任务）
    ├── train_diffusion_transformer_lowdim_workspace.yaml          # Transformer扩散策略（低维）
    ├── train_diffusion_transformer_real_hybrid_workspace.yaml     # Transformer扩散策略（真实机器人）
    ├── train_diffusion_unet_ddim_hybrid_workspace.yaml           # U-Net扩散策略（DDIM采样，混合）
    ├── train_diffusion_unet_ddim_lowdim_workspace.yaml            # U-Net扩散策略（DDIM采样，低维）
    ├── train_diffusion_unet_hybrid_workspace.yaml                 # U-Net扩散策略（混合）
    ├── train_diffusion_unet_image_pretrained_workspace.yaml       # U-Net扩散策略（图像，预训练）
    ├── train_diffusion_unet_image_workspace.yaml                  # U-Net扩散策略（图像）
    ├── train_diffusion_unet_lowdim_workspace.yaml                 # U-Net扩散策略（低维）
    ├── train_diffusion_unet_real_hybrid_workspace.yaml            # U-Net扩散策略（真实机器人，混合）
    ├── train_diffusion_unet_real_image_workspace.yaml             # U-Net扩散策略（真实机器人，图像）
    ├── train_diffusion_unet_real_pretrained_workspace.yaml        # U-Net扩散策略（真实机器人，预训练）
    ├── train_diffusion_unet_video_workspace.yaml                  # U-Net扩散策略（视频）
    ├── train_ibc_dfo_hybrid_workspace.yaml                        # IBC方法训练配置（混合）
    ├── train_ibc_dfo_lowdim_workspace.yaml                        # IBC方法训练配置（低维）
    ├── train_ibc_dfo_real_hybrid_workspace.yaml                   # IBC方法训练配置（真实机器人）
    ├── train_robomimic_image_workspace.yaml                       # Robomimic方法训练配置（图像）
    ├── train_robomimic_lowdim_workspace.yaml                      # Robomimic方法训练配置（低维）
    └── train_robomimic_real_image_workspace.yaml                  # Robomimic方法训练配置（真实机器人）
```

### dataset/ 数据集适配器
```sh
dataset/
├── base_dataset.py                    # 数据集基类（定义统一接口）
├── blockpush_lowdim_dataset.py        # 方块推任务数据集（低维观测）
├── kitchen_lowdim_dataset.py          # Kitchen任务数据集（低维观测）
├── kitchen_mjl_lowdim_dataset.py      # Kitchen任务数据集（MJL格式，低维观测）
├── mujoco_image_dataset.py            # MuJoCo图像数据集
├── pusht_dataset.py                   # Push-T任务数据集（低维观测）
├── pusht_image_dataset.py             # Push-T任务数据集（图像观测）
├── real_pusht_image_dataset.py        # 真实机器人Push-T任务数据集（图像观测）
├── robomimic_replay_image_dataset.py  # Robomimic重放数据集（图像观测）
└── robomimic_replay_lowdim_dataset.py # Robomimic重放数据集（低维观测）
```

### env/ 环境实现
```sh
env/
├── block_pushing/                     # 方块推任务环境
│   ├── assets/                        # 资源文件（URDF模型、OBJ网格等）
│   ├── block_pushing.py               # 方块推环境主文件
│   ├── block_pushing_discontinuous.py # 不连续方块推环境
│   ├── block_pushing_multimodal.py    # 多模态方块推环境
│   ├── oracles/                       # Oracle策略（专家演示）
│   │   ├── discontinuous_push_oracle.py
│   │   ├── multimodal_push_oracle.py
│   │   ├── oriented_push_oracle.py
│   │   ├── pushing_info.py
│   │   └── reach_oracle.py
│   └── utils/                         # 工具函数
│       ├── pose3d.py                  # 3D位姿工具
│       ├── utils_pybullet.py          # PyBullet工具
│       └── xarm_sim_robot.py          # XArm仿真机器人
│
├── kitchen/                           # Kitchen任务环境
│   ├── base.py                        # Kitchen环境基类
│   ├── kitchen_lowdim_wrapper.py     # Kitchen低维观测包装器
│   ├── kitchen_util.py                # Kitchen工具函数
│   ├── relay_policy_learning/         # 从Relay Policy Learning项目继承的代码
│   │   ├── adept_envs/                # Adept环境实现
│   │   ├── adept_models/              # Adept模型
│   │   └── third_party/                # 第三方代码（Franka机器人）
│   └── v0.py                          # Kitchen环境v0版本
│
├── pusht/                             # Push-T任务环境
│   ├── pusht_env.py                   # Push-T环境（低维观测）
│   ├── pusht_image_env.py             # Push-T环境（图像观测）
│   ├── pusht_keypoints_env.py         # Push-T环境（关键点观测）
│   ├── pymunk_keypoint_manager.py     # PyMunk关键点管理器
│   └── pymunk_override.py             # PyMunk覆盖
│
└── robomimic/                         # Robomimic任务包装器
    ├── robomimic_image_wrapper.py     # Robomimic图像观测包装器
    └── robomimic_lowdim_wrapper.py    # Robomimic低维观测包装器
```

### env_runner/ 环境运行器
```sh
env_runner/
├── base_image_runner.py               # 图像观测环境运行器基类
├── base_lowdim_runner.py              # 低维观测环境运行器基类
├── blockpush_lowdim_runner.py        # 方块推任务运行器（低维）
├── kitchen_lowdim_runner.py          # Kitchen任务运行器（低维）
├── pusht_image_runner.py             # Push-T任务运行器（图像）
├── pusht_keypoints_runner.py         # Push-T任务运行器（关键点）
├── real_pusht_image_runner.py        # 真实机器人Push-T任务运行器（图像）
├── robomimic_image_runner.py         # Robomimic任务运行器（图像）
└── robomimic_lowdim_runner.py        # Robomimic任务运行器（低维）
```

### gym_util/ Gym工具
```sh
gym_util/
├── async_vector_env.py                # 异步向量化环境（多进程，绕过GIL）
├── multistep_wrapper.py               # 多步包装器
├── sync_vector_env.py                 # 同步向量化环境
├── video_recording_wrapper.py         # 视频录制包装器
└── video_wrapper.py                   # 视频包装器
```

### model/ 模型组件
```sh
model/
├── bet/                               # BET（Behavior Transformer）方法
│   ├── action_ae/                     # 动作自编码器
│   │   ├── discretizers/             # 离散化器
│   │   │   └── k_means.py            # K-means离散化
│   │   └── __init__.py
│   ├── latent_generators/             # 潜在生成器
│   │   ├── latent_generator.py
│   │   ├── mingpt.py                 # MiniGPT实现
│   │   └── transformer.py            # Transformer实现
│   ├── libraries/                     # 库文件
│   │   ├── loss_fn.py                # 损失函数
│   │   └── mingpt/                   # MiniGPT库
│   │       ├── model.py
│   │       ├── trainer.py
│   │       └── utils.py
│   └── utils.py
│
├── common/                             # 通用模型组件
│   ├── dict_of_tensor_mixin.py        # 张量字典混入类
│   ├── lr_scheduler.py                # 学习率调度器
│   ├── module_attr_mixin.py           # 模块属性混入类
│   ├── normalizer.py                 # 线性归一化器（处理obs/action归一化）
│   ├── rotation_transformer.py        # 旋转变换器（处理四元数等）
│   ├── shape_util.py                  # 形状工具函数
│   └── tensor_util.py                 # 张量工具函数
│
├── diffusion/                         # 扩散模型核心组件
│   ├── conditional_unet1d.py          # 条件U-Net 1D（用于动作序列生成）
│   ├── conv1d_components.py           # 1D卷积组件
│   ├── ema_model.py                   # 指数移动平均模型
│   ├── mask_generator.py              # 掩码生成器（用于动作掩码）
│   ├── positional_embedding.py        # 位置编码
│   └── transformer_for_diffusion.py   # 用于扩散的Transformer
│
└── vision/                            # 视觉编码器
    ├── crop_randomizer.py             # 随机裁剪增强
    ├── model_getter.py                # 模型获取器（加载预训练视觉编码器）
    └── multi_image_obs_encoder.py    # 多图像观测编码器
```

### policy/ 策略实现
```sh
policy/
├── base_image_policy.py               # 图像观测策略基类
├── base_lowdim_policy.py              # 低维观测策略基类
├── bet_lowdim_policy.py               # BET策略（低维观测）
├── diffusion_transformer_hybrid_image_policy.py  # Transformer扩散策略（混合观测，图像）
├── diffusion_transformer_lowdim_policy.py        # Transformer扩散策略（低维观测）
├── diffusion_unet_hybrid_image_policy.py         # U-Net扩散策略（混合观测，图像）
├── diffusion_unet_image_policy.py                 # U-Net扩散策略（图像观测）
├── diffusion_unet_lowdim_policy.py                # U-Net扩散策略（低维观测）
├── diffusion_unet_video_policy.py                 # U-Net扩散策略（视频观测）
├── ibc_dfo_hybrid_image_policy.py                 # IBC策略（混合观测，图像）
├── ibc_dfo_lowdim_policy.py                       # IBC策略（低维观测）
├── robomimic_image_policy.py                      # Robomimic策略（图像观测）
└── robomimic_lowdim_policy.py                     # Robomimic策略（低维观测）
```

### real_world/ 真实机器人支持
```sh
real_world/
├── keystroke_counter.py               # 按键计数器（用于遥操作）
├── multi_camera_visualizer.py        # 多相机可视化工具
├── multi_realsense.py                 # 多RealSense相机管理（多进程）
├── real_data_conversion.py            # 真实数据转换工具
├── real_env.py                        # 真实环境接口（异步get_obs/exec_actions）
├── real_inference_util.py             # 真实推理工具
├── realsense_config/                  # RealSense相机配置
│   ├── 415_high_accuracy_mode.json   # D415高精度模式配置
│   └── 435_high_accuracy_mode.json   # D435高精度模式配置
├── rtde_interpolation_controller.py   # UR5机器人RTDE插值控制器
├── single_realsense.py                # 单RealSense相机（单进程）
├── spacemouse.py                      # SpaceMouse遥操作支持
├── spacemouse_shared_memory.py        # SpaceMouse共享内存接口
└── video_recorder.py                  # 视频录制器
```

### scripts/ 脚本工具
```sh
scripts/
├── bet_blockpush_conversion.py        # BET方块推数据转换
├── blockpush_abs_conversion.py       # 方块推绝对动作转换
├── episode_lengths.py                 # 计算episode长度统计
├── generate_bet_blockpush.py          # 生成BET方块推数据
├── real_dataset_conversion.py         # 真实数据集转换
├── real_pusht_metrics.py              # 真实Push-T任务指标计算
├── real_pusht_successrate.py          # 真实Push-T任务成功率计算
├── robomimic_dataset_action_comparison.py  # Robomimic数据集动作比较
└── robomimic_dataset_conversion.py    # Robomimic数据集转换
```

### shared_memory/ 共享内存数据结构
```sh
shared_memory/
├── shared_memory_queue.py             # 共享内存队列（FIFO）
├── shared_memory_ring_buffer.py      # 共享内存环形缓冲区（FILO，无锁）
├── shared_memory_util.py              # 共享内存工具函数
└── shared_ndarray.py                  # 共享numpy数组
```

### workspace/ 训练工作空间
```sh
workspace/
├── base_workspace.py                  # 工作空间基类（检查点管理、训练循环框架）
├── train_bet_lowdim_workspace.py     # BET方法训练工作空间（低维）
├── train_diffusion_transformer_hybrid_workspace.py      # Transformer扩散策略训练（混合）
├── train_diffusion_transformer_lowdim_workspace.py      # Transformer扩散策略训练（低维）
├── train_diffusion_unet_hybrid_workspace.py              # U-Net扩散策略训练（混合）
├── train_diffusion_unet_image_workspace.py              # U-Net扩散策略训练（图像）
├── train_diffusion_unet_lowdim_workspace.py             # U-Net扩散策略训练（低维）
├── train_diffusion_unet_video_workspace.py              # U-Net扩散策略训练（视频）
├── train_ibc_dfo_hybrid_workspace.py                    # IBC方法训练工作空间（混合）
├── train_ibc_dfo_lowdim_workspace.py                    # IBC方法训练工作空间（低维）
├── train_robomimic_image_workspace.py                   # Robomimic方法训练工作空间（图像）
└── train_robomimic_lowdim_workspace.py                 # Robomimic方法训练工作空间（低维）
```

## doc/ 文档目录
```sh
doc/
├── note.md                            # 技术笔记（Transformer到Diffusion Policy的演进）
├── paper-cn.md                        # 论文中文翻译
├── paper-cn.pdf                       # 论文中文PDF
├── README-cn.md                       # 项目说明文档（中文）
├── tree.md                            # 本文档：文件结构说明
└── Chi et al. - 2024 - Diffusion Policy Visuomotor Policy Learning via Action Diffusion.pdf  # 原始论文
```

## tests/ 测试目录
```sh
tests/
├── test_block_pushing.py              # 方块推任务测试
├── test_cv2_util.py                   # OpenCV工具测试
├── test_multi_realsense.py            # 多RealSense相机测试
├── test_pose_trajectory_interpolator.py  # 位姿轨迹插值器测试
├── test_precise_sleep.py              # 精确睡眠函数测试
├── test_replay_buffer.py              # 重放缓冲区测试
├── test_ring_buffer.py                # 环形缓冲区测试
├── test_robomimic_image_runner.py    # Robomimic图像运行器测试
├── test_robomimic_lowdim_runner.py   # Robomimic低维运行器测试
├── test_shared_queue.py               # 共享队列测试
├── test_single_realsense.py          # 单RealSense相机测试
└── test_timestamp_accumulator.py      # 时间戳累加器测试
```

## media/ 媒体资源
```sh
media/
├── teaser.png                         # 项目宣传图
└── multimodal_sim.png                 # 多模态仿真示意图
```

## 关键概念说明

### 观测类型
- **lowdim（低维观测）**：状态向量，如机器人关节角度、物体位置等
- **image（图像观测）**：RGB图像
- **hybrid（混合观测）**：同时包含低维和图像观测

### 动作类型
- **相对动作（relative）**：相对于当前状态的动作增量
- **绝对动作（abs）**：绝对位置/姿态

### 核心接口
- **Dataset**：返回 `obs` 和 `action`，提供 `get_normalizer()` 方法
- **Policy**：实现 `predict_action(obs_dict)` 方法，返回动作
- **EnvRunner**：实现 `run(policy)` 方法，执行策略并返回指标
- **Workspace**：管理训练和评估的完整生命周期

### 时间概念
- **To (n_obs_steps)**：观测视野，使用过去To步的观测
- **Ta (n_action_steps)**：动作视野，预测未来Ta步的动作
- **T (horizon)**：预测视野，总的预测时间步数

