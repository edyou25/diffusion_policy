# Diffusion Policy（扩散策略）

[[项目主页]](https://diffusion-policy.cs.columbia.edu/)
[[论文]](https://diffusion-policy.cs.columbia.edu/#paper)
[[数据]](https://diffusion-policy.cs.columbia.edu/data/)
[[Colab（状态型环境）]](https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B?usp=sharing)
[[Colab（视觉型环境）]](https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing)

[程驰](http://cheng-chi.github.io/)<sup>1</sup>、
[冯思源](https://www.cs.cmu.edu/~sfeng/)<sup>2</sup>、
[杜逸伦](https://yilundu.github.io/)<sup>3</sup>、
[徐振家](https://www.zhenjiaxu.com/)<sup>1</sup>、
[Eric Cousineau](https://www.eacousineau.com/)<sup>2</sup>、
[Benjamin Burchfiel](http://www.benburchfiel.com/)<sup>2</sup>、
[宋舒然](https://www.cs.columbia.edu/~shurans/)<sup>1</sup>

<sup>1</sup>哥伦比亚大学、
<sup>2</sup>丰田研究院、
<sup>3</sup>麻省理工学院

<img src="media/teaser.png" alt="示意图" width="100%"/>
<img src="media/multimodal_sim.png" alt="示意图" width="100%"/>

## 🛝 快速体验！
我们提供独立的Google Colab笔记本，是体验Diffusion Policy最便捷的方式。分别针对[状态型环境](https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B?usp=sharing)和[视觉型环境](https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing)提供专用笔记本。

## 🧾 查看实验日志！
对于论文[[论文]](https://diffusion-policy.cs.columbia.edu/#paper)中表I、II和IV的每一组实验，我们提供以下资源：
1. `config.yaml`：包含复现实验所需的全部参数；
2. 详细的训练/评估日志 `logs.json.txt`（记录每一步训练过程）；
3. 最优模型检查点 `epoch=*-test_mean_score=*.ckpt` 和每轮训练的最终检查点 `latest.ckpt`。

实验日志托管在我们的网站上，目录格式如下：
`https://diffusion-policy.cs.columbia.edu/data/experiments/<image|low_dim>/<task>/<method>/`

每个实验目录的结构如下：
```
.
├── config.yaml
├── metrics
│   └── logs.json.txt
├── train_0
│   ├── checkpoints
│   │   ├── epoch=0300-test_mean_score=1.000.ckpt
│   │   └── latest.ckpt
│   └── logs.json.txt
├── train_1
│   ├── checkpoints
│   │   ├── epoch=0250-test_mean_score=1.000.ckpt
│   │   └── latest.ckpt
│   └── logs.json.txt
└── train_2
    ├── checkpoints
    │   ├── epoch=0250-test_mean_score=1.000.ckpt
    │   └── latest.ckpt
    └── logs.json.txt
```
`metrics/logs.json.txt` 文件通过 `multirun_metrics.py` 聚合了3次训练运行中每50个epoch的评估指标。论文中报告的数值对应 `max` 和 `k_min_train_loss` 聚合键。

如需下载某个子目录下的所有文件，使用以下命令：
```console
$ wget --recursive --no-parent --no-host-directories --relative --reject="index.html*" https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/square_ph/diffusion_policy_cnn/
```

## 🛠️ 安装指南
### 🖥️ 仿真环境
如需复现仿真基准测试结果，请在配备NVIDIA GPU的Linux机器上安装conda环境。在Ubuntu 20.04系统中，需先安装以下mujoco依赖的apt包：
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

我们推荐使用[Mambaforge](https://github.com/conda-forge/miniforge#mambaforge)（而非标准Anaconda）以加快安装速度：
```console
$ mamba env create -f conda_environment.yaml
```

也可使用标准conda安装：
```console
$ conda env create -f conda_environment.yaml
```

`conda_environment_macos.yaml` 仅适用于MacOS开发环境，不支持完整的基准测试功能。

### 🦾 真实机器人环境
硬件配置（适用于Push-T任务）：
* 1台 [UR5-CB3](https://www.universal-robots.com/cb3) 或 [UR5e](https://www.universal-robots.com/products/ur5-robot/)（需支持[RTDE接口](https://www.universal-robots.com/articles/ur/interface-communication/real-time-data-exchange-rtde-guide/)）
* 2台 [RealSense D415](https://www.intelrealsense.com/depth-camera-d415/) 深度相机
* 1台 [3Dconnexion SpaceMouse](https://3dconnexion.com/us/product/spacemouse-wireless/)（用于遥操作）
* 1台 [Millibar Robotics 手动工具更换器](https://www.millibar.com/manual-tool-changer/)（仅需机器人端部件）
* 1个3D打印[末端执行器](https://cad.onshape.com/documents/a818888644a15afa6cc68ee5/w/2885b48b018cda84f425beca/e/3e8771c2124cee024edd2fed?renderMode=0&uiState=63ffcba6631ca919895e64e5)
* 1个3D打印[T型块](https://cad.onshape.com/documents/f1140134e38f6ed6902648d5/w/a78cf81827600e4ff4058d03/e/f35f57fb7589f72e05c76caf?renderMode=0&uiState=63ffcbc9af4a881b344898ee)
* RealSense相机专用USB-C线缆及固定螺丝

软件配置：
* Ubuntu 20.04.3（已测试兼容）
* Mujoco依赖：
`sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`
* [RealSense SDK](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)
* SpaceMouse依赖：
`sudo apt install libspnav-dev spacenavd; sudo systemctl start spacenavd`
* Conda环境：`mamba env create -f conda_environment_real.yaml`

## 🖥️ 复现仿真基准测试结果
### 下载训练数据
在仓库根目录下创建data子目录：
```console
[diffusion_policy]$ mkdir data && cd data
```

从[https://diffusion-policy.cs.columbia.edu/data/training/](https://diffusion-policy.cs.columbia.edu/data/training/) 下载对应的数据压缩包：
```console
[data]$ wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
```

解压训练数据：
```console
[data]$ unzip pusht.zip && rm -f pusht.zip && cd ..
```

下载对应实验的配置文件：
```console
[diffusion_policy]$ wget -O image_pusht_diffusion_policy_cnn.yaml https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/config.yaml
```

### 单种子训练
激活conda环境并登录[wandb](https://wandb.ai)（若未登录）：
```console
[diffusion_policy]$ conda activate robodiff
(robodiff)[diffusion_policy]$ wandb login
```

在GPU 0上以种子42启动训练：
```console
(robodiff)[diffusion_policy]$ python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

训练过程会创建如下格式的目录：`data/outputs/yyyy.mm.dd/hh.mm.ss_<方法名>_<任务名>`，用于存储配置文件、日志和检查点。模型每50个epoch会进行一次评估，成功率将以 `test/mean_score` 为名记录在wandb中，同时生成部分滚动轨迹的视频：
```console
(robodiff)[diffusion_policy]$ tree data/outputs/2023.03.01/20.02.03_train_diffusion_unet_hybrid_pusht_image -I wandb
data/outputs/2023.03.01/20.02.03_train_diffusion_unet_hybrid_pusht_image
├── checkpoints
│   ├── epoch=0000-test_mean_score=0.134.ckpt
│   └── latest.ckpt
├── .hydra
│   ├── config.yaml
│   ├── hydra.yaml
│   └── overrides.yaml
├── logs.json.txt
├── media
│   ├── 2k5u6wli.mp4
│   ├── 2kvovxms.mp4
│   ├── 2pxd9f6b.mp4
│   ├── 2q5gjt5f.mp4
│   ├── 2sawbf6m.mp4
│   └── 538ubl79.mp4
└── train.log

3 directories, 13 files
```

### 多种子训练
启动本地ray集群（大规模实验建议配置[带自动扩缩容的AWS集群](https://docs.ray.io/en/master/cluster/vms/user-guides/launching-clusters/aws.html)，其他命令保持不变）：
```console
(robodiff)[diffusion_policy]$ export CUDA_VISIBLE_DEVICES=0,1,2  # 指定ray集群管理的GPU
(robodiff)[diffusion_policy]$ ray start --head --num-gpus=3
```

启动ray客户端，将创建3个训练worker（对应3个种子）和1个指标监控worker：
```console
(robodiff)[diffusion_policy]$ python ray_train_multirun.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml --seeds=42,43,44 --monitor_key=test/mean_score -- multi_run.run_dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' multi_run.wandb_name_base='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}'
```

除每个训练worker独立记录的wandb日志外，指标监控worker会将3次训练的聚合指标记录到wandb项目 `diffusion_policy_metrics` 中。本地的配置文件、日志和检查点将存储在 `data/outputs/yyyy.mm.dd/hh.mm.ss_<方法名>_<任务名>` 目录下，结构与我们的[训练日志](https://diffusion-policy.cs.columbia.edu/data/experiments/)一致：
```console
(robodiff)[diffusion_policy]$ tree data/outputs/2023.03.01/22.13.58_train_diffusion_unet_hybrid_pusht_image -I 'wandb|media'
data/outputs/2023.03.01/22.13.58_train_diffusion_unet_hybrid_pusht_image
├── config.yaml
├── metrics
│   ├── logs.json.txt
│   ├── metrics.json
│   └── metrics.log
├── train_0
│   ├── checkpoints
│   │   ├── epoch=0000-test_mean_score=0.174.ckpt
│   │   └── latest.ckpt
│   ├── logs.json.txt
│   └── train.log
├── train_1
│   ├── checkpoints
│   │   ├── epoch=0000-test_mean_score=0.131.ckpt
│   │   └── latest.ckpt
│   ├── logs.json.txt
│   └── train.log
└── train_2
    ├── checkpoints
    │   ├── epoch=0000-test_mean_score=0.105.ckpt
    │   └── latest.ckpt
    ├── logs.json.txt
    └── train.log

7 directories, 16 files
```

### 🆕 评估预训练检查点
从已发布的训练日志文件夹中下载检查点，例如：[https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt](https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt)

运行评估脚本：
```console
(robodiff)[diffusion_policy]$ python eval.py --checkpoint data/0550-test_mean_score=0.969.ckpt --output_dir data/pusht_eval_output --device cuda:0
```

评估完成后将生成如下目录结构：
```console
(robodiff)[diffusion_policy]$ tree data/pusht_eval_output
data/pusht_eval_output
├── eval_log.json
└── media
    ├── 1fxtno84.mp4
    ├── 224l7jqd.mp4
    ├── 2fo4btlf.mp4
    ├── 2in4cn7a.mp4
    ├── 34b3o2qq.mp4
    └── 3p7jqn32.mp4

1 directory, 7 files
```

`eval_log.json` 包含训练过程中记录到wandb的各项指标：
```console
(robodiff)[diffusion_policy]$ cat data/pusht_eval_output/eval_log.json
{
  "test/mean_score": 0.9150393806777066,
  "test/sim_max_reward_4300000": 1.0,
  "test/sim_max_reward_4300001": 0.9872969750774386,
...
  "train/sim_video_1": "data/pusht_eval_output//media/2fo4btlf.mp4"
}
```

## 🦾 真实机器人的演示、训练与评估
确保UR5机器人已启动并可通过网络接口接收指令（全程确保急停按钮可随时触发）、RealSense相机已连接到工作站（可通过 `realsense-viewer` 测试）、SpaceMouse已连接且 `spacenavd` 守护进程正在运行（可通过 `systemctl status spacenavd` 验证）。

启动演示数据采集脚本：按“C”开始录制，通过SpaceMouse控制机器人运动，按“S”停止录制：
```console
(robodiff)[diffusion_policy]$ python demo_real_robot.py -o data/demo_pusht_real --robot_ip 192.168.0.204
```

采集完成后，`data/demo_pusht_real` 目录将生成演示数据集，结构与示例[真实Push-T训练数据集](https://diffusion-policy.cs.columbia.edu/data/training/pusht_real.zip)一致。

如需训练Diffusion策略，使用以下命令启动训练：
```console
(robodiff)[diffusion_policy]$ python train.py --config-name=train_diffusion_unet_real_image_workspace task.dataset_path=data/demo_pusht_real
```
若相机配置不同，可修改 [`diffusion_policy/config/task/real_pusht_image.yaml`](./diffusion_policy/config/task/real_pusht_image.yaml) 文件。

假设训练已完成，且检查点路径为 `data/outputs/blah/checkpoints/latest.ckpt`，启动评估脚本：
```console
python eval_real_robot.py -i data/outputs/blah/checkpoints/latest.ckpt -o data/eval_pusht_real --robot_ip 192.168.0.204
```
按“C”开始评估（控制权移交至策略），按“S”停止当前轮次评估。

## 🗺️ 代码库教程
本代码库的设计遵循以下原则：
1. 实现N个任务和M种方法仅需O(N+M)量级的代码，而非O(N×M)；
2. 同时保留最大灵活性。

为实现上述目标，我们：
1. 维护了任务与方法之间简单统一的接口；
2. 使任务和方法的实现相互独立。

这些设计决策可能导致任务与方法之间存在少量代码重复，但我们认为，这种代价远小于“添加/修改任务/方法时不影响其他模块”以及“通过线性阅读代码即可理解任务/方法”带来的收益 😊。

### 模块拆分
#### 任务侧（Task Side）包含：
* `Dataset`（数据集）：将第三方数据集适配到统一接口；
* `EnvRunner`（环境运行器）：执行符合接口规范的`Policy`（策略），并生成日志和指标；
* `config/task/<task_name>.yaml`：包含构建`Dataset`和`EnvRunner`所需的全部配置；
* （可选）`Env`（环境）：兼容`gym==0.21.0`的环境类，封装任务场景。

#### 策略侧（Policy Side）包含：
* `Policy`（策略）：实现符合接口规范的推理逻辑及部分训练流程；
* `Workspace`（工作空间）：管理方法的训练与评估（交替进行）生命周期；
* `config/<workspace_name>.yaml`：包含构建`Policy`和`Workspace`所需的全部配置。

### 接口规范
#### 低维数据接口（Low Dim）
[`LowdimPolicy`](./diffusion_policy/policy/base_lowdim_policy.py) 接收观测字典：
- `"obs":` 张量，形状为 `(B, To, Do)`（批次大小×观测视界×观测维度）

输出动作字典：
- `"action": ` 张量，形状为 `(B, Ta, Da)`（批次大小×动作视界×动作维度）

[`LowdimDataset`](./diffusion_policy/dataset/base_dataset.py) 返回样本字典：
- `"obs":` 张量，形状为 `(To, Do)`（观测视界×观测维度）
- `"action":` 张量，形状为 `(Ta, Da)`（动作视界×动作维度）

其 `get_normalizer` 方法返回包含 `"obs"` 和 `"action"` 键的 [`LinearNormalizer`](./diffusion_policy/model/common/normalizer.py)（线性归一化器）。

策略通过内置的`LinearNormalizer`在GPU上处理归一化，归一化器的参数会作为策略权重检查点的一部分保存。

#### 图像数据接口（Image）
[`ImagePolicy`](./diffusion_policy/policy/base_image_policy.py) 接收观测字典：
- `"key0":` 张量，形状为 `(B, To, *)`（批次大小×观测视界×其他维度）
- `"key1":` 张量，例如形状为 `(B, To, H, W, 3)`（批次大小×观测视界×高度×宽度×通道数），数据类型为float32，取值范围[0,1]

输出动作字典：
- `"action": ` 张量，形状为 `(B, Ta, Da)`（批次大小×动作视界×动作维度）

[`ImageDataset`](./diffusion_policy/dataset/base_dataset.py) 返回样本字典：
- `"obs":` 字典，包含：
    - `"key0":` 张量，形状为 `(To, *)`（观测视界×其他维度）
    - `"key1":` 张量，形状为 `(To, H, W, 3)`（观测视界×高度×宽度×通道数）
- `"action":` 张量，形状为 `(Ta, Da)`（动作视界×动作维度）

其 `get_normalizer` 方法返回包含 `"key0"`、`"key1"` 和 `"action"` 键的 [`LinearNormalizer`](./diffusion_policy/model/common/normalizer.py)（线性归一化器）。

#### 示例说明
```
To = 3（观测视界）
Ta = 4（动作视界）
T = 6（预测视界）
|o|o|o|
| | |a|a|a|a|
|o|o|
| |a|a|a|a|a|
| | | | |a|a|
```
论文术语与代码库变量对应关系：
- 观测视界（Observation Horizon）：`To` / `n_obs_steps`
- 动作视界（Action Horizon）：`Ta` / `n_action_steps`
- 预测视界（Prediction Horizon）：`T` / `horizon`

经典的单步观测/动作形式（如MDP）是该接口的特例，此时 `To=1` 且 `Ta=1`。

## 🔩 核心组件
### `Workspace`（工作空间）
`Workspace` 对象封装了运行实验所需的全部状态和代码：
* 继承自 [`BaseWorkspace`](./diffusion_policy/workspace/base_workspace.py)；
* 由`hydra`生成的单个`OmegaConf`配置对象应包含构建工作空间和运行实验的全部信息（对应 `config/<workspace_name>.yaml` + hydra覆盖参数）；
* `run` 方法包含实验的完整流程；
* 检查点以`Workspace`为单位保存，所有作为对象属性实现的训练状态会被`save_checkpoint`方法自动保存；
* 实验的其他状态应实现为`run`方法的局部变量。

训练的入口文件是 `train.py`，该文件使用`@hydra.main`装饰器。有关命令行参数和配置覆盖的详细说明，请参考[hydra官方文档](https://hydra.cc/)。例如，参数 `task=<task_name>` 会将配置中的`task`子树替换为 `config/task/<task_name>.yaml` 的内容，从而指定当前实验要运行的任务。

### `Dataset`（数据集）
`Dataset` 对象：
* 继承自 `torch.utils.data.Dataset`；
* 根据任务类型（低维/图像观测），返回符合[接口规范](#接口规范)的样本；
* 包含 `get_normalizer` 方法，返回符合[接口规范](#接口规范)的 `LinearNormalizer`（线性归一化器）。

归一化是项目开发中常见的bug来源，有时需要打印`LinearNormalizer`中每个键对应的`scale`（缩放系数）和`bias`（偏移量）向量以排查问题。

我们的大多数`Dataset`实现结合了[`ReplayBuffer`](#replaybuffer)（回放缓冲区）和 [`SequenceSampler`](./diffusion_policy/common/sampler.py)（序列采样器）来生成样本。根据`To`和`Ta`正确处理每个演示片段开头和结尾的填充，对保证模型性能至关重要。实现自定义采样方法前，请先阅读 [`SequenceSampler`](./diffusion_policy/common/sampler.py) 的代码。

### `Policy`（策略）
`Policy` 对象：
* 继承自 `BaseLowdimPolicy` 或 `BaseImagePolicy`；
* 包含 `predict_action` 方法，接收观测字典并输出符合[接口规范](#接口规范)的动作；
* 包含 `set_normalizer` 方法，接收`LinearNormalizer`并在策略内部处理观测/动作的归一化；
* （可选）包含 `compute_loss` 方法，接收批次数据并返回待优化的损失值；
* （可选）由于不同方法的训练和评估流程存在差异，通常每个`Policy`类对应一个`Workspace`类。

### `EnvRunner`（环境运行器）
`EnvRunner` 对象抽象了不同任务环境之间的细微差异：
* 包含 `run` 方法，接收用于评估的`Policy`对象，返回日志和指标字典（所有值需兼容`wandb.log`）。

为最大化评估速度，我们通常使用修改后的 [`gym.vector.AsyncVectorEnv`](./diffusion_policy/gym_util/async_vector_env.py) 对环境进行向量化——每个环境在独立进程中运行（规避Python GIL限制）。

⚠️ 在Linux系统中，子进程通过`fork`创建，对于初始化时会创建OpenGL上下文的环境（如robosuite），需特别注意：OpenGL上下文会被继承到子进程的内存空间，常导致段错误等难以排查的bug。解决方案是提供`dummy_env_fn`，用于创建不初始化OpenGL的环境。

### `ReplayBuffer`（回放缓冲区）
[`ReplayBuffer`](./diffusion_policy/common/replay_buffer.py) 是核心数据结构，支持内存和磁盘存储演示数据集（含分块和压缩功能）。该结构大量使用 [`zarr`](https://zarr.readthedocs.io/en/stable/index.html) 格式，同时提供`numpy`后端以降低访问开销。

磁盘存储形式支持嵌套目录（如 `data/pusht_cchi_v7_replay.zarr`）或压缩包（如 `data/robomimic/datasets/square/mh/image_abs.hdf5.zarr.zip`）。

由于我们的数据集规模相对较小，通常可将整个图像数据集通过[`Jpeg2000压缩`](./diffusion_policy/codecs/imagecodecs_numcodecs.py)加载到内存中，以消除训练过程中的磁盘IO开销（代价是增加CPU计算量）。

示例目录结构：
```
data/pusht_cchi_v7_replay.zarr
 ├── data
 │   ├── action (25650, 2) float32  # 动作数据（时间步×动作维度）
 │   ├── img (25650, 96, 96, 3) float32  # 图像数据（时间步×高度×宽度×通道数）
 │   ├── keypoint (25650, 9, 2) float32  # 关键点数据（时间步×关键点数量×坐标维度）
 │   ├── n_contacts (25650, 1) float32  # 接触次数数据（时间步×1）
 │   └── state (25650, 5) float32  # 状态数据（时间步×状态维度）
 └── meta
     └── episode_ends (206,) int64  # 每个演示片段的结束时间步索引
```

`data` 目录下的每个数组存储所有演示片段的某一类数据（按时间维度拼接）；`meta/episode_ends` 数组存储每个演示片段在时间维度上的结束索引。

### `SharedMemoryRingBuffer`（共享内存环形缓冲区）
[`SharedMemoryRingBuffer`](./diffusion_policy/shared_memory/shared_memory_ring_buffer.py) 是无锁的先进后出（FILO）数据结构，在[真实机器人实现](./diffusion_policy/real_world)中广泛使用——可充分利用多CPU核心，同时避免`multiprocessing.Queue`的pickle序列化和锁开销。

示例场景：从5台RealSense相机获取最近`To`帧图像。通过 [`SingleRealsense`](./diffusion_policy/real_world/single_realsense.py) 为每台相机启动独立的SDK/流水线进程，每个进程持续将采集到的图像写入与主进程共享的`SharedMemoryRingBuffer`。主进程可利用环形缓冲区的FILO特性，快速获取最新的`To`帧图像。

我们还实现了先进先出（FIFO）的 [`SharedMemoryQueue`](./diffusion_policy/shared_memory/shared_memory_queue.py)，用于 [`RTDEInterpolationController`](./diffusion_policy/real_world/rtde_interpolation_controller.py)。

### `RealEnv`（真实环境）
与[OpenAI Gym](https://gymnasium.farama.org/)不同，我们的策略与环境采用异步交互方式。在 [`RealEnv`](./diffusion_policy/real_world/real_env.py) 中，`gym` 的 `step` 方法被拆分为两个独立方法：`get_obs` 和 `exec_actions`。

- `get_obs` 方法：从`SharedMemoryRingBuffer`获取最新观测及对应的时间戳，可在评估轮次中随时调用；
- `exec_actions` 方法：接收动作序列及每个动作的预期执行时间戳，调用后仅将动作入队到`RTDEInterpolationController`，无需等待执行完成即可返回。

## 🩹 添加新任务
参考以下文件的实现方式：
* `diffusion_policy/dataset/pusht_image_dataset.py`
* `diffusion_policy/env_runner/pusht_image_runner.py`
* `diffusion_policy/config/task/pusht_image.yaml`

确保 `shape_meta` 与任务的输入/输出形状一致；确保 `env_runner._target_` 和 `dataset._target_` 指向新增的类。训练时，在 `train.py` 的参数中添加 `task=<你的任务名>` 即可。

## 🩹 添加新方法
参考以下文件的实现方式：
* `diffusion_policy/workspace/train_diffusion_unet_image_workspace.py`
* `diffusion_policy/policy/diffusion_unet_image_policy.py`
* `diffusion_policy/config/train_diffusion_unet_image_workspace.yaml`

确保工作空间配置文件的 `_target_` 指向新增的工作空间类。

## 🏷️ 许可证
本仓库基于MIT许可证开源，详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢
* 我们的 [`ConditionalUnet1D`](./diffusion_policy/model/diffusion/conditional_unet1d.py) 实现改编自 [Planning with Diffusion](https://github.com/jannerm/diffuser)；
* 我们的 [`TransformerForDiffusion`](./diffusion_policy/model/diffusion/transformer_for_diffusion.py) 实现改编自 [MinGPT](https://github.com/karpathy/minGPT)；
* BET基线模型 [`BET`](./diffusion_policy/model/bet) 改编自其[官方仓库](https://github.com/notmahi/bet)；
* IBC基线模型 [`IBC`](./diffusion_policy/policy/ibc_dfo_lowdim_policy.py) 改编自 [Kevin Zakka的复现版本](https://github.com/kevinzakka/ibc)；
* 本项目大量使用了 [Robomimic](https://github.com/ARISE-Initiative/robomimic) 的任务和 [`ObservationEncoder`](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/models/obs_nets.py)（观测编码器）；
* Push-T任务 [`Push-T`](./diffusion_policy/env/pusht) 改编自 [IBC](https://github.com/google-research/ibc)；
* 方块推送任务 [`Block Pushing`](./diffusion_policy/env/block_pushing) 改编自 [BET](https://github.com/notmahi/bet) 和 [IBC](https://github.com/google-research/ibc)；
* 厨房任务 [`Kitchen`](./diffusion_policy/env/kitchen) 改编自 [BET](https://github.com/notmahi/bet) 和 [Relay Policy Learning](https://github.com/google-research/relay-policy-learning)；
* 共享内存数据结构 [`shared_memory`](./diffusion_policy/shared_memory) 深受 [shared-ndarray2](https://gitlab.com/osu-nrsg/shared-ndarray2) 启发。