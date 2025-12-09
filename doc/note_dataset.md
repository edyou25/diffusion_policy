=== Zarr 数据集结构 ===
/
 ├── data
 │   ├── action (25650, 2) float32
 │   ├── img (25650, 96, 96, 3) float32
 │   ├── keypoint (25650, 9, 2) float32
 │   ├── n_contacts (25650, 1) float32
 │   └── state (25650, 5) float32
 └── meta
     └── episode_ends (206,) int64

=== 数据详情 ===

[data 组]:

  action:
    形状: (25650, 2)
    数据类型: float32
    块大小: (161, 2)
    原始大小: 0.20 MB
    范围: [12.000, 511.000]
    均值: 261.114
    标准差: 104.181

  img:
    形状: (25650, 96, 96, 3)
    数据类型: float32
    块大小: (161, 96, 96, 3)
    原始大小: 2705.27 MB
    图像值范围: [65.000, 255.000]

  keypoint:
    形状: (25650, 9, 2)
    数据类型: float32
    块大小: (161, 9, 2)
    原始大小: 1.76 MB

  n_contacts:
    形状: (25650, 1)
    数据类型: float32
    块大小: (161, 1)
    原始大小: 0.10 MB

  state:
    形状: (25650, 5)
    数据类型: float32
    块大小: (161, 5)
    原始大小: 0.49 MB
    agent_pos (x,y) 范围: [13.5, 511.0]


[meta 组]:

  episode_ends:
    形状: (206,)
    数据类型: int64
    Episode 数量: 206
    Episode 长度: min=49, max=246, mean=124.5
    总时间步: 25650


=== 总结 ===
数据集总大小: 2707.82 MB
Episodes: 206
总时间步: 25650

数据包含:
  - img: 96x96 RGB 图像
  - state: (agent_x, agent_y, block_x, block_y, block_theta)
  - action: (delta_x, delta_y) 控制指令
  - keypoint: 9个关键点坐标 (agent + block corners)
  - n_contacts: 接触数量