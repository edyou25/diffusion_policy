#!/usr/bin/env python3
"""查看 PushT Zarr 数据集的结构和内容"""

import zarr
import numpy as np

# 打开 zarr 数据集
store = zarr.open('data/pusht/pusht_cchi_v7_replay.zarr', mode='r')

print('=== Zarr 数据集结构 ===')
print(store.tree())
print()

# 查看数据组
print('=== 数据详情 ===')

# 访问 data 组中的数组
data_group = store['data']
print('\n[data 组]:')
for key in data_group.keys():
    arr = data_group[key]
    print(f'\n  {key}:')
    print(f'    形状: {arr.shape}')
    print(f'    数据类型: {arr.dtype}')
    print(f'    块大小: {arr.chunks}')
    
    # 计算大小
    total_size = arr.nbytes / 1024 / 1024
    print(f'    原始大小: {total_size:.2f} MB')
    
    # 显示一些统计信息
    if key == 'action':
        data = arr[:]
        print(f'    范围: [{data.min():.3f}, {data.max():.3f}]')
        print(f'    均值: {data.mean():.3f}')
        print(f'    标准差: {data.std():.3f}')
    elif key == 'state':
        # state 包含 (x, y, block_x, block_y, block_theta)
        data = arr[:]
        print(f'    agent_pos (x,y) 范围: [{data[:,:2].min():.1f}, {data[:,:2].max():.1f}]')
    elif key == 'img':
        print(f'    图像值范围: [{arr[0].min():.3f}, {arr[0].max():.3f}]')

# 访问 meta 组
meta_group = store['meta']
print('\n\n[meta 组]:')
for key in meta_group.keys():
    arr = meta_group[key]
    print(f'\n  {key}:')
    print(f'    形状: {arr.shape}')
    print(f'    数据类型: {arr.dtype}')
    
    if key == 'episode_ends':
        episode_ends = arr[:]
        num_episodes = len(episode_ends)
        print(f'    Episode 数量: {num_episodes}')
        
        # 计算每个 episode 的长度
        episode_lengths = np.diff(np.concatenate([[0], episode_ends]))
        print(f'    Episode 长度: min={episode_lengths.min()}, '
              f'max={episode_lengths.max()}, '
              f'mean={episode_lengths.mean():.1f}')
        
        # 总时间步数
        total_steps = episode_ends[-1]
        print(f'    总时间步: {total_steps}')

# 总结
print('\n\n=== 总结 ===')
total_size = sum(data_group[k].nbytes for k in data_group.keys()) / 1024 / 1024
print(f'数据集总大小: {total_size:.2f} MB')
print(f'Episodes: {len(meta_group["episode_ends"])}')
print(f'总时间步: {meta_group["episode_ends"][-1]}')
print(f'\n数据包含:')
print(f'  - img: 96x96 RGB 图像')
print(f'  - state: (agent_x, agent_y, block_x, block_y, block_theta)')
print(f'  - action: (delta_x, delta_y) 控制指令')
print(f'  - keypoint: 9个关键点坐标 (agent + block corners)')
print(f'  - n_contacts: 接触数量')