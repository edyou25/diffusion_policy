#!/usr/bin/env python3
"""测试神经网络是否能在GPU上运行"""

import torch
import torch.nn as nn
import time

print("测试神经网络在GPU上的运行...")

try:
    # 创建一个小的CNN
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
    )
    
    # 移到GPU
    print("1. 移动模型到GPU...")
    model = model.cuda()
    print("   ✅ 成功")
    
    # 创建输入
    print("2. 创建输入batch...")
    x = torch.randn(4, 3, 32, 32).cuda()
    print("   ✅ 成功")
    
    # 前向传播
    print("3. 前向传播...")
    start = time.time()
    y = model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   ✅ 成功 (耗时: {elapsed:.3f}秒)")
    print(f"   输出shape: {y.shape}")
    
    # 反向传播
    print("4. 反向传播...")
    loss = y.sum()
    loss.backward()
    print("   ✅ 成功")
    
    # 多次迭代
    print("5. 迭代测试（10次）...")
    start = time.time()
    for i in range(10):
        y = model(x)
        loss = y.sum()
        loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   ✅ 成功 (总耗时: {elapsed:.3f}秒, 平均: {elapsed/10:.3f}秒/次)")
    
    print("\n✅ 所有测试通过！神经网络可以在GPU上运行")
    print("💡 建议: 可以尝试在GPU上训练")
    
except Exception as e:
    print(f"\n❌ 失败: {e}")
    print("\n💡 建议:")
    print("  1. 尝试安装PyTorch nightly版本")
    print("  2. 或使用CPU训练")
    import traceback
    traceback.print_exc()

