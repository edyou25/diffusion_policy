#!/usr/bin/env python3
"""快速测试CUDA是否工作正常"""

import torch
import time
import sys

print("=" * 60)
print("PyTorch CUDA 诊断测试")
print("=" * 60)

# 基本信息
print(f"\n📦 版本信息:")
print(f"  PyTorch版本: {torch.__version__}")
print(f"  CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA版本: {torch.version.cuda}")
    print(f"  cuDNN版本: {torch.backends.cudnn.version()}")

if not torch.cuda.is_available():
    print("\n❌ CUDA不可用！")
    print("原因可能是：")
    print("  1. 没有安装GPU版本的PyTorch")
    print("  2. CUDA驱动问题")
    print("  3. GPU不可用")
    sys.exit(1)

print(f"\n🎮 GPU信息:")
print(f"  GPU数量: {torch.cuda.device_count()}")
print(f"  当前GPU: {torch.cuda.current_device()}")
print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"  GPU显存: {mem_total:.2f} GB")

print(f"\n🔧 CUDA架构支持:")
capability = torch.cuda.get_device_capability(0)
print(f"  GPU计算能力: sm_{capability[0]}{capability[1]}")
arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else []
if arch_list:
    print(f"  PyTorch支持的架构: {', '.join(arch_list)}")
    if f"sm_{capability[0]}{capability[1]}" not in ' '.join(arch_list):
        print(f"  ⚠️  警告: 当前GPU架构 sm_{capability[0]}{capability[1]} 不在支持列表中!")
        print(f"  这可能导致性能下降或崩溃")
        print(f"  建议升级PyTorch到2.5+")

print("\n" + "=" * 60)
print("🧪 运行CUDA测试")
print("=" * 60)

tests_passed = 0
tests_failed = 0

try:
    # 测试1: 创建张量
    print("\n1️⃣  创建CPU张量...")
    x = torch.randn(1000, 1000)
    print("   ✅ 成功")
    tests_passed += 1
except Exception as e:
    print(f"   ❌ 失败: {e}")
    tests_failed += 1

try:
    # 测试2: 迁移到GPU
    print("\n2️⃣  迁移到GPU...")
    print("   (如果这里卡住>10秒，说明有严重的兼容性问题)")
    start = time.time()
    x = x.cuda()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    if elapsed > 5:
        print(f"   ⚠️  完成但耗时过长: {elapsed:.3f}秒")
        print("   这表明GPU初始化有问题")
    else:
        print(f"   ✅ 成功 (耗时: {elapsed:.3f}秒)")
    tests_passed += 1
except Exception as e:
    print(f"   ❌ 失败: {e}")
    tests_failed += 1
    sys.exit(1)

try:
    # 测试3: GPU计算
    print("\n3️⃣  GPU矩阵乘法...")
    start = time.time()
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   ✅ 成功 (耗时: {elapsed:.3f}秒)")
    tests_passed += 1
except Exception as e:
    print(f"   ❌ 失败: {e}")
    tests_failed += 1

try:
    # 测试4: 迭代计算
    print("\n4️⃣  迭代测试（10次矩阵乘法）...")
    start = time.time()
    for i in range(10):
        y = torch.matmul(x, x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    avg = elapsed / 10
    print(f"   ✅ 成功 (总耗时: {elapsed:.3f}秒, 平均: {avg:.3f}秒/次)")
    tests_passed += 1
except Exception as e:
    print(f"   ❌ 失败: {e}")
    tests_failed += 1

try:
    # 测试5: 大张量测试（接近实际训练）
    print("\n5️⃣  大张量测试 (模拟batch)...")
    # 模拟 batch_size=64, image=96x96x3, 类似实际训练
    batch = torch.randn(64, 3, 96, 96).cuda()
    start = time.time()
    # 简单卷积操作
    conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
    output = conv(batch)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   输入shape: {batch.shape}")
    print(f"   输出shape: {output.shape}")
    print(f"   ✅ 成功 (耗时: {elapsed:.3f}秒)")
    tests_passed += 1
except Exception as e:
    print(f"   ❌ 失败: {e}")
    tests_failed += 1

try:
    # 测试6: 显存分配测试
    print("\n6️⃣  显存分配测试...")
    mem_before = torch.cuda.memory_allocated() / 1024**2
    print(f"   当前显存使用: {mem_before:.2f} MB")
    
    # 尝试分配1GB
    large_tensor = torch.randn(256, 1024, 1024).cuda()
    mem_after = torch.cuda.memory_allocated() / 1024**2
    print(f"   分配后显存使用: {mem_after:.2f} MB")
    print(f"   分配了: {mem_after - mem_before:.2f} MB")
    
    # 释放
    del large_tensor
    torch.cuda.empty_cache()
    mem_final = torch.cuda.memory_allocated() / 1024**2
    print(f"   释放后显存使用: {mem_final:.2f} MB")
    print(f"   ✅ 成功")
    tests_passed += 1
except Exception as e:
    print(f"   ❌ 失败: {e}")
    tests_failed += 1

# 总结
print("\n" + "=" * 60)
print("📊 测试总结")
print("=" * 60)
print(f"通过: {tests_passed}/{tests_passed + tests_failed}")
print(f"失败: {tests_failed}/{tests_passed + tests_failed}")

if tests_failed == 0:
    print("\n✅ 所有测试通过！GPU工作正常")
    print("\n💡 建议:")
    if capability[0] >= 12:  # sm_120+
        print("  - 你的GPU是Blackwell架构（sm_120+）")
        print("  - 如果PyTorch < 2.5，强烈建议升级:")
        print("    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121")
else:
    print(f"\n❌ {tests_failed} 个测试失败")
    print("\n🔧 建议:")
    print("  1. 升级PyTorch到2.5+")
    print("  2. 检查CUDA驱动是否正确安装")
    print("  3. 尝试重启系统")
    print("  4. 如果问题持续，考虑使用CPU训练")

print("\n" + "=" * 60)

