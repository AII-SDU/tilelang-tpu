#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

"""TPU JIT 编译API方式 - reduce_sum 示例"""

import torch
import torch_tpu
import tilelang
import tilelang.language as T


@T.prim_func
def reduce_sum_func(
        X: T.Tensor((8192, 8192), "float32"),
        Y: T.Tensor((8192, 1), "float32"),
):
    """归约求和内核 - 编译API版本"""
    with T.Kernel(1, T.ceildiv(8192, 512), is_cpu=True) as (bx, by):
        # 分配共享内存
        X_shared = T.alloc_shared((512, 8192), "float32")
        Y_shared = T.alloc_shared((512, 1), "float32")
        # 初始化 Y_shared 为0
        T.ppl_fill(Y_shared, T.float32(float(0)))
        # 复制输入数据到共享内存
        T.ppl_copy(X[by * 512, 0], X_shared)
        # 执行 reduce_sum 操作
        # 参数说明：输入张量，输出张量，沿哪个维度执行(1表示列方向)
        T.ppl_reduce_sum(X_shared, Y_shared, 1)
        # 将结果从共享内存复制回全局内存
        T.ppl_copy(Y_shared, Y[by * 512, 0])


if __name__ == "__main__":
    print("=== TPU JIT 编译API方式 - reduce_sum 测试 ===")
    
    # 编译内核
    print("编译内核...")
    reduce_sum = tilelang.compile(reduce_sum_func, target="tpu", out_idx=[-1])
    print(f"✅ 编译完成: {type(reduce_sum)}")
    
    # 创建输入和输出张量
    print("创建输入和输出张量...")
    device = "tpu:0"
    input_tensor = torch.rand((8192, 8192), device=device)
    output_tensor = torch.empty((8192, 1), device=device)

    print(f"输入张量设备: {input_tensor.device}")
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")

    # 使用编译API版本调用
    print("执行 TPU JIT 编译API内核...")
    reduce_sum[(1,)](input_tensor, output_tensor)

    # 验证结果
    print("验证结果...")
    expected = torch.sum(input_tensor, dim=1, keepdim=True)
    max_diff = torch.max(torch.abs(output_tensor - expected)).item()
    mean_diff = torch.mean(torch.abs(output_tensor - expected)).item()

    print(f"最大差异: {max_diff:.8f}")
    print(f"平均差异: {mean_diff:.8f}")
    print(f"是否完全相同: {torch.allclose(output_tensor, expected)}")

    # 测试profiler功能
    print("\n测试profiler功能...")
    source = reduce_sum.get_kernel_source()
    print(f"✅ get_kernel_source(): 获取源码 {len(source)} 字符")
    
    profiler = reduce_sum.get_profiler()
    print(f"✅ get_profiler(): {type(profiler)}")

    if max_diff < 1e-5:
        print("\n🎉 TPU JIT 编译API方式测试成功！")
        print("✅ 使用: tilelang.compile(func, target='tpu', out_idx=[-1])")
    else:
        print(f"\n❌ 测试失败，最大差异: {max_diff}")

    print("\n=== 测试完成 ===")