#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

"""TPU JIT 装饰器方式 - reduce_max 示例"""

import torch
import torch_tpu
import tilelang
import tilelang.language as T


@tilelang.jit(target="tpu", out_idx=[-1])
@T.prim_func
def reduce_max(
        X: T.Tensor((8192, 1020), "float32"),
        Y: T.Tensor((8192, 1), "float32"),
):
    """归约最大值内核 - 装饰器版本"""
    with T.Kernel(1, T.ceildiv(8192, 512), is_cpu=True) as (bx, by):
        # 分配共享内存
        X_shared = T.alloc_shared((512, 1020), "float32")
        Y_shared = T.alloc_shared((512, 1), "float32")
        # 复制输入数据到共享内存
        T.ppl_copy(X[by * 512, 0], X_shared)
        # 执行 reduce_max 操作
        # 参数说明：输入张量，输出张量，沿哪个维度执行(1表示列方向)，是否保持维度
        T.ppl_reduce_max(X_shared, Y_shared, 1, True)
        # 将结果从共享内存复制回全局内存
        T.ppl_copy(Y_shared, Y[by * 512, 0])


if __name__ == "__main__":
    print("=== TPU JIT 装饰器方式 - reduce_max 测试 ===")
    
    # 创建输入和输出张量
    print("创建输入和输出张量...")
    device = "tpu:0"
    input_tensor = torch.rand((8192, 1020), device=device)
    output_tensor = torch.empty((8192, 1), device=device)

    print(f"输入张量设备: {input_tensor.device}")
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")

    # 使用装饰器版本调用
    print("执行 TPU JIT 装饰器内核...")
    reduce_max[(1,)](input_tensor, output_tensor)

    # 验证结果
    print("验证结果...")
    expected = torch.max(input_tensor, dim=1, keepdim=True)[0]
    max_diff = torch.max(torch.abs(output_tensor - expected)).item()
    mean_diff = torch.mean(torch.abs(output_tensor - expected)).item()

    print(f"最大差异: {max_diff:.8f}")
    print(f"平均差异: {mean_diff:.8f}")
    print(f"是否完全相同: {torch.allclose(output_tensor, expected)}")

    # 测试profiler功能
    print("\n测试profiler功能...")
    source = reduce_max.get_kernel_source()
    print(f"✅ get_kernel_source(): 获取源码 {len(source)} 字符")
    
    profiler = reduce_max.get_profiler()
    print(f"✅ get_profiler(): {type(profiler)}")

    if max_diff < 1e-5:
        print("\n🎉 TPU JIT 装饰器方式测试成功！")
        print("✅ 使用: @tilelang.jit(target='tpu', out_idx=[-1])")
    else:
        print(f"\n❌ 测试失败，最大差异: {max_diff}")

    print("\n=== 测试完成 ===")