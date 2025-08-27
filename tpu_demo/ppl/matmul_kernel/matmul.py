#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

"""TPU JIT 编译API方式 - matmul 示例"""

import torch
import torch_tpu
import tilelang
import tilelang.language as T


@T.prim_func
def matmul_func(
        A: T.Tensor((384, 786), "float16"),
        B: T.Tensor((786, 786), "float16"),
        C: T.Tensor((384, 786), "float32"),
):
    """矩阵乘法内核 - 编译API版本"""
    with T.Kernel(T.ceildiv(786, 128), T.ceildiv(384, 128), is_cpu=True) as (bx, by):
        A_shared = T.alloc_shared((128, 128), "float16")
        B_shared = T.alloc_shared((128, 128), "float16")
        C_shared = T.alloc_shared((128, 128), "float32")

        T.ppl_fill(C_shared, T.float32(0))
        for k in T.Pipelined(T.ceildiv(786, 128), num_stages=2):
            T.ppl_copy(A[by * 128, k * 128], A_shared)
            T.ppl_copy(B[k * 128, bx * 128], B_shared)
            T.ppl_gemm(A_shared, B_shared, C_shared)

        T.ppl_copy(C_shared, C[by * 128, bx * 128])


if __name__ == "__main__":
    print("=== TPU JIT 编译API方式 - matmul 测试 ===")
    
    # 编译内核
    print("编译内核...")
    matmul = tilelang.compile(matmul_func, target="tpu", out_idx=[-1])
    print(f"✅ 编译完成: {type(matmul)}")
    
    # 创建输入和输出张量
    print("创建输入和输出张量...")
    device = "tpu:0"
    A = torch.rand((384, 786), device=device, dtype=torch.float16)
    B = torch.rand((786, 786), device=device, dtype=torch.float16)
    C = torch.empty((384, 786), device=device, dtype=torch.float32)

    print(f"输入张量设备: A={A.device}, B={B.device}")
    print(f"输入张量形状: A={A.shape}, B={B.shape}")
    print(f"输出张量形状: C={C.shape}")

    # 使用编译API版本调用
    print("执行 TPU JIT 编译API内核...")
    matmul[(1,)](A, B, C)

    # 验证结果
    print("验证结果...")
    expected = torch.mm(A.float(), B.float())
    max_diff = torch.max(torch.abs(C - expected)).item()
    mean_diff = torch.mean(torch.abs(C - expected)).item()

    print(f"最大差异: {max_diff:.8f}")
    print(f"平均差异: {mean_diff:.8f}")
    print(f"是否接近相同: {torch.allclose(C, expected, rtol=1e-3, atol=1e-3)}")

    # 测试profiler功能
    print("\n测试profiler功能...")
    source = matmul.get_kernel_source()
    print(f"✅ get_kernel_source(): 获取源码 {len(source)} 字符")
    
    profiler = matmul.get_profiler()
    print(f"✅ get_profiler(): {type(profiler)}")

    if max_diff < 1e-2:  # 由于float16精度，放宽容差
        print("\n🎉 TPU JIT 编译API方式测试成功！")
        print("✅ 使用: tilelang.compile(func, target='tpu', out_idx=[-1])")
    else:
        print(f"\n❌ 测试失败，最大差异: {max_diff}")

    print("\n=== 测试完成 ===")

# for mm in range(64,4097,64):
#     for nn in range(64, 1025, 64):
#         for kk in range(64, 8193, 64):
#             for stages in [0, 2, 3, 4]:
#               if 4096 % mm != 0 or 8192 % nn != 0 or 1024 % kk != 0:
#                   continue
#               func = matmul(4096, 8192, 1024, mm, nn, kk, stages)
#               mod: str = tilelang.lower(func)
#               if mod.count(".addr = 0") > 1:
#                   print(f"{mm}_{nn}_{kk} configure failed!")
#                   continue
#               mod = mod.replace("""#include "ppl_helper.h"
# static data_type_t __ppl_get_dtype(int type) {
#   data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
#     DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
#     DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
#     DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
#   return __dtype[type];
# }

# void main(global_addr_t v1, global_addr_t v2, global_addr_t v3) {""", """#include "ppl_helper.h"

# static data_type_t __ppl_get_dtype(int type) {
#   data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
#       DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
#       DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
#       DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
#   return __dtype[type];
# }

# typedef struct {
#   dim4 shape;
#   dim4 stride;
#   global_addr_t addr;
#   data_type_t dtype;
#   int mode;
#   int align_mode;
#   int size;
#   int offset;
#   bool unsigned_flag;
#   bool default_stride;
# } __ppl_tensor_info;
# typedef struct {
#   global_addr_t ptr_left_v1;
#   global_addr_t ptr_right_v2;
#   global_addr_t ptr_res_v3;
# } tpu_kernel_api_mm2_fp16_t;
# void mm2_fp16_inner(global_addr_t v1, global_addr_t v2, global_addr_t v3) {""")
#               mod = mod.replace("""typedef struct {
#   global_addr_t v1;
#   global_addr_t v2;
#   global_addr_t v3;
# } tpu_kernel_api_main_args_t;
# void main_kernel(const void * args) {
#   tpu_kernel_api_main_args_t *api = (tpu_kernel_api_main_args_t*)args;
#   main(api->v1,
#     api->v2,
#     api->v3);
#   tpu_poll();
# }
# TPUKERNEL_FUNC_REGISTER(main_kernel)""", """int mm2_fp16(const void * args) {
#   tpu_kernel_api_mm2_fp16_t *api = (tpu_kernel_api_mm2_fp16_t*)args;
#   tpu_initialize();
#   mm2_fp16_inner(api->ptr_left_v1,
#     api->ptr_right_v2,
#     api->ptr_res_v3);
#   tpu_poll();
#   return 0;
# }
# TPUKERNEL_FUNC_REGISTER(mm2_fp16)""")

#               with open(f"kernel_a/matmul_{mm}_{nn}_{kk}_{stages}.c", "w") as f:
#                   f.write(mod)
