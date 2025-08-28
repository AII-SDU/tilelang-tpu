# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""RISC-V Vector (RVV) Kernel Adapter for TileLang JIT"""

import os
import hashlib
import inspect
import subprocess
import shutil
import torch
from typing import List, Callable
from .base import BaseKernelAdapter
from tilelang.engine.param import CompiledArtifact

def compile_to_elf(src_path: str, elf_path: str, arch: str = "rv64gcv") -> bool:
    """编译 C 源文件为 RISC-V ELF 可执行文件"""
    # 获取编译器路径
    compiler_path = shutil.which("g++")
    if not compiler_path:
        raise RuntimeError("找不到 RISC-V GCC 编译器")
    
    # 构建编译命令
    cc_cmd = [
        compiler_path,
        f"-march={arch}",
        "-mabi=lp64d",
        "-O0",
        src_path,
        "-o", elf_path,
    ]
    
    print(f"编译命令: {' '.join(cc_cmd)}")
    try:
        ret = subprocess.check_call(cc_cmd)
        return ret == 0
    except subprocess.CalledProcessError as e:
        print(f"编译失败: {e}")
        return False

def run_on_simulator(elf_path: str) -> bool:
    """在 RISC-V 模拟器上运行编译的程序"""
    # 获取模拟器路径
    qemu_riscv = shutil.which("qemu-riscv64")
    if qemu_riscv is None:
        print("警告：未找到 qemu-riscv64 模拟器")
        return False
    
    # 构建运行命令
    run_cmd = [qemu_riscv, elf_path]
    
    print(f"运行命令: {' '.join(run_cmd)}")
    try:
        result = subprocess.run(run_cmd, capture_output=True, text=True)
        print(f"模拟器输出: {result.stdout}")
        if result.stderr:
            print(f"模拟器错误: {result.stderr}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"模拟器运行失败: {e}")
        return False
    
class RVVKernelAdapter(BaseKernelAdapter):
    """
    RISC-V Vector Kernel Adapter that integrates with standard tilelang JIT interface
    """
    
    def __init__(self, compiled_artifact: CompiledArtifact, result_idx: List[int], fn_name: str = None, arch: str = "rv64gcv"):
        super().__init__(compiled_artifact.device_mod, compiled_artifact.params, result_idx)
        self.compiled_artifact = compiled_artifact
        
        # 获取函数名
        if fn_name is None:
            fn_name = "main"
        self.fn_name = fn_name
        
        # 转换负索引为正索引
        self.result_idx = []
        for idx in result_idx:
            if idx < 0:
                self.result_idx.append(len(compiled_artifact.params) + idx)
            else:
                self.result_idx.append(idx)
        self.arch = arch
        self.elf_path = None
        
    def _prepare_executable(self):
        """编译RVV内核到可执行文件"""
        if self.elf_path and os.path.exists(self.elf_path):
            return self.elf_path
            
        # 生成基于源代码哈希的唯一缓存路径
        c_code = self.compiled_artifact.kernel_source
        hash_val = hashlib.md5(c_code.encode()).hexdigest()
        cache_dir = os.path.join(os.getcwd(), ".tilelang_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 使用原始函数名
        kernel_fn_name = self.fn_name
        
        c_path = os.path.join(cache_dir, f"{kernel_fn_name}_{hash_val}.c")
        self.elf_path = os.path.join(cache_dir, f"{kernel_fn_name}_{hash_val}.elf")
        
        if os.path.exists(self.elf_path):
            print(f"使用缓存的内核: {self.elf_path}")
            return self.elf_path
            
        # 添加必要的头文件
        header_includes = """
#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stddef.h>
#include <float.h>
"""
        
        # 在代码开头添加头文件
        c_code = header_includes + c_code
            
        # 将C代码写入文件，然后编译文件
        with open(c_path, "w") as f:
            f.write(c_code)
        print(f"生成的 C 代码已保存到: {c_path}")
        
        if not compile_to_elf(c_path, self.elf_path, self.arch):
            raise RuntimeError(f"编译 {c_path} 失败")
        print(f"编译成功: {self.elf_path}")
            
        return self.elf_path

    def _convert_torch_func(self) -> Callable:
        """转换为torch函数"""
        
        def torch_func(*args):
            elf_path = self._prepare_executable()
            
            # 处理参数
            tensors = []
            tensors_index = []
            fp_scalars = []
            fp_scalars_index = []
            fixed_scalars = []
            fixed_scalars_index = []
            
            for index, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    tensors.append(arg)
                    tensors_index.append(index)
                else:
                    if isinstance(arg, (int, bool)):
                        fixed_scalars.append(arg)
                        fixed_scalars_index.append(index)
                    elif isinstance(arg, float):
                        fp_scalars.append(arg)
                        fp_scalars_index.append(index)
                    else:
                        raise TypeError(f"不支持的参数类型: {type(arg)}")
            
            print(f"执行内核: {self.fn_name}")
            print(f"ELF 路径: {elf_path}")
            print(f"张量参数: {[(t.shape, t.dtype, t.device) for t in tensors]}")
            
            # 在模拟器中运行
            if not run_on_simulator(elf_path):
                print("警告：模拟器运行失败，尝试其他执行方式")
                # 这里可以添加其他执行方式，如通过RPC连接到真实硬件

        return torch_func
    
    def __getitem__(self, grid):
        """支持grid语法"""
        def runner(*args):
            elf_path = self._prepare_executable()
            
            # 处理参数
            tensors = []
            tensors_index = []
            fp_scalars = []
            fp_scalars_index = []
            fixed_scalars = []
            fixed_scalars_index = []
            
            for index, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    tensors.append(arg)
                    tensors_index.append(index)
                else:
                    if isinstance(arg, (int, bool)):
                        fixed_scalars.append(arg)
                        fixed_scalars_index.append(index)
                    elif isinstance(arg, float):
                        fp_scalars.append(arg)
                        fp_scalars_index.append(index)
                    else:
                        raise TypeError(f"不支持的参数类型: {type(arg)}")
            
            print(f"执行内核: {self.fn_name}")
            print(f"ELF 路径: {elf_path}")
            print(f"张量参数: {[(t.shape, t.dtype, t.device) for t in tensors]}")
            
            # 在模拟器中运行
            if not run_on_simulator(elf_path):
                print("警告：模拟器运行失败，尝试其他执行方式")

        return runner
        
    def get_kernel_source(self) -> str:
        """获取生成的RVV内核源码"""
        return self.compiled_artifact.kernel_source
        
    def get_profiler(self, tensor_supply_type=None):
        """获取性能分析器，为RVV设备定制"""
        from tilelang.profiler import Profiler
        from tilelang.engine.param import KernelParam
        import torch
        import time
        
        # 创建RVV专用的profiler
        profiler = Profiler(
            self.params,
            self.result_idx,
            tensor_supply_type or "Auto"
        ).with_default_adapter(self)
        
        # 重写supply函数，使其生成适合RVV的张量
        original_supply = profiler.supply
        
        def rvv_supply(param: KernelParam) -> torch.Tensor:
            # 创建适合RVV的张量
            dtype: torch.dtype = param.dtype
            shape = tuple(int(s) for s in param.shape)
            
            # 根据tensor_supply_type生成相应的张量
            if tensor_supply_type in ['Auto', 'Random', 'Integer']:
                return torch.randn(*shape).to(dtype)
            elif tensor_supply_type == 'Zero':
                return torch.zeros(*shape, dtype=dtype)
            elif tensor_supply_type == 'One':
                return torch.ones(*shape, dtype=dtype)
            else:
                # 默认使用随机张量
                return torch.randn(*shape).to(dtype)
            
        profiler.supply = rvv_supply
        
        # 重写do_bench接口，添加RVV特定的性能分析
        original_do_bench = profiler.do_bench
        
        def rvv_do_bench(func=None, warmup=25, rep=100, n_warmup=1, n_repeat=1, input_tensors=None):
            """RVV特定的性能分析"""
            print(f"RVV性能分析: 使用模拟器进行基准测试")
            
            # 编译内核
            elf_path = self._prepare_executable()
            
            # 使用模拟器进行基准测试
            start_time = time.time()
            
            # 运行多次以获取平均性能
            for i in range(warmup):
                run_on_simulator(elf_path)
            
            times = []
            for i in range(rep):
                iter_start = time.time()
                run_on_simulator(elf_path)
                iter_end = time.time()
                times.append(iter_end - iter_start)
            
            end_time = time.time()
            
            # 计算统计信息
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                print(f"平均时间: {avg_time*1000:.2f}ms, 最小时间: {min_time*1000:.2f}ms, 最大时间: {max_time*1000:.2f}ms")
                return avg_time
            else:
                print("无法测量性能")
                return float('inf')
        
        profiler.do_bench = rvv_do_bench
        return profiler