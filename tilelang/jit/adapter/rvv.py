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

def c_to_h(src_path: str, h_path: str, arch: str = "rv64gcv") -> bool:
    """将C源码转换为规范化的头文件，仅保留源文件中定义的函数声明
    
    Args:
        src_path: 输入的C源文件路径
        h_path: 输出的头文件路径
        arch: 目标架构（默认rv64gcv）
    
    Returns:
        bool: 是否转换成功
    """
    # 获取编译器路径 - 使用gcc
    compiler_path = shutil.which("gcc")
    if not compiler_path:
        raise RuntimeError("找不到 RISC-V GCC 编译器")
    
    # 获取源文件的绝对路径用于过滤
    abs_src_path = os.path.abspath(src_path)
    
    # 临时文件路径
    temp_h_path = h_path + ".tmp"
    
    # 构建编译命令生成原始头文件
    cc_cmd = [
        compiler_path,
        f"-march={arch}",
        "-aux-info",
        temp_h_path,
        src_path,
    ]
    print(f"h文件转换命令: {' '.join(cc_cmd)}")
    try:
        # 1. 生成原始头文件
        result = subprocess.run(
            cc_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 即使有链接错误，只要头文件生成成功就继续
        if not os.path.exists(temp_h_path):
            print(f"[错误] 头文件未生成: {result.stderr}")
            return False
        
        # 2. 读取原始内容并过滤
        filtered_content = []
        with open(temp_h_path, 'r') as f:
            for line in f:
                # 只保留源文件中定义的函数（根据路径匹配）
                if abs_src_path in line:
                    filtered_content.append(line)
        
        if not filtered_content:
            print(f"[警告] 未找到源文件中的函数声明: {abs_src_path}")
            # 仍然创建头文件但内容为空
        
        # 3. 生成保护宏名称
        header_guard = f"TILELANG_{hashlib.md5(os.path.basename(h_path).encode()).hexdigest().upper()}_H"
        
        # 4. 写入规范化的头文件
        with open(h_path, 'w') as f:
            f.write(f"""#ifndef {header_guard}
#define {header_guard}

// 基本头文件包含
#include <riscv_vector.h>  // RVV内置类型
#include <stddef.h>       // size_t

#ifdef __cplusplus
extern "C" {{
#endif

/* 源文件中定义的函数 */
{"".join(filtered_content)}

#ifdef __cplusplus
}}
#endif

#endif // {header_guard}
""")
        
        # 5. 删除临时文件
        if os.path.exists(temp_h_path):
            os.unlink(temp_h_path)
            
        print(f"成功生成过滤后的头文件: {h_path}")
        return True
        
    except Exception as e:
        print(f"[错误] 头文件生成异常: {str(e)}")
        if os.path.exists(temp_h_path):
            os.unlink(temp_h_path)
        return False

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
        h_path = os.path.join(cache_dir, f"{kernel_fn_name}_{hash_val}.h")
        self.elf_path = os.path.join(cache_dir, f"{kernel_fn_name}_{hash_val}")
        
        if os.path.exists(self.elf_path):
            print(f"使用缓存的内核: {self.elf_path}")
            return self.elf_path
            
        # 添加必要的头文件
        header_includes = """#include <riscv_vector.h>
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

        if not c_to_h(c_path, h_path, self.arch):
            raise RuntimeError(f"转换 {c_path} 为 {h_path} 失败")
        print(f"转换成功: {h_path}")

        if not compile_to_elf(c_path, self.elf_path, self.arch):
            raise RuntimeError(f"编译 {c_path} 失败")
        print(f"编译成功: {self.elf_path}")
            
        return self.elf_path

    def _convert_torch_func(self) -> Callable:
        """转换为torch函数"""
        
        def torch_func(*args):
            elf_path = self._prepare_executable()

        return torch_func
    
    def __getitem__(self, grid):
        """支持grid语法"""
        def runner(*args):
            elf_path = self._prepare_executable()

        return runner
        
    def get_kernel_source(self) -> str:
        """获取生成的RVV内核源码"""
        return self.compiled_artifact.kernel_source