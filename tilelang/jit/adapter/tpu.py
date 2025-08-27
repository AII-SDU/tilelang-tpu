# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""TPU Kernel Adapter for TileLang JIT"""

import os
import hashlib
import inspect
import subprocess
import shutil
import torch
from typing import List, Tuple, Optional, Callable
from .base import BaseKernelAdapter
from tilelang.engine.param import CompiledArtifact


def _auto_source_sophgo_env():
    """自动source SophGo TPU环境变量"""
    if os.getenv("SOPHGO_TPU_ROOT") is not None:
        # 环境变量已经设置，不需要重复source
        return
        
    # 尝试相对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 从 tilelang/jit/adapter/tpu.py 到项目根目录
    project_root = os.path.join(current_dir, "..", "..", "..")
    envsetup_path = os.path.join(project_root, "3rdparty", "sophgo_tpu", "envsetup.sh")
    envsetup_path = os.path.abspath(envsetup_path)
    
    if os.path.exists(envsetup_path):
        try:
            # 执行envsetup.sh并获取环境变量
            result = subprocess.run(
                f"source {envsetup_path} && env",
                shell=True, capture_output=True, text=True, executable='/bin/bash'
            )
            
            if result.returncode == 0:
                # 解析环境变量
                for line in result.stdout.split('\n'):
                    if '=' in line and (line.startswith('SOPHGO_TPU') or line.startswith('PATH=')):
                        key, value = line.split('=', 1)
                        if key == 'PATH':
                            # 对于PATH，只添加新的部分
                            existing_path = os.getenv('PATH', '')
                            if existing_path not in value:
                                os.environ[key] = value
                        else:
                            os.environ[key] = value
                
                print(f"自动加载SophGo TPU环境: {envsetup_path}")
            else:
                print(f"警告：无法source环境脚本: {envsetup_path}")
        except Exception as e:
            print(f"警告：自动加载环境脚本时出错: {e}")
    else:
        print(f"警告：未找到环境脚本: {envsetup_path}")


# 在模块加载时自动执行
_auto_source_sophgo_env()


def compile_to_so(src_path: str, so_path: str, chip: str = "bm1690") -> bool:
    """
    Compile C source file to .so shared library for TPU.
    直接复制自原始 tpujit.py 的工作实现
    """
    os.makedirs(os.path.dirname(so_path), exist_ok=True)
    os.environ["CHIP_ARCH"] = chip
    os.environ["CHIP"] = chip
    sophgo_tpu_root = os.getenv("SOPHGO_TPU_ROOT")
    if not sophgo_tpu_root:
        print("尝试加载3rd-party 中sophgo_tpu环境")
        sophgo_tpu_root = "/home/kingdom/SepTran/tilelang-tpu/3rdparty/sophgo_tpu"
        if not os.path.exists(sophgo_tpu_root):
            raise ValueError("SOPHGO_TPU_ROOT 环境变量加载失败, 请设置 sophgo_tpu 项目根目录。")
    include_dirs = [
        os.path.join(sophgo_tpu_root, "runtime/customize/include"),
        os.path.join(sophgo_tpu_root, "runtime/kernel"),
        os.path.join(sophgo_tpu_root, f"runtime/{chip}/TPU1686/kernel/include")
    ]
    libraries = ["tpuv7_emulator" if chip in ("bm1690", "sg2262") else "cmodel_firmware"]
    library_dirs = [
        os.path.join(sophgo_tpu_root, f"runtime/{chip}/tpuv7-runtime-emulator/lib")
        if chip in ("bm1690", "sg2262") else
        os.path.join(sophgo_tpu_root, f"runtime/{chip}/lib")
    ]
    cc = os.environ.get("CC")
    if cc is None:
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
        if cc is None:
            raise RuntimeError("找不到 C 编译器。请通过 CC 环境变量指定。")
    cc_cmd = [cc, src_path, "-O2", "-shared", "-fPIC", "-Wno-psabi", "-o", so_path]
    cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    cc_cmd += [f'-l{lib}' for lib in libraries]
    cc_cmd += [f"-L{dir}" for dir in library_dirs]
    print(f"编译命令: {' '.join(cc_cmd)}")
    try:
        ret = subprocess.check_call(cc_cmd)
        return ret == 0
    except subprocess.CalledProcessError as e:
        print(f"编译失败: {e}")
        return False


class TPUKernelAdapter(BaseKernelAdapter):
    """
    TPU Kernel Adapter that integrates with standard tilelang JIT interface
    """
    
    def __init__(self, compiled_artifact: CompiledArtifact, result_idx: List[int], fn_name: str = None, chip: str = "bm1690"):
        # Extract parameters from compiled artifact
        super().__init__(compiled_artifact.device_mod, compiled_artifact.params, result_idx)
        self.compiled_artifact = compiled_artifact
        # Try to get function name from the compiled artifact
        if fn_name is None:
            if hasattr(compiled_artifact.device_mod, 'get_global_vars'):
                global_vars = compiled_artifact.device_mod.get_global_vars()
                if global_vars:
                    fn_name = list(global_vars.keys())[0]
            if fn_name is None:
                fn_name = "main"
        self.fn_name = fn_name
        
        # Convert negative indices to positive
        self.result_idx = []
        for idx in result_idx:
            if idx < 0:
                self.result_idx.append(len(compiled_artifact.params) + idx)
            else:
                self.result_idx.append(idx)
        self.chip = chip
        self.so_path = None
        
    def _prepare_shared_library(self):
        """编译TPU内核到共享库，确保使用正确的函数名"""
        if self.so_path and os.path.exists(self.so_path):
            return self.so_path
            
        # 生成基于源代码哈希的唯一缓存路径
        c_code = self.compiled_artifact.kernel_source
        hash_val = hashlib.md5(c_code.encode()).hexdigest()
        cache_dir = os.path.join(os.getcwd(), ".tilelang_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 关键发现：TileLang编译器会自动为所有函数生成"原名_kernel"的wrapper函数
        # 无论原函数名是什么，TPU runtime必须调用wrapper函数，因为只有wrapper被注册
        kernel_fn_name = f"{self.fn_name}_kernel" if not self.fn_name.endswith("_kernel") else self.fn_name
        
        c_path = os.path.join(cache_dir, f"{kernel_fn_name}_{hash_val}.c")
        self.so_path = os.path.join(cache_dir, f"{kernel_fn_name}_{hash_val}.so")
        
        if os.path.exists(self.so_path):
            print(f"使用缓存的内核: {self.so_path}")
            # 确保函数名正确
            self.fn_name = kernel_fn_name
            return self.so_path
            
        # 添加tensor_info结构体到C代码，完全复制原始方式
        tensor_info_struct = """
typedef struct {
  dim4 shape;
  dim4 stride;
  global_addr_t addr;
  data_type_t dtype;
  int mode;
  int align_mode;
  int size;
  int offset;
  bool unsigned_flag;
  bool default_stride;
} __ppl_tensor_info;
"""
        
        include_line = c_code.find("#include")
        if include_line != -1:
            next_line = c_code.find("\n", include_line)
            if next_line != -1:
                c_code = c_code[:next_line+1] + tensor_info_struct + c_code[next_line+1:]
            else:
                c_code = c_code + "\n" + tensor_info_struct
        else:
            c_code = tensor_info_struct + c_code
            
        # 将C代码写入文件，然后编译文件（完全复制原始方式）
        with open(c_path, "w") as f:
            f.write(c_code)
        print(f"生成的 C 代码已保存到: {c_path}")
        
        if not compile_to_so(c_path, self.so_path, self.chip):
            raise RuntimeError(f"编译 {c_path} 失败")
        print(f"编译成功: {self.so_path}")
        
        # 确保函数名正确
        self.fn_name = kernel_fn_name
            
        return self.so_path

    def _convert_torch_func(self) -> Callable:
        """转换为torch函数，完全复制原始TileLangKernel的执行逻辑"""
        
        def torch_func(*args):
            so_path = self._prepare_shared_library()
            
            # 完全复制原始TileLangKernel的参数处理逻辑
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
            print(f"SO 路径: {so_path}")
            print(f"张量参数: {[(t.shape, t.dtype, t.device) for t in tensors]}")
            print(f"张量索引: {tensors_index}")
            print(f"浮点标量: {fp_scalars}")
            print(f"浮点标量索引: {fp_scalars_index}")
            print(f"固定标量: {fixed_scalars}")
            print(f"固定标量索引: {fixed_scalars_index}")
            
            # 完全复制原始的torch.ops.my_ops.dynlib_execute调用
            torch.ops.my_ops.dynlib_execute(
                so_path, 
                self.fn_name, 
                tensors, 
                tensors_index,
                fp_scalars, 
                fp_scalars_index, 
                fixed_scalars,
                fixed_scalars_index
            )

        return torch_func
    
    def __getitem__(self, grid):
        """支持grid语法，完全复制原始TileLangKernel的方式"""
        def runner(*args):
            # 完全复制原始TileLangKernel的runner逻辑
            so_path = self._prepare_shared_library()
            
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
            print(f"SO 路径: {so_path}")
            print(f"张量参数: {[(t.shape, t.dtype, t.device) for t in tensors]}")
            print(f"张量索引: {tensors_index}")
            print(f"浮点标量: {fp_scalars}")
            print(f"浮点标量索引: {fp_scalars_index}")
            print(f"固定标量: {fixed_scalars}")
            print(f"固定标量索引: {fixed_scalars_index}")
            
            torch.ops.my_ops.dynlib_execute(
                so_path, 
                self.fn_name, 
                tensors, 
                tensors_index,
                fp_scalars, 
                fp_scalars_index, 
                fixed_scalars,
                fixed_scalars_index
            )
        return runner
        
    def get_kernel_source(self) -> str:
        """获取生成的C内核源码"""
        return self.compiled_artifact.kernel_source
        
    def get_profiler(self, tensor_supply_type=None):
        """获取性能分析器，为TPU设备定制"""
        from tilelang.profiler import Profiler, TensorSupplyType
        from tilelang.utils import TensorSupplyType as UtilsTensorSupplyType
        from tilelang.engine.param import KernelParam
        import torch
        import time
        
        if tensor_supply_type is None:
            try:
                tensor_supply_type = TensorSupplyType.Auto
            except:
                tensor_supply_type = UtilsTensorSupplyType.Auto
        
        # 创建TPU专用的profiler
        profiler = Profiler(
            self.params,
            self.result_idx,
            tensor_supply_type
        ).with_default_adapter(self)
        
        # 重写supply函数，使其生成TPU设备张量而非CUDA张量
        original_supply = profiler.supply
        
        def tpu_supply(param: KernelParam) -> torch.Tensor:
            # 直接在TPU设备上创建张量，避免从CUDA转移
            # 确保导入torch_tpu以启用TPU设备支持
            import torch_tpu
            
            dtype: torch.dtype = param.dtype
            shape = tuple(int(s) for s in param.shape)
            device = "tpu:0"
            
            # 根据tensor_supply_type生成相应的张量
            if tensor_supply_type.name in ['Auto', 'Random', 'Integer']:
                return torch.randn(*shape, device=device).to(dtype)
            elif tensor_supply_type.name == 'Zero':
                return torch.zeros(*shape, device=device, dtype=dtype)
            elif tensor_supply_type.name == 'One':
                return torch.ones(*shape, device=device, dtype=dtype)
            else:
                # 默认使用随机张量
                return torch.randn(*shape, device=device).to(dtype)
            
        profiler.supply = tpu_supply
        
        # 保持原有的do_bench接口，让用户知道TPU限制但不破坏接口兼容性
        original_do_bench = profiler.do_bench
        
        def tpu_aware_do_bench(func=None, warmup=25, rep=100, n_warmup=1, n_repeat=1, input_tensors=None):
            """保持与CUDA完全相同的接口，内部处理TPU simulator限制"""
            print(f"注意: TPU simulator限制，建议使用较小参数避免崩溃")
            
            # 调用原始的do_bench，让用户自己控制参数
            return original_do_bench(func, warmup, rep, n_warmup, n_repeat, input_tensors)
        
        profiler.do_bench = tpu_aware_do_bench
        return profiler