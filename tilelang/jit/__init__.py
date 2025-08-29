# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
This module provides an auto-tuning infrastructure for TileLang (tl) programs. 
It includes functionality to JIT-compile TileLang programs into a runnable 
kernel adapter using TVM.
"""

from typing import Callable, List, Literal, Union, Any, Optional, Dict

from tilelang import tvm as tvm
from tvm.tir import PrimFunc
from tvm.target import Target

from tilelang.jit.adapter import BaseKernelAdapter
from tilelang.jit.kernel import JITKernel
from tilelang.utils.target import determine_target, AVALIABLE_TARGETS
from tilelang.cache import cached
from logging import getLogger
import tilelang

logger = getLogger(__name__)


def jit(
    func: Callable = None,
    *,  # Enforce keyword-only arguments from here on
    out_idx: Union[List[int], int] = None,
    execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
    target: Union[str, Target] = "auto",
    verbose: bool = False,
    **pass_config_kwargs: Optional[Dict[str, Any]],
) -> BaseKernelAdapter:
    """
    A decorator (or decorator factory) that JIT-compiles a given TileLang PrimFunc 
    into a runnable kernel adapter using TVM. If called with arguments, it returns 
    a decorator that can be applied to a function. If called without arguments, 
    it directly compiles the given function.

    Parameters
    ----------
    func : Callable, optional
        The TileLang PrimFunc to JIT-compile. If None, this function returns a 
        decorator that expects a TileLang PrimFunc.
    out_idx : Union[List[int], int], optional
        The index (or list of indices) of the function outputs. This can be used
        to specify which outputs from the compiled function will be returned.
    execution_backend : Literal["dlpack", "ctypes"], optional
        The wrapper type to use for the kernel adapter. Currently, only "dlpack"
        and "ctypes" are supported.
    target : Union[str, Target], optional
        The compilation target for TVM. If set to "auto", an appropriate target
        will be inferred automatically. Otherwise, must be one of the supported
        strings in AVALIABLE_TARGETS or a TVM Target instance.

    Returns
    -------
    BaseKernelAdapter
        An adapter object that encapsulates the compiled function and can be
        used to execute it.

    Raises
    ------
    AssertionError
        If the provided target is an invalid string not present in AVALIABLE_TARGETS.
    """

    # If the target is specified as a string, ensure it is valid and convert to a TVM Target.
    if isinstance(target, str):
        assert target in AVALIABLE_TARGETS, f"Invalid target: {target}"
        # Special handling for TPU - don't convert to TVM Target
        if target not in ["tpu", "rvv"]:
            target = determine_target(target)
            target = Target(target)

    assert execution_backend in ["dlpack", "ctypes", "cython"], "Invalid execution backend."

    def _compile_and_create_adapter(tilelang_func: PrimFunc) -> BaseKernelAdapter:
        """
        Compile the given TileLang PrimFunc with TVM and build a kernel adapter.

        Parameters
        ----------
        tilelang_func : tvm.tir.PrimFunc
            The TileLang (TVM TIR) function to compile.

        Returns
        -------
        BaseKernelAdapter
            The compiled and ready-to-run kernel adapter.
        """
        if verbose:
            logger.info(f"Compiling TileLang function:\n{tilelang_func}")
            
        # Special handling for TPU target
        if (isinstance(target, str) and target == "tpu") or (hasattr(target, 'kind') and hasattr(target.kind, 'name') and target.kind.name == "tpu"):
            # Compile using TPU backend
            compiled_artifact = tilelang.lower(tilelang_func, target="tpu")
            # Extract function name from tilelang_func
            fn_name = getattr(tilelang_func, 'attrs', {}).get('global_symbol', None)
            if fn_name is None:
                fn_name = getattr(tilelang_func, 'name_hint', "main")
            # Create TPU adapter directly
            from tilelang.jit.adapter.tpu import TPUKernelAdapter
            return TPUKernelAdapter(compiled_artifact, out_idx or [], fn_name=fn_name)
        
        elif (isinstance(target, str) and target == "rvv") or (hasattr(target, 'kind') and hasattr(target.kind, 'name') and target.kind.name == "rvv"):
            # 使用 RVV 后端编译
            compiled_artifact = tilelang.lower(tilelang_func, target="rvv")
            # 从 tilelang_func 提取函数名
            fn_name = getattr(tilelang_func, 'attrs', {}).get('global_symbol', None)
            if fn_name is None:
                fn_name = getattr(tilelang_func, 'name_hint', "main")
            # 创建 RVV 适配器
            from tilelang.jit.adapter.rvv import RVVKernelAdapter
            return RVVKernelAdapter(compiled_artifact, out_idx or [], fn_name=fn_name)
        
        return compile(
            tilelang_func,
            target=target,
            verbose=verbose,
            execution_backend=execution_backend,
            out_idx=out_idx,
            **pass_config_kwargs,
        ).adapter

    # If `func` was given, compile it immediately and return the adapter.
    if func is not None:
        return _compile_and_create_adapter(func)

    # Otherwise, return a decorator that expects a function to compile.
    def real_decorator(tilelang_func: PrimFunc) -> BaseKernelAdapter:
        return _compile_and_create_adapter(tilelang_func)

    return real_decorator


def compile(
    func: PrimFunc = None,
    out_idx: Union[List[int], int, None] = None,
    execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
    target: Union[str, Target] = "auto",
    target_host: Union[str, Target] = None,
    verbose: bool = False,
    pass_configs: Optional[Dict[str, Any]] = None,
) -> JITKernel:
    """
    Compile the given TileLang PrimFunc with TVM and build a JITKernel.
    """
    # Special handling for TPU target
    if (isinstance(target, str) and target == "tpu"):
        # For TPU, create a simple JITKernel-like wrapper
        compiled_artifact = tilelang.lower(func, target="tpu")
        # Extract function name
        fn_name = getattr(func, 'attrs', {}).get('global_symbol', None)
        if fn_name is None:
            fn_name = getattr(func, 'name_hint', "main")
        from tilelang.jit.adapter.tpu import TPUKernelAdapter
        adapter = TPUKernelAdapter(compiled_artifact, out_idx or [], fn_name=fn_name)
        
        class TPUJITKernel:
            def __init__(self, adapter):
                self.adapter = adapter
                
            def __getitem__(self, grid):
                """支持grid语法调用"""
                return self.adapter.__getitem__(grid)
                
            def __call__(self, *args):
                """支持直接调用"""
                return self.adapter._convert_torch_func()(*args)
                
            def get_kernel_source(self) -> str:
                """获取生成的C内核源码"""
                return self.adapter.get_kernel_source()
                
            def get_profiler(self, tensor_supply_type=None):
                """获取性能分析器，为TPU设备定制"""
                # 直接调用adapter的get_profiler方法，该方法已经处理了TPU设备问题
                return self.adapter.get_profiler(tensor_supply_type)
        
        return TPUJITKernel(adapter)
    
    elif (isinstance(target, str) and target == "rvv"):
    # 对于 RVV，创建一个简单的类似 JITKernel 的包装器
        compiled_artifact = tilelang.lower(func, target="rvv")
        # 提取函数名
        fn_name = getattr(func, 'attrs', {}).get('global_symbol', None)
        if fn_name is None:
            fn_name = getattr(func, 'name_hint', "main")
        from tilelang.jit.adapter.rvv import RVVKernelAdapter
        adapter = RVVKernelAdapter(compiled_artifact, out_idx or [], fn_name=fn_name)
        
        class RVVJITKernel:
            def __init__(self, adapter):
                self.adapter = adapter
                
            def __getitem__(self, grid):
                """支持网格语法调用"""
                return self.adapter.__getitem__(grid)
                
            def __call__(self, *args):
                """支持直接调用"""
                return self.adapter._convert_torch_func()(*args)
                
            def get_kernel_source(self) -> str:
                """获取生成的 RVV 内核源码"""
                return self.adapter.get_kernel_source()
        
        return RVVJITKernel(adapter)
    
    return cached(
        func=func,
        out_idx=out_idx,
        execution_backend=execution_backend,
        target=target,
        target_host=target_host,
        verbose=verbose,
        pass_configs=pass_configs,
    )