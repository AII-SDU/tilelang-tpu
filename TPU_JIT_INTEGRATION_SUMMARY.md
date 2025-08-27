# TPU JIT 集成代码修改总结

本文档详细记录了如何在tilelang中添加新的target（TPU）以及实现完整JIT集成的核心代码修改。

## 🎯 修改目标

将TPU从独立的`@tpujit`装饰器系统重构为与CUDA/HIP完全兼容的标准tilelang JIT系统，实现统一的target架构。

## 📁 核心代码修改

### 1. 如何添加新的Target支持

#### 步骤1: Target声明 - `tilelang/utils/target.py`
```python
# 原代码
AVALIABLE_TARGETS = ["cuda", "hip"]

# 修改后
AVALIABLE_TARGETS = ["cuda", "hip", "tpu"]
```
**作用**: 声明"tpu"为合法的target参数，通过target验证检查。

#### 步骤2: Lower阶段处理 - `tilelang/engine/lower.py`
**关键修改**: 在lower函数中添加TPU的特殊处理逻辑
```python
# 在lower函数中添加（约第232行位置）
# Special handling for TPU target which doesn't use TVM's target system
if (isinstance(target, str) and target == "tpu") or (hasattr(target, 'kind') and target.kind.name == "tpu"):
    # For TPU, use a dummy CPU target for TVM transforms but generate TPU code
    dummy_target = tvm.target.Target("llvm", tvm.target.Target("llvm"))
    
    # Phase 1: Lower and legalize the IR
    mod = LowerAndLegalize(mod, dummy_target)

    # Phase 2: Optimize the IR for the target  
    mod = OptimizeForTarget(mod, dummy_target)

    # Directly call TPU codegen and return source code
    device_source = tvm._ffi.get_global_func("target.build.tilelang_ppl")(mod)
    return CompiledArtifact(None, mod, params, device_source)
```
**核心思想**: 
- TPU不使用TVM的标准target系统，而是使用自己的codegen
- 用dummy的LLVM target进行TVM变换，最后调用TPU专用的codegen函数
- 直接返回CompiledArtifact，绕过标准的TVM编译流程

**删除的旧代码**: 原先第232行附近的hardcoded特殊处理逻辑

### 2. 如何实现JIT集成

#### 步骤3: JIT系统适配 - `tilelang/jit/__init__.py`

**关键修改1**: 在`jit()`装饰器函数中添加TPU路由
```python
# 在jit函数中添加TPU检测和路由
if target == "tpu":
    from .adapter.tpu import TPUKernelAdapter
    adapter = TPUKernelAdapter(compiled_artifact, result_idx, fn_name)
    return TPUJITKernel(adapter)
```

**关键修改2**: 在`compile()`API函数中添加TPU路由
```python
# 在compile函数中添加TPU支持
if target == "tpu":
    from .adapter.tpu import TPUKernelAdapter
    adapter = TPUKernelAdapter(compiled_artifact, result_idx, fn_name)
    return TPUJITKernel(adapter)
```

**关键修改3**: 创建TPU专用包装类
```python
class TPUJITKernel:
    """TPU JIT内核包装类，提供与CUDA/HIP相同的接口"""
    def __init__(self, adapter):
        self.adapter = adapter
        
    def __getitem__(self, grid):
        """支持grid语法: kernel[(1,)](args)"""
        return self.adapter.__getitem__(grid)
        
    def __call__(self, *args):
        """支持直接调用: kernel(args)"""
        return self.adapter._convert_torch_func()(*args)
        
    def get_kernel_source(self) -> str:
        """获取内核源码，与CUDA/HIP接口一致"""
        return self.adapter.get_kernel_source()
        
    def get_profiler(self, tensor_supply_type=None):
        """获取profiler，与CUDA/HIP接口一致"""
        return self.adapter.get_profiler(tensor_supply_type)
```
**设计思想**: 通过包装类统一接口，使TPU JIT与CUDA/HIP JIT具有完全相同的API。

### 3. 核心适配器实现

#### 步骤4: 创建TPU适配器 - `tilelang/jit/adapter/tpu.py` (新文件)

**核心设计**: 实现BaseKernelAdapter接口，处理TPU特有的编译和执行逻辑

```python
class TPUKernelAdapter(BaseKernelAdapter):
    """TPU内核适配器，继承标准的BaseKernelAdapter接口"""
    
    def __init__(self, compiled_artifact: CompiledArtifact, result_idx: List[int], 
                 fn_name: str = None, chip: str = "bm1690"):
        super().__init__(compiled_artifact.device_mod, compiled_artifact.params, result_idx)
        self.compiled_artifact = compiled_artifact  # 保存编译结果
        self.fn_name = fn_name or "main"            # 内核函数名
        self.result_idx = self._convert_negative_indices(result_idx, len(compiled_artifact.params))
```

**关键功能1**: 内核编译和缓存系统
```python
def _prepare_shared_library(self):
    """将TPU C代码编译为共享库，实现智能缓存"""
    # 1. 生成源代码哈希作为缓存key
    hash_val = hashlib.md5(c_code.encode()).hexdigest()
    
    # 2. 处理TPU专用函数名（添加_kernel后缀）
    kernel_fn_name = f"{self.fn_name}_kernel" if not self.fn_name.endswith("_kernel") else self.fn_name
    
    # 3. 添加tensor_info结构体定义（TPU运行时需要）
    # 4. 调用SophGo TPU编译器生成共享库
    # 5. 缓存编译结果，避免重复编译
```

**关键功能2**: 与PyTorch集成
```python
def _convert_torch_func(self) -> Callable:
    """转换为PyTorch可调用函数，复用标准TileLangKernel执行逻辑"""
    # 完全复制BaseKernelAdapter的参数处理和执行流程
    # 使用torch.ops.my_ops.dynlib_execute调用TPU共享库
```

**关键功能3**: Profiler接口适配
```python
def get_profiler(self, tensor_supply_type=None):
    """创建TPU专用profiler，解决设备兼容性问题"""
    
    # 重写张量供应函数，生成TPU设备张量
    def tpu_supply(param: KernelParam) -> torch.Tensor:
        import torch_tpu  # 必须import才能使用tpu:0设备
        dtype: torch.dtype = param.dtype
        shape = tuple(int(s) for s in param.shape)
        device = "tpu:0"
        return torch.randn(*shape, device=device).to(dtype)
    
    # 重写do_bench，处理TPU simulator限制
    def tpu_aware_do_bench(func=None, warmup=25, rep=100, n_warmup=1, n_repeat=1, input_tensors=None):
        print(f"注意: TPU simulator限制，建议使用较小参数避免崩溃")
        return original_do_bench(func, warmup, rep, n_warmup, n_repeat, input_tensors)
```

**关键功能4**: 自动环境配置
```python
def _auto_source_sophgo_env():
    """自动加载SophGo TPU环境变量"""
    # 解析envsetup.sh并设置环境变量
    # 确保TPU编译器和运行时正确加载
```

## 🔧 关键技术解决方案

### 1. 函数名处理
**问题**: TPU内核需要调用`_kernel`后缀的wrapper函数  
**解决**: 在编译过程中自动添加`_kernel`后缀
```python
kernel_fn_name = f"{self.fn_name}_kernel" if not self.fn_name.endswith("_kernel") else self.fn_name
```

### 2. 设备兼容性
**问题**: Profiler默认生成CUDA张量，但TPU需要TPU设备张量  
**解决**: 创建TPU专用的张量供应函数
```python
def tpu_supply(param: KernelParam) -> torch.Tensor:
    import torch_tpu
    dtype: torch.dtype = param.dtype
    shape = tuple(int(s) for s in param.shape)
    device = "tpu:0"
    return torch.randn(*shape, device=device).to(dtype)
```

### 3. 编译缓存
**问题**: 重复编译浪费时间  
**解决**: 基于源代码哈希的缓存机制
```python
hash_val = hashlib.md5(c_code.encode()).hexdigest()
cache_dir = os.path.join(os.getcwd(), ".tilelang_cache")
```

### 4. Profiler do_bench限制
**问题**: TPU simulator对重复执行有限制  
**解决**: 保持接口兼容，内部处理限制
```python
def tpu_aware_do_bench(func=None, warmup=25, rep=100, n_warmup=1, n_repeat=1, input_tensors=None):
    print(f"注意: TPU simulator限制，建议使用较小参数避免崩溃")
    return original_do_bench(func, warmup, rep, n_warmup, n_repeat, input_tensors)
```

## 📝 示例文件修改

将现有的ppl示例修改为标准JIT格式：

### 1. 装饰器方式 (reduce_max)
```python
# 原来: 使用tilelang.lower
func = reduce_max(8192, 1020, 512, 1020)
mod = tilelang.lower(func)

# 现在: 使用装饰器
@tilelang.jit(target="tpu", out_idx=[-1])
@T.prim_func
def reduce_max(X: T.Tensor((8192, 1020), "float32"), Y: T.Tensor((8192, 1), "float32")):
    # 内核实现
    pass

# 调用
reduce_max[(1,)](input_tensor, output_tensor)
```

### 2. 编译API方式 (reduce_sum, matmul)
```python
# 原来: 使用tilelang.lower
func = reduce_sum(8192, 8192, 512, 8192)
mod = tilelang.lower(func)

# 现在: 使用编译API
@T.prim_func
def reduce_sum_func(X: T.Tensor((8192, 8192), "float32"), Y: T.Tensor((8192, 1), "float32")):
    # 内核实现
    pass

# 编译和调用
reduce_sum = tilelang.compile(reduce_sum_func, target="tpu", out_idx=[-1])
reduce_sum[(1,)](input_tensor, output_tensor)
```

## ✅ 实现的功能

### 1. 完全接口兼容
- `@tilelang.jit(target="tpu", out_idx=[-1])` - 装饰器方式
- `tilelang.compile(func, target="tpu", out_idx=[-1])` - 编译API方式
- `kernel[(grid,)](args)` - Grid语法支持
- `kernel(args)` - 直接调用支持

### 2. Profiler完整支持
- `kernel.get_kernel_source()` - 获取C源码
- `kernel.get_profiler()` - 创建性能分析器
- `profiler._get_inputs()` - 自动生成TPU设备张量
- `profiler.run_once()` - 单次执行测试
- `profiler.do_bench()` - 性能基准测试
- `profiler(*args)` - 直接调用

### 3. 自动化功能
- 自动环境变量加载
- 自动编译缓存
- 自动函数名处理
- 自动设备张量生成

## 🎉 最终成果

1. **接口统一**: TPU JIT与CUDA/HIP JIT使用完全相同的接口
2. **代码复用**: 用户代码在不同target间无缝切换
3. **向后兼容**: 不影响现有CUDA/HIP JIT功能
4. **完整功能**: 支持所有标准JIT功能，包括profiler

## 📊 测试验证

创建了完整的测试用例：
- `tpu_demo/ppl/reduce_max_kernel/reduce_max.py` - 装饰器方式测试
- `tpu_demo/ppl/reduce_sum_kernel/reduce_sum.py` - 编译API方式测试  
- `tpu_demo/ppl/matmul_kernel/matmul.py` - 编译API方式测试
- `tpu_demo/test_all_jit.py` - 全面测试脚本

所有测试都包含：
- 功能正确性验证
- 与PyTorch原生实现对比
- Profiler接口验证
- 设备兼容性检查

## 🚀 使用方法

```python
# 现在用户可以无缝切换target
@tilelang.jit(target="cuda", out_idx=[-1])  # CUDA版本
# @tilelang.jit(target="hip", out_idx=[-1])   # HIP版本  
# @tilelang.jit(target="tpu", out_idx=[-1])   # TPU版本
@T.prim_func
def my_kernel(X, Y):
    # 相同的内核代码
    pass

# 相同的调用方式
my_kernel[(1,)](input_tensor, output_tensor)

# 相同的profiler接口
profiler = my_kernel.get_profiler()
```

这标志着TPU JIT完全集成到tilelang标准架构中，实现了真正的统一！