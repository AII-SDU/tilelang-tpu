import os
import hashlib
import inspect
import subprocess
import shutil
import torch
from typing import Callable, Optional, TypeVar, List, Tuple

T = TypeVar('T')

def compile_to_so(src_path: str, so_path: str, chip: str = "bm1690") -> bool:
    """
    Compile C source file to .so shared library for TPU.
    """
    os.makedirs(os.path.dirname(so_path), exist_ok=True)
    os.environ["CHIP_ARCH"] = chip
    os.environ["CHIP"] = chip
    sophgo_tpu_root = os.getenv("SOPHGO_TPU_ROOT")
    if not sophgo_tpu_root:
        print("尝试加载3rd-party 中sophgo_tpu环境")
        sophgo_tpu_root = "../../3rd-party/sophgo_tpu"
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

class TileLangKernel:
    """
    TileLang 内核包装类，TPU JIT接口
    """
    def __init__(self, fn_name: str, so_path: str):
        self.fn_name = fn_name
        self.so_path = so_path
    def __getitem__(self, grid: Tuple[int, ...]):
        def runner(*args):
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
            print(f"SO 路径: {self.so_path}")
            print(f"张量参数: {[(t.shape, t.dtype, t.device) for t in tensors]}")
            print(f"张量索引: {tensors_index}")
            print(f"浮点标量: {fp_scalars}")
            print(f"浮点标量索引: {fp_scalars_index}")
            print(f"固定标量: {fixed_scalars}")
            print(f"固定标量索引: {fixed_scalars_index}")
            torch.ops.my_ops.dynlib_execute(
                self.so_path, 
                self.fn_name, 
                tensors, 
                tensors_index,
                fp_scalars, 
                fp_scalars_index, 
                fixed_scalars,
                fixed_scalars_index
            )
        return runner

def tpujit(fn=None, *, cache_dir=None, chip="bm1690"):
    """
    TileLang TPU JIT 装饰器
    """
    _cache_dir = cache_dir
    _chip = chip
    def decorator(fn):
        fn_name = fn.__name__
        fn_source = inspect.getsource(fn)
        hash_val = hashlib.md5(fn_source.encode()).hexdigest()
        local_cache_dir = _cache_dir
        if local_cache_dir is None:
            local_cache_dir = os.path.join(os.getcwd(), ".tilelang_cache")
        os.makedirs(local_cache_dir, exist_ok=True)
        c_path = os.path.join(local_cache_dir, f"{fn_name}_{hash_val}.c")
        so_path = os.path.join(local_cache_dir, f"{fn_name}_{hash_val}.so")
        if os.path.exists(so_path):
            print(f"使用缓存的内核: {so_path}")
            return TileLangKernel(fn_name, so_path)
        import tilelang
        tilelang_func = fn()
        c_code = tilelang.lower(tilelang_func)
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
        with open(c_path, "w") as f:
            f.write(c_code)
        print(f"生成的 C 代码已保存到: {c_path}")
        if not compile_to_so(c_path, so_path, _chip):
            raise RuntimeError(f"编译 {c_path} 失败")
        print(f"编译成功: {so_path}")
        return TileLangKernel(fn_name, so_path)
    if fn is not None:
        return decorator(fn)
    return decorator
