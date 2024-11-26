import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

# 定义扩展模块的路径
_ext_src_root = "_ext_src"
this_dir = os.path.dirname(os.path.abspath(__file__))
_ext_include_dir = os.path.join(this_dir, _ext_src_root, "include")
_ext_sources = glob.glob(f"{_ext_src_root}/src/*.cpp") + glob.glob(f"{_ext_src_root}/src/*.cu")
_ext_headers = glob.glob(f"{_ext_include_dir}/*")

# 设置 CUDA 架构列表（RTX 4090: 8.9, RTX 4060 : 8.6）
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

setup(
    name='pointnet2',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            include_dirs=[_ext_include_dir],
            extra_compile_args={
                "cxx": [
                    "-O3",  # 高级优化
                    "-std=c++17",  # 使用现代 C++ 标准
                    f"-I{_ext_include_dir}",  # 显式包含头文件路径
                ],
                "nvcc": [
                    "-O3",  # 优化 CUDA 代码
                    "-DCUDA_HAS_FP16=1",  # 开启半精度支持
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                    f"-I{_ext_include_dir}",  # 显式包含头文件路径
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)},
)
