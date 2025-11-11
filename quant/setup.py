from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = "32"
    common_nvcc_args = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=" + nvcc_threads,
        "-gencode=arch=compute_80,code=sm_80"
    ]
    return nvcc_extra_args + common_nvcc_args

ext_modules = [
    CUDAExtension(
        name='cuda_my_quant',
        sources=['csrc/my_quant_kernel.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='cuda_my_score',
        sources=['csrc/my_score_kernel.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='cuda_my_score_half',
        sources=['csrc/my_score_kernel_half.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='cuda_my_key_value_quant_V2',
        sources=['csrc/my_key_value_quant_kernel_V2.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    )
    ,
    CUDAExtension(
        name='cuda_my_key_value_dequant_V2',
        sources=['csrc/my_key_value_dequant_kernel_V2.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='cuda_my_key_value_dequant_select',
        sources=['csrc/my_key_value_dequant_kernel_select.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='cuda_my_key_value_dequant_select_fuse',
        sources=['csrc/my_key_value_dequant_kernel_select_fuse.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='cuda_my_key_value_dequant_V3',
        sources=['csrc/my_key_value_dequant_kernel_V3.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='kvquantv3',
        sources=['csrc/my_key_value_quant_kernel_V3.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='get_lut',
        sources=['csrc/my_get_query_lut.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='lut_gemv',
        sources=['csrc/lut_gemv.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    )
]

setup(
    name='combined_cuda_extensions',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=["torch"]
)
