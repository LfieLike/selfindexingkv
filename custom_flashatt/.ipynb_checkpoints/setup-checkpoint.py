from torch.utils.cpp_extension import load
import os

os.environ['TORCH_USE_CUDA_DSA'] = "1"
# suported GPU architectures
os.environ['TORCH_CUDA_ARCH_LIST'] = "8.0 8.6 8.7 8.9 9.0"
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

# ext_modules = [
#     CUDAExtension(
#         name='cuda_my_quant',
#         sources=['csrc/my_quant_kernel.cu'],
#         extra_compile_args={
#             "cxx": ["-g", "-O3"],
#             "nvcc": append_nvcc_threads([])
#         }
#     ),
# simple way to compile CUDA kernels and bind them as a pyTorch extension
flash_attention_rami_extension = load(
    name='custom_kernels_extension',
    sources=["custom_kernels.cu", "main.cpp"],
    with_cuda=True,
    extra_cuda_cflags=[""],
    build_directory='.',
    is_python_module=False,
)
