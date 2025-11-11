from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='cpu_lut_gemv',
    ext_modules=[
        CppExtension(
            'cpu_lut_gemv',
            ['csrc/cpu_lut_gemv.cpp'],
            extra_compile_args={
            "cxx": ["-g", "-O3","-mavx2","-fopenmp"],
        }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
