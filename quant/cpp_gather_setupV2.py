from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='cpu_gather_all',
    ext_modules=[
        CppExtension(
            'cpu_gather_all',
            ['csrc/cpu_gather_all.cpp'],
            extra_compile_args={
            "cxx": ["-g", "-O3","-mavx2","-fopenmp"],
        }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
