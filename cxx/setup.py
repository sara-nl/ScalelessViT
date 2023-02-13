from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='crop_interpolate',
    ext_modules=[
        CUDAExtension('crop_interpolate', [
            'kernels.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
