#python3 setup.py install
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

setup(
    name='pointops',
    author='Hengshuang Zhao',
    ext_modules=[
        CUDAExtension('pointops_cuda', [
            'src/pointops_api.cpp',
            'src/knnquery/knnquery_cuda.cpp',
            'src/knnquery/knnquery_cuda_kernel.cu',
            'src/sampling/sampling_cuda.cpp',
            'src/sampling/sampling_cuda_kernel.cu',
            'src/grouping/grouping_cuda.cpp',
            'src/grouping/grouping_cuda_kernel.cu',
            'src/interpolation/interpolation_cuda.cpp',
            'src/interpolation/interpolation_cuda_kernel.cu',
            'src/subtraction/subtraction_cuda.cpp',
            'src/subtraction/subtraction_cuda_kernel.cu',
            'src/aggregation/aggregation_cuda.cpp',
            'src/aggregation/aggregation_cuda_kernel.cu',
            ],
            include_dirs=[
                '/usr/local/cuda-11.6/include',
                '/home/ubunone/anaconda3/envs/fenge/lib/python3.8/site-packages/torch/include',
                '/home/ubunone/anaconda3/envs/fenge/lib/python3.8/site-packages/torch/include/torch/csrc/api/include',
                '/home/ubunone/anaconda3/envs/fenge/lib/python3.8/site-packages/torch/include/TH',
                '/home/ubunone/anaconda3/envs/fenge/lib/python3.8/site-packages/torch/include/THC'
            ],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
