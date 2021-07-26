from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy as np
import os
import ot
import ot.lp

import sys
import subprocess

compile_args = ['-O3']
if sys.platform.startswith('darwin'):
    compile_args.append('-stdlib=libc++')
    sdk_path = subprocess.check_output(['xcrun', '--show-sdk-path'])
    os.environ['CFLAGS'] = '-isysroot "{}"'.format(sdk_path.rstrip().decode('utf-8'))

ot_lp_root = os.path.dirname(ot.lp.__file__)

setup(
    name='tasksim',
    ext_modules=cythonize(
        Extension(
            name='process_chunk',
            sources=['tasksim/process_chunk.pyx', os.path.join(ot_lp_root, 'EMD_wrapper.cpp')],
            language="c++",
            include_dirs=[np.get_include(), ot_lp_root],
            extra_compile_args=compile_args
        )
    ),
    version='0.0.1',
    packages=['tasksim'],
    install_requires=[
        'scipy',
        'POT',
        'pandas',
        'pillow',
        'matplotlib',
        'numpy',
        'Cython',
        'seaborn',
        'sklearn',
        'sympy',
        'pyinstrument',
        'numba',
        'ray',
        'ray[default]',
        'gym',
        'torch',
        'ray[rllib]'
    ],
    zip_safe=False,
)