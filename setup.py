from setuptools import setup

setup(
    name='tasksim',
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
        'ray',
        'ray[default]'
    ],
)