import pathlib
from setuptools import setup, find_packages

# Get the directory of this file
HERE = pathlib.Path(__file__).parent

# Install and update wheel for packages that depend on it
# import pip
# pip.main(['install', '--upgrade', 'wheel'])

setup(
    name='curriculum_tools',
    version='0.1.0',
    description='Curriculum Tools',
    long_description=(HERE / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    author='Edward Staley',
    author_email='edward.staley@jhuapl.edu',
    license='UNLICENSED',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3'
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=find_packages(),
    install_requires=[
        'gym'
    ]
)
