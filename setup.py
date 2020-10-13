'''
pySRM: Statistical Region Merging for python3 (C++ backend)

Note that "python setup.py test" invokes pytest on the package. With appropriately
configured setup.cfg, this will check both xxx_test modules and docstrings.

Copyright 2019, Markus Joppich.
Licensed under MIT.
'''
import sys, os
from setuptools import setup, find_packages, Extension
from setuptools.command.test import test as TestCommand

# This is a plug-in for setuptools that will invoke py.test
# when you run python setup.py test
class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest  # import here, because outside the required eggs aren't loaded yet
        sys.exit(pytest.main(self.test_args))

ext_lib_path = 'rectangle'
include_dir = os.path.join(ext_lib_path, 'include')

curBaseFolder = os.path.relpath(os.path.abspath(os.path.dirname(__file__)), os.getcwd())

print(curBaseFolder, file=sys.stderr)

segment_sources = [os.path.join(curBaseFolder, 'cIMZ/segment.cpp')]
imageregion_sources = [os.path.join(curBaseFolder, 'cIMZ/src/imageregion.cpp')]
srm_sources = [os.path.join(curBaseFolder, 'cIMZ/src/srm.cpp')]

print(segment_sources, file=sys.stderr)

compileArgs = ["-std=c++1z", "-Wall", "-fopenmp", "-fPIC", "-std=gnu++17",'-O3']

libPIMZ=Extension('pIMZ.libPIMZ',
                      sources=srm_sources+imageregion_sources+segment_sources,
                      language='c++',
                      libraries=['z', 'gomp'],
                      extra_compile_args=compileArgs,
                      extra_objects=[],
                      
                      )

version = "1.0a"
import pathlib


setup(name="pIMZ",
      version=version,
      description="pIMZ: an integrative framework for imaging mass spectrometry analysis",
      long_description=open("README.md").read(),
      long_description_content_type='text/markdown',
      classifiers=[ # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
      ],
      keywords="IMS MSI imaging mass-spectrometry interactive integrative", # Separate with spaces
      author="Markus Joppich",
      author_email="pIMZ@compbio.cc",
      url="https://github.com/mjoppich/pIMZ",
      license="MIT",
      packages=find_packages(exclude=['examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      tests_require=['pytest'],
      cmdclass={'test': PyTest},
      python_requires='>=3.8',
      # TODO: List of packages that this one depends upon:  
      install_requires=[
        'numpy>=1.17.5', 
        'h5py',
        'matplotlib',
        'pandas',
        'scipy', 
        'scikit-image',
        'scikit-learn', 
        'dill', 
        'pathos',
        'ms_peak_picker', 
        'globus_sdk', 
        'progressbar', 
        'anndata', 
        'diffxpy', 
        'pyimzml',
        'natsort',
        'seaborn',
        'llvmlite',
        'imageio', 'umap', 'jinja2', 'hdbscan', 'regex',
        'Pillow'], #dabest
      # TODO: List executable scripts, provided by the package (this is just an example)
      entry_points={
        'console_scripts': 
            []
      },
      ext_modules=[libPIMZ]
)
