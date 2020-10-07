'''
pySRM: Statistical Region Merging for python3 (C++ backend)

Note that "python setup.py test" invokes pytest on the package. With appropriately
configured setup.cfg, this will check both xxx_test modules and docstrings.

Copyright 2019, Markus Joppich.
Licensed under MIT.
'''
import sys
from setuptools import setup, find_packages
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


version = "1.0"

setup(name="pIMZ",
      version=version,
      description="pIMZ: an integrative framework for imaging mass spectrometry analysis",
      long_description=open("README.rst").read(),
      classifiers=[ # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
      ],
      keywords="IMS MSI imaging mass-spectrometry interactive integrative", # Separate with spaces
      author="Markus Joppich",
      author_email="pypi@mjoppich.net",
      url="https://github.com/mjoppich/pIMZ",
      license="MIT",
      packages=find_packages(exclude=['examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      tests_require=['pytest'],
      cmdclass={'test': PyTest},
      
      # TODO: List of packages that this one depends upon:   
      install_requires=['numpy', 'pandas', 'ctypes', 'globus_sdk', 'progressbar', 'anndata', 'diffxpy', 'scipy', 'dill', 'pathos', 'pyimzml', 'natsort', 'seaborn', 'matplotlib', 'dabest', 'imageio', 'Pillow'],
      # TODO: List executable scripts, provided by the package (this is just an example)
      entry_points={
        'console_scripts': 
            []
      }
)
