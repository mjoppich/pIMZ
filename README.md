# pIMZ: an integrative framework for imaging mass spectrometry analysis

[![Documentation Status](https://readthedocs.org/projects/pimz/badge/?version=latest)](https://pimz.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/mjoppich/pIMZ.svg?branch=master)](https://travis-ci.org/mjoppich/pIMZ)
![PyPI](https://img.shields.io/pypi/v/pIMZ)
![GitHub All Releases](https://img.shields.io/github/downloads/mjoppich/pIMZ/total)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pIMZ)
![PyPI - License](https://img.shields.io/pypi/l/pIMZ)
[![DOI](https://zenodo.org/badge/203115135.svg)](https://zenodo.org/badge/latestdoi/203115135)


## Description

pIMZ is a framework for Imaging Mass Spectrometry (IMS) data analysis.
The python implementation is available from  [github.com/mjoppich/pIMZ](http://github.com/mjoppich/pIMZ).
For installation follow the instructions given here.

pIMZ focuses on a differential setting, where masses, specific to certain areas are first determined, and then serve as input for a cell-type detection framework and/or a differential expression setting.

pIMZ's documentation is available here: [![Documentation Status](https://readthedocs.org/projects/pimz/badge/?version=latest)](https://pimz.readthedocs.io/en/latest/?badge=latest) .


## Installation

The easiest way to install most Python packages is via ``pip``.

If not already done, you must first install the following dependencies manually. This is because at the time of writing this document, dabest requires ``pandas~=0.25``, which is incompatible with probably the rest of the world nowadays. So first dabest is installed, then numpy and pandas are upgraded again ::

    sudo pip3 install dabest
    sudo pip3 install numpy pandas --upgrade

Only then we should install ``pIMZ`` ::
    sudo pip3 install pIMZ

``pIMZ`` is now ready to go!

## Usage

References to examples and example notebooks can be found in the ``examples`` folder, or in the [documentation](https://pimz.readthedocs.io/en/latest/usage.html).

All available classes and their functions are explained in the [modules/API documentation](https://pimz.readthedocs.io/en/latest/modules.html).
