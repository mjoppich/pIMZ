# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'pIMZ'
copyright = '2020, Markus Joppich, Margaryta Olenchuk, Ralf Zimmer'
author = 'Markus Joppich, Margaryta Olenchuk, Ralf Zimmer'

master_doc = 'index'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False


autodoc_mock_imports = [
        'numpy', 
        'h5py',
        'matplotlib',
        'pandas',
        'scipy', 
        'scikit-image',
        'scikit-learn', 
        'sklearn', 'skimage',
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
        'Pillow']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']