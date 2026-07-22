# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'X-Ray Image Analysis'
copyright = '2025, Tom Pearson'
author = 'Tom Pearson'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Adjust to point to your source code

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',     # For Google/Numpy style docstrings
    'sphinx.ext.viewcode',
    'autoapi.extension',
]

autoapi_type = 'python'
autoapi_dirs = ['../../']  # path to your Python package

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
