# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import ratspy

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RATSpy'
copyright = '2025, Felix Kern, Tejas Pradhan, Aryamaan Basu Roy'
author = 'Felix Kern, Tejas Pradhan, Aryamaan Basu Roy'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
   'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
   'numpydoc',
   'myst_parser',
   'sphinx_rtd_theme',
]

source_suffix = ['.rst', '.md']

numpydoc_class_members_toctree = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
