# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import re
import sys
sys.path.insert(0, os.path.abspath('../../src'))
# Mock imports for packages not available on RTD
autodoc_mock_imports = ['heyoka']

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Kyklos'
copyright = '2026, Shane Billingsley'
author = 'Shane Billingsley'
# Single source of truth: read __version__ from the package source without
# importing kyklos (the import would need Heyoka, which is mocked here).
_init_py = os.path.abspath('../../src/kyklos/__init__.py')
with open(_init_py, encoding='utf-8') as _f:
    release = re.search(r'^__version__\s*=\s*"([^"]+)"', _f.read(), re.M).group(1)
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',     # Support NumPy-style docstrings
    'sphinx.ext.autosummary',  # Generate summary tables
    'sphinx.ext.mathjax',      # Render math equations
    'sphinx.ext.viewcode',     # Add [source] links to code
    'myst_parser',             # Read Markdown
]

autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = []

# The README slices included in installation.rst and quickstart.rst start at
# H3 headings, which myst flags as a heading-level skip. The skip is intended.
suppress_warnings = ['myst.header']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
