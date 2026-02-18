"""Sphinx configuration for MetalMom API reference."""

import os
import sys

# Add the Python package to sys.path
sys.path.insert(0, os.path.abspath("../../python"))

# -- Project information --
project = "MetalMom"
copyright = "2026, Zak Keown"
author = "Zak Keown"
release = "0.1.0"

# -- General configuration --
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = True

# Mock native imports that require the dylib (not available on ReadTheDocs)
autodoc_mock_imports = ["cffi", "metalmom._native", "metalmom._buffer"]

# Napoleon settings (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output --
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/zakkeown/MetalMom",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "header_links_before_dropdown": 6,
}
html_title = "MetalMom API Reference"
html_short_title = "MetalMom"

# Source settings
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]
