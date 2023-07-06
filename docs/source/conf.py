# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Lab4D"
copyright = (
    "2023, Gengshan Yang, Jeff Tan, Alex Lyons, Neehar Peri, Carnegie Mellon University"
)
release = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import sys, os

# Path to lab4d
sys.path.insert(
    0,
    "%s/../../" % os.path.join(os.path.dirname(__file__)),
)

# Allow auto-generated docs from Google format docstrings
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

# other pakcages
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pytorch": ("https://pytorch.org/docs/stable/", None),
}

# Allow documentation of multiple return types
napoleon_custom_sections = [("Returns", "params_style")]

templates_path = ["_templates"]
exclude_patterns = []

# Mocking the imports of modules that requires cuda
autodoc_mock_imports = ["_quaternion"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pytorch_sphinx_theme"
html_theme_path = ["../pytorch_sphinx_theme"]
html_static_path = ["_static"]
