
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
#import os
#import sys
#sys.path.insert(0, os.path.abspath('../../../src'))

# -- Project information -----------------------------------------------------

# Configuration file for the Sphinx documentation builder

# -- Project information -----------------------------------------------------
project = 'csalt++'
copyright = '2025, Jonathan A. Webb'
author = 'Jonathan A. Webb'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "breathe",
]

# Auto-documentation settings
autodoc_member_order = "groupwise"
autosummary_generate = True

# Breathe configuration
breathe_projects = {
    "csalt++": "../xml"  # Adjust this path if your XML ends up elsewhere
}
breathe_default_project = "csalt++"
breathe_domain_by_extension = {
    "h": "cpp",
    "hpp": "cpp",
    "cpp": "cpp"
}

# Optional: if using breathe_projects_source for automatic matching
# breathe_projects_source = {
#     "csalt++": ("../include", ["matrix.hpp"])
# }

# Source file parsers and default language
highlight_language = "c++"
primary_domain = "cpp"

# Templates and exclusions
templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- ToDo extension ----------------------------------------------------------
todo_include_todos = True

