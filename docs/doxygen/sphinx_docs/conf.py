
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

# -- Project info ------------------------------------------------------------
project = 'csalt++-lib'
copyright = '2025, Jonathan A. Webb'
author = 'Jonathan A. Webb'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "breathe",
]

# Sphinx: replace deprecated autodoc_default_flags with autodoc_default_options
autodoc_member_order = "groupwise"
autodoc_default_options = {"members": True, "show-inheritance": True}
autosummary_generate = True

# Breathe configuration
from pathlib import Path
HERE = Path(__file__).parent.resolve()
DOXYGEN_XML_DIR = HERE.parent / "build" / "xml"   # docs/_build/doxygen/xml

breathe_default_project = "csalt++"
breathe_domain_by_extension = {"hpp": "cpp", "cpp": "cpp", "h": "hpp"}
breathe_projects = {"csalt++": str(DOXYGEN_XML_DIR)}

# Optional sanity check: prints a hint during Sphinx build if XML is missing
if not DOXYGEN_XML_DIR.exists():
    print(f"[conf.py] WARNING: Doxygen XML not found at {DOXYGEN_XML_DIR}. "
          f"Run 'doxygen Doxyfile' before building Sphinx.")
    
templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"

