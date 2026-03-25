
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
    "sphinx_copybutton",
    "breathe",
]
 
templates_path = ["_templates"]
exclude_patterns = []
 
# Sphinx behavior
autodoc_member_order = "groupwise"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autosummary_generate = True
todo_include_todos = True
 
# -- Breathe / Doxygen configuration -----------------------------------------
 
# Use an absolute path to eliminate any working-directory ambiguity.
# This path must point to the directory that contains index.xml.
DOXYGEN_XML_DIR = "/home/jonwebb/Code_Dev/C++/csaltpp/docs/doxygen/build/xml"
 
breathe_default_project = "csaltpp"
breathe_projects = {
    "csaltpp": DOXYGEN_XML_DIR,
}
 
# Tell Breathe that .c/.cpp files are implementation files so it prefers
# header declarations when resolving directives.
breathe_implementation_filename_extensions = [".c", ".cc", ".cpp"]
 
# Map file extensions to the C++ domain
breathe_domain_by_extension = {
    "hpp": "cpp",
    "cpp": "cpp",
}
 
# Sanity check — printed during sphinx-build so you can confirm the path
import os
if not os.path.isfile(os.path.join(DOXYGEN_XML_DIR, "index.xml")):
    print(
        f"\n[conf.py] WARNING: index.xml not found in {DOXYGEN_XML_DIR}\n"
        "Run 'doxygen Doxyfile' from the docs/doxygen directory before "
        "building Sphinx.\n"
    )
else:
    print(f"[conf.py] Breathe XML path OK: {DOXYGEN_XML_DIR}")
 
# -- Options for HTML output -------------------------------------------------
 
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
}
 
