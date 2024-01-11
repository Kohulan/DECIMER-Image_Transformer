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

sys.path.insert(0, os.path.abspath(".."))
import DECIMER
from datetime import datetime

# -- Project information -----------------------------------------------------

project = "DECIMER-Image_Transformer"
version = DECIMER.__version__
current_year = datetime.today().year
copyright = "2019-{}, Kohulan Rajan at the Friedrich Schiller University Jena".format(
    current_year
)
author = "Kohulan Rajan"
rst_prolog = """
.. |current_year| replace:: {}
""".format(
    current_year
)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Decides the language used for syntax highlighting of code blocks.
highlight_language = "python3"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#E37B74",
        "color-brand-content": "#E37B74",
        "color-code-background": "#F8F8F8",
        "color-code-border": "#E37B74",
        "color-admonition-background": "#FEECEC",
        "color-link": "#E37B74",
        "color-pre-background": "#F8F8F8",
        "color-pre-border": "#E37B74",
    },
    "dark_css_variables": {
        "color-brand-primary": "#E37B74",
        "color-brand-content": "#E37B74",
        "color-code-background": "#222222",
        "color-code-border": "#E37B74",
        "color-admonition-background": "#331E1C",
        "color-link": "#E37B74",
        "color-pre-background": "#222222",
        "color-pre-border": "#E37B74",
    },
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_favicon = "_static/DECIMER.gif"
html_logo = "_static/DECIMER.gif"
# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
