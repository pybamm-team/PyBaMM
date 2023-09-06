# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import pybamm

# Path for repository root
sys.path.insert(0, os.path.abspath("../"))

# Path for local Sphinx extensions
sys.path.append(os.path.abspath("./sphinxext/"))


# -- Project information -----------------------------------------------------

project = "PyBaMM"
copyright = "2018-2023, The PyBaMM Team"
author = "The PyBaMM Team"

# The short X.Y version
version = pybamm.__version__
# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.inheritance_diagram",
    # Local and custom extensions
    "extend_parent",
    "inheritance_diagram",
    # Third-party extensions
    "sphinx_design",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_inline_tabs",
    "sphinxcontrib.bibtex",
    "sphinx_docsearch",
    "sphinx_last_updated_by_git",
    "nbsphinx",  # to be kept below JavaScript-enabled extensions, always
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_gallery.load_style",
    "hoverxref.extension",
]


napoleon_use_rtype = True
napoleon_google_docstring = False

doctest_global_setup = """
from docs import *
"""

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]

# Suppress warnings generated by Sphinx and/or by Sphinx extensions
suppress_warnings = ["git.too_shallow"]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]

# Theme

# pydata theme options (see
# https://pydata-sphinx-theme.readthedocs.io/en/latest/index.html# for more information)
# mostly copied from numpy, scipy, pandas
html_logo = "_static/pybamm_logo.png"
html_favicon = "_static/favicon/favicon.png"

html_theme_options = {
    "logo": {
        "image_light": "pybamm_logo.png",
        "image_dark": "pybamm_logo.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pybamm-team/pybamm",
            "icon": "fa-brands fa-square-github",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/pybamm_",
            "icon": "fa-brands fa-square-twitter",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pybamm/",
            "icon": "fa-solid fa-box",
        },
    ],
    "collapse_navigation": True,
    "external_links": [
        {
            "name": "Contributing",
            "url": "https://github.com/pybamm-team/PyBaMM/tree/develop/CONTRIBUTING.md",
        },
    ],
    # should be kept versioned to use for the version warning bar
    "switcher": {
        "version_match": version,
        "json_url": "https://docs.pybamm.org/en/latest/_static/versions.json",
    },
    # turn to False to not fail build if json_url is not found
    "check_switcher": True,
    # for dark mode toggle and social media links
    # Note: the version switcher was removed in favour of the readthedocs one
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    # add Algolia to the persistent navbar, this removes the default search icon
    "navbar_persistent": "algolia-searchbox",
    "use_edit_page_button": True,
    "pygment_light_style": "xcode",
    "pygment_dark_style": "monokai",
    "footer_start": [
        "copyright",
        "sphinx-version",
    ],
    "footer_end": [
        "theme-version",
        "last-updated",
    ],
}

html_title = "%s v%s Manual" % (project, version)
html_last_updated_fmt = "%Y-%m-%d"
html_css_files = ["pybamm.css"]
html_context = {"default_mode": "light"}
html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = ".html"

htmlhelp_basename = "pybamm"

html_sidebars = {"**": ["sidebar-nav-bs.html", "sidebar-ethical-ads.html"]}

# For edit button
html_context = {
    "github_user": "pybamm-team",
    "github_repo": "pybamm",
    "github_version": "develop",
    "doc_path": "docs/",
}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "PyBaMMdoc"


# -- Options for LaTeX output ------------------------------------------------

# Note: we exclude the examples directory from the LaTeX build because it has
# problems with the creation of PDFs on Read the Docs
# https://github.com/readthedocs/readthedocs.org/issues/2045

# Detect if we are building LaTeX output through the invocation of the build commands
if any("latex" in arg for arg in sys.argv) or any("latexmk" in arg for arg in sys.argv):
    exclude_patterns.append("source/examples/*")
    print("Skipping compilation of .ipynb files for LaTeX build.")

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [(master_doc, "PyBaMM.tex", "PyBaMM Documentation", author, "manual")]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "pybamm", "PyBaMM Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "PyBaMM",
        "PyBaMM Documentation",
        author,
        "PyBaMM",
        "One line description of project.",
        "Miscellaneous",
    )
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- sphinxcontrib-bibtex configuration --------------------------------------

bibtex_bibfiles = ["../pybamm/CITATIONS.bib"]
bibtex_style = "unsrt"
bibtex_footbibliography_header = """.. rubric:: References"""
bibtex_reference_style = "author_year"
bibtex_tooltips = True

# -- nbsphinx configuration options ------------------------------------------

# Important: ensure require.js is not loaded. this is needed to avoid
# a conflict with the sphinx-docsearch extension for Algolia search

nbsphinx_requirejs_path = ""
nbsphinx_prolog = r"""

{% set github_docname =
'github/pybamm-team/pybamm/blob/develop/docs/' +
env.doc2path(env.docname, base=None) %}

{% set readthedocs_download_url =
'https://docs.pybamm.org/en/latest/' %}

{% set doc_path = env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div class="admonition tip">
        <p class="admonition-title">
            Tip
        </p>
        <p>
            An interactive online version of this notebook is available, which can be
            accessed via
            <a href="https://colab.research.google.com/{{ github_docname | e }}"
            target="_blank">
            <img src="https://colab.research.google.com/assets/colab-badge.svg"
            alt="Open this notebook in Google Colab"/></a>
        </p>
            <hr>
        <p>
            Alternatively, you may
            <a href="{{ readthedocs_download_url | e }}{{ doc_path | e }}"
            target="_blank" download>
            download this notebook</a> and run it offline.
        </p>
    </div>

"""

# -- sphinxext/inheritance_diagram.py options --------------------------------

graphviz_output_format = "svg"
inheritance_graph_attrs = dict(
    rankdir="TB",
    size='"10.0, 10.0"',
    fontsize=10,
    ratio="auto",
    center="true",
    nodesep=5,
    ranksep=0.35,
    bgcolor="white",
)
inheritance_node_attrs = dict(
    shape="box",
    fontsize=14,
    fontname="monospace",
    height=0.20,
    color="black",
    style="filled",
)
inheritance_edge_attrs = dict(
    arrowsize=0.75,
    style='"setlinewidth(0.5)"',
)

# -- Options for sphinx-hoverxref --------------------------------------------

# Hoverxref settings

hoverxref_default_type = "tooltip"
hoverxref_auto_ref = True

hoverxref_roles = ["class", "meth", "func", "ref", "term"]
hoverxref_role_types = dict.fromkeys(hoverxref_roles, "tooltip")

hoverxref_domains = ["py"]

# Currently, only projects that are hosted on readthedocs + CPython, NumPy, and
# SymPy are supported
hoverxref_intersphinx = list(intersphinx_mapping.keys())

# Tooltips settings
hoverxref_tooltip_lazy = False
hoverxref_tooltip_maxwidth = 750
hoverxref_tooltip_animation = "fade"
hoverxref_tooltip_animation_duration = 1
hoverxref_tooltip_content = "Loading information..."
hoverxref_tooltip_theme = ["tooltipster-shadow", "tooltipster-shadow-custom"]

# -- Options for Algolia DocSearch (sphinx-docsearch) ------------------------

# DocSearch settings
docsearch_app_id = "BXYTEF2JI8"
docsearch_api_key = "b7e7f1fc1a7c40a1587e52e8f4ff3b45"  # search API key, safe to use
docsearch_index_name = "pybamm"

# Searchbox settings
docsearch_container = "#algolia-docsearch"
docsearch_placeholder = "Search the PyBaMM documentation"

# -- Jinja templating --------------------------------------------------------
# Credit to: https://ericholscher.com/blog/2016/jul/25/integrating-jinja-rst-sphinx/


def rstjinja(app, docname, source):
    """
    Render our pages as a jinja template for fancy templating goodness.
    """
    # Make sure we're outputting HTML
    if app.builder.format != "html":
        return
    src = source[0]
    rendered = app.builder.templates.render_string(src, app.config.html_context)
    source[0] = rendered


def setup(app):
    app.connect("source-read", rstjinja)


# Context for Jinja Templates
html_context.update(
    {
        "parameter_sets": pybamm.parameter_sets,
    }
)
