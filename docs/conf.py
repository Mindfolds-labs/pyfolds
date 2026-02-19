from __future__ import annotations

import os
import sys
from datetime import datetime

# Adiciona src ao path
sys.path.insert(0, os.path.abspath("../src"))

# -----------------------------------------------------------------------------
# Project info
# -----------------------------------------------------------------------------

project = "PyFolds"
author = "PyFolds Contributors"
copyright = f"{datetime.now().year}, {author}"

# -----------------------------------------------------------------------------
# Extensions
# -----------------------------------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.plantuml",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "pt_BR"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
    "linkify",
    "substitution",
]

# -----------------------------------------------------------------------------
# HTML
# -----------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/brand/pyfolds-icon-dark.svg"
html_favicon = "_static/brand/favicon.svg"

html_theme_options = {
    "logo": {"text": "PyFolds Docs"},
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-icon-links"],
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "secondary_sidebar_items": ["page-toc"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/",
            "icon": "fa-brands fa-github",
        }
    ],
}

# -----------------------------------------------------------------------------
# Warnings policy (baseline control)
# -----------------------------------------------------------------------------

nitpicky = False

# Suprime warning massivo de documentos fora do toctree
suppress_warnings = [
    "toc.not_included",
]

