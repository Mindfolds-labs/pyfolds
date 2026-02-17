from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../src"))

project = "PyFolds"
author = "PyFolds Contributors"
copyright = f"{datetime.now().year}, {author}"

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

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/brand/pyfolds-logo.png"
html_favicon = "_static/brand/favicon.png"

html_theme_options = {
    "logo": {
        "text": "PyFolds Docs",
    },
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

nitpicky = False
