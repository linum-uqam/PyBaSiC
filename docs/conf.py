# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""Sphinx configuration for the linum-basic documentation."""

import sys
from datetime import datetime
from pathlib import Path

# Make the project package importable for autoapi/autodoc.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# -- Project information ----------------------------------------------------
project = "linum-basic"
author = "The LINUM developers"
copyright = f"{datetime.now().year}, LINUM"

# Pull version from installed package metadata when available.
try:
    from importlib.metadata import version as _get_version

    release = _get_version("linum-basic")
except Exception:
    release = "0.2.0"
version = ".".join(release.split(".")[:2])

# -- General configuration --------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "autoapi.extension",
    "sphinxarg.ext",
    "myst_parser",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "notfound.extension",
    "sphinx_sitemap",
    "sphinxext.opengraph",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST: render the existing Markdown docs.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "linkify",
    "substitution",
    "tasklist",
]
myst_fence_as_directive = ["mermaid"]

# -- nbsphinx (notebook rendering) -----------------------------------------
# Execute notebooks during the Sphinx build so Read the Docs renders outputs.
nbsphinx_execute = "always"
# Kernel used for notebook execution — matches the ipykernel installed via
# the docs extra.  Override with NBSphinx_KERNEL_NAME env var if needed.
nbsphinx_kernel_name = "python3"
# Fail the build if a notebook raises an exception so broken notebooks are
# caught early. Set to True if environment-specific cells may fail on RTD.
nbsphinx_allow_errors = False
# Increase per-cell timeout for the ALM solver (seconds).
nbsphinx_timeout = 120
# Embed ipywidgets widget state so tqdm.auto progress bars render inline
# as static HTML rather than as raw text streams.
nbsphinx_widgets_path = ""  # use CDN delivery of the widget JS

# Mermaid: interactive zoom/pan + fullscreen, with readable defaults.
mermaid_d3_zoom = True
mermaid_fullscreen = True
mermaid_fullscreen_button = "⛶"
mermaid_height = "640px"
mermaid_light_theme = "neutral"
mermaid_dark_theme = "dark"
# startOnLoad must be False — sphinxcontrib-mermaid's default.js calls
# mermaid.run() itself after wiring d3 zoom and the fullscreen modal.
mermaid_init_config = {
    "startOnLoad": False,
    "securityLevel": "loose",
    "flowchart": {"htmlLabels": True, "curve": "basis", "useMaxWidth": True},
    "themeVariables": {"fontSize": "16px"},
}
myst_heading_anchors = 4

# Autoapi: generate API reference from the linum_basic package.
autoapi_type = "python"
autoapi_dirs = [str(ROOT / "linum_basic")]
autoapi_root = "api"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_ignore = ["*/tests/*"]
autoapi_keep_files = True
autoapi_add_toctree_entry = True

# Autodoc settings.
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Napoleon: support Google + NumPy docstrings.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True
# Render "Attributes:" docstring sections as :ivar: fields instead of
# emitting a separate :py:attribute: directive for each — avoids the
# "duplicate object description" warnings when autoapi also documents
# the same class attributes from their type annotations.
napoleon_use_ivar = True

# Intersphinx mappings.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Nitpicky mode: off for production builds (autoapi extracts informal types
# from NumPy-style docstrings that cannot resolve as cross-references).
# Run ``sphinx-build -n`` ad-hoc when reviewing docstring quality.
nitpicky = False

# -- HTML output ------------------------------------------------------------
# pydata-sphinx-theme: https://pydata-sphinx-theme.readthedocs.io/
html_theme = "pydata_sphinx_theme"
html_title = "Linum BaSiC"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/linum-uqam/Linum-BaSiC",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_prev_next": True,
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/linum-uqam/Linum-BaSiC",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

html_theme_options["secondary_sidebar_items"] = {
    "**": ["page-toc"],
    "index": [],
    "getting_started": [],
    "algorithm": [],
    "parameters": [],
    "gpu": [],
    "contributing": [],
    "validation": [],
    "reference": [],
}

html_context = {
    "github_user": "linum-uqam",
    "github_repo": "Linum-BaSiC",
    "github_version": "dev",
    "doc_path": "docs",
}

# Suppress known-harmless warnings generated by autoapi from NumPy-style
# docstrings: informal composite types (N, P, shape, NDArray, str/Path, etc.),
# unreferenced footnotes from References sections, and duplicate footnote
# labels when multiple functions each have a References section in the same
# autoapi-generated RST file.
suppress_warnings = [
    "autoapi.python_import_resolution",
    "ref.python",
    # Unresolvable informal types emitted by autoapi (ref.class, ref.func, ref.mod).
    "ref.class",
    "ref.func",
    "ref.mod",
    "ref.data",  # autoapi default-arg names (e.g. GPU_MIN_ELEMENTS in BaSiC signature)
    "ref.footnote",
    "docutils",
    "misc.highlighting_failure",
]

# -- UX extensions ----------------------------------------------------------
# sphinx-copybutton: copy button on code blocks; strip prompt characters.
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False

# sphinx-notfound-page: serve a friendly 404 with absolute links to assets.
notfound_context = {
    "title": "Page not found",
    "body": (
        "<h1>Page not found</h1>"
        "<p>Sorry, we couldn't find that page. Try the "
        "<a href='/'>documentation home</a> or use the search box above.</p>"
    ),
}
notfound_urls_prefix = "/"

# sphinx-sitemap: emit sitemap.xml at the docs root for SEO.
html_baseurl = "https://linum-basic.readthedocs.io/en/latest/"
sitemap_url_scheme = "{link}"

# sphinxext-opengraph: rich link previews on social platforms.
ogp_site_url = html_baseurl
ogp_site_name = "Linum BaSiC documentation"
ogp_use_first_image = True
ogp_enable_meta_description = True
