# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))

autodoc_mock_imports = ['ignite', 'ignite.metrics', 'ignite.utils', 'ignite.contrib.handlers.tqdm_logger', 'ignite.engine',
                'ignite.engine.engine', 'ignite.contrib.handlers.tensorboard_logger', 'torch', 'torch.nn', 'torch.optim',
                'smart_open', 'tensorboardX', 'pandas', 'numpy', 'torch.utils.data']

# -- Project information -----------------------------------------------------

project = 'Transfer NLP'
copyright = '2019, Peter Martigny'
author = 'Peter Martigny'

# The full version, including alpha/beta/rc tags
release = '0.0.3'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinxcontrib.napoleon']

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
}
html_logo = '_static/TransferNLP_Logo.jpg '
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_context = {
    'css_files': [
        'https://fonts.googleapis.com/css?family=Lato',
        '_static/css/pytorch_theme.css'
    ],
}
