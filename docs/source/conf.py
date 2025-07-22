import os
import sys
from distutils.util import convert_path

# -- Path setup --------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(__file__, "../../src/")))

# -- Project information -----------------------------------------------------
project = 'Ploonetide'
author = 'Ploonetide developers'
copyright = 'Jaime A. Alvarado-Montes'

# Load version from version.py
main_ns = {}
ver_path = convert_path('../../src/ploonetide/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)
release = main_ns['__version__']

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinx.ext.githubpages',
    'numpydoc',
    'myst_nb',  # Enables Jupyter notebook parsing and execution
]

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'inherited-members': True,
}
autoclass_content = 'both'
default_role = 'py:obj'

# -- myst-nb configuration ---------------------------------------------------
# New (current as of myst-nb >= 1.0)
nb_execution_mode = "auto"  # auto / force / cache / off
nb_execution_timeout = 300  # seconds

# -- Source files and exclusions ---------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
master_doc = 'index'
exclude_patterns = ['_build', '**/.ipynb_checkpoints']
language = 'en'
pygments_style = 'default'
todo_include_todos = True

# -- HTML output -------------------------------------------------------------
html_theme = 'furo'
html_title = "Ploonetide"
# html_logo = '_static/logo_nobkg.png'
html_static_path = ['_static']

# -- Intersphinx mappings ----------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'astropy': ('https://docs.astropy.org/en/latest/', None),
}

# -- GitHub integration for Furo theme ---------------------------------------
html_theme_options = {
    "source_repository": "https://github.com/JAAlvarado-Montes/ploonetide/",
    "source_branch": "main",
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
}
