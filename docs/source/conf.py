import os
import sys

from distutils.util import convert_path
sys.path.append(os.path.abspath(
    os.path.join(__file__, "../../src/")
))

# -- Project information -----------------------------------------------------

project = f'Ploonetide'
copyright = 'Jaime A. Alvarado-Montes'
author = 'Ploonetide developers'


# Load the __version__ variable without importing the package already
main_ns = {}
ver_path = convert_path('../../src/ploonetide/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

# The full version, including alpha/beta/rc tags.
release = main_ns['__version__']


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'numpydoc',
    'myst_parser'
]

autosummary_generate = True

autoclass_content = 'both'

# Exclude build directory and Jupyter backup files:
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Execute notebooks? Possible values: 'always', 'never', 'auto' (default)
nbsphinx_execute = "auto"

# Some notebook cells take longer than 60 seconds to execute
nbsphinx_timeout = 300

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["**/.ipynb_checkpoints"]

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Execute notebooks? Possible values: 'always', 'never', 'auto' (default)
nbsphinx_execute = "auto"

# Some notebook cells take longer than 60 seconds to execute
nbsphinx_timeout = 300

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "external_links": [],
    "github_url": "https://github.com/JAAlvarado-Montes/ploonetide",
}


html_title = "Ploonetide "

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "defaul.css"
html_static_path = ['_static']
html_logo = '_static/images/logo_nobkg.png'


# Raw files we want to copy using the sphinxcontrib-rawfiles extension:
# - CNAME tells GitHub the domain name to use for hosting the docs
# - .nojekyll prevents GitHub from hiding the `_static` dir
rawfiles = ['CNAME', '.nojekyll']

# Make sure text marked up `like this` will be interpreted as Python objects
default_role = 'py:obj'

# intersphinx enables links to classes/functions in the packages defined here:
intersphinx_mapping = {'python': ('https://docs.python.org/3/', None),
                       'numpy': ('https://docs.scipy.org/doc/numpy/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
                       'matplotlib': ('https://matplotlib.org', None),
                       'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
                       'astropy': ('https://docs.astropy.org/en/latest/', None)}
