.. _installation:

Installation
============

Create a virtual environment
----------------------------

ploonetide requires `Python 3.10 or above`_. It is recommended that you create a dedicated `Python environment`_ before you install ploonetide. In your project directory, run the following commands:

.. code-block:: bash

   python -m venv env

Then activate your new virtual environment.

On macOS and Linux:

.. code-block:: bash

   source env/bin/activate

On Windows:

.. code-block:: bash

   .\env\Scripts\activate

Install ploonetide and its dependencies
---------------------------------------

With your virtual environment activated, install the package from PyPI:

.. code-block:: bash

   pip install ploonetide

Or, if you are working from source and want to install all core, optional, and development dependencies:

**Option 1 – Using Poetry** (recommended for contributors):

.. code-block:: bash

   poetry install --with dev --extras "dev"
   poetry shell

**Option 2 – Using pip** (editable install):

.. code-block:: bash

   pip install -e .[dev]

This will install:

- Core dependencies
- GUI, plotting, notebook, and documentation extras
- Developer tools (e.g., pytest, flake8, black)

Testing your ploonetide installation
------------------------------------

With your environment activated, start Python and type the following:

.. code-block:: python

    import ploonetide
    simulation = ploonetide.TidalSimulation()

You should see a print statement.


.. _Python 3.10 or above: https://www.python.org/downloads/
.. _Python environment: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment
