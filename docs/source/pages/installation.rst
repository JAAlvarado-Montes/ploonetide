.. _installation:

Installation
============

Create and activate an environment
----------------------------------

**Option 1: Poetry (recommended)**

If you don't have Poetry, you can install it with:

.. code-block:: bash

   pip install --user poetry

Then install and activate the environment:

.. code-block:: bash

   make install
   poetry shell

**Option 2: Conda**

If you prefer using Conda, the same command will export an environment file and create the env:

.. code-block:: bash

   make install
   conda activate ploonetide-env

Install ploonetide and its dependencies
---------------------------------------

After the environment is active, all dependencies and the package will be installed automatically. You can also manually install with:

.. code-block:: bash

   pip install -e .[dev]

Testing your ploonetide installation
------------------------------------

Once inside the environment, test your install:

.. code-block:: python

   import ploonetide
   simulation = ploonetide.TidalSimulation()

You should see a print statement.
