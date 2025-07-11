
.. _installation:

Installation
============

Create a virtual environment (Poetry or Conda)
----------------------------------------------

ploonetide works with `Python 3.10 or above`_. You can set up your environment using **Poetry** (recommended) or **Conda**.

**Option 1: Poetry (recommended)**

Install Poetry if you haven't already:

.. code-block:: bash

   pip install --user poetry

Then create and activate the virtual environment and install all dependencies:

.. code-block:: bash

   make install

This will:
- Install Poetry (if missing)
- Create and activate a dedicated virtual environment
- Install ploonetide and all core/dev dependencies

**Option 2: Conda**

If you prefer Conda and have it installed:

.. code-block:: bash

   conda env create -f conda_environment.yml
   conda activate ploonetide-env

Testing your ploonetide installation
------------------------------------

After activating your environment (Poetry or Conda), test the package:

.. code-block:: python

   import ploonetide
   simulation = ploonetide.TidalSimulation()

You should see a print statement from the simulation.

.. _Python 3.10 or above: https://www.python.org/downloads/
