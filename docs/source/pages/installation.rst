.. _installation:

Installation
============

Installing Ploonetide from PyPI
-------------------------------

To install the latest stable version of **ploonetide** from PyPI, simply run:

.. code-block:: bash

   pip install ploonetide

This will install all core dependencies required to use Ploonetide.

Requirements
------------

- Python 3.10 or above
- `pip` (Python package installer)

Testing your installation
-------------------------

After installation, you can test that everything is working correctly:

.. code-block:: python

   import ploonetide
   simulation = ploonetide.TidalSimulation()

You should see an informational message from the simulation.

.. note::

   If you're a developer looking to contribute or modify the source code, please refer to the :ref:`Developer Guide <developer_guide>` for full setup instructions.
