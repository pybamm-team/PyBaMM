Installation
============

PyBaMM is available on GNU/Linux, MacOS and Windows.
.. grid-item-card::
      :column: col-lg-6 col-md-6 col-sm-12 col-xs-12 p-3

      Working with conda?
      ^^^^^^^^^^^^^^^^^^^

      PyBaMM is part of the `Anaconda <https://docs.continuum.io/anaconda/>`__
      distribution and is available as a conda package through the conda-forge channel.

      ++++++++++++++++++++++

      .. code-block:: bash

         conda install -c conda-forge pybamm

   .. grid-item-card::
   
      Prefer pip?
      ^^^^^^^^^^^

      PyBaMM can be installed via pip from `PyPI <https://pypi.org/project/pybamm>`__.

      ++++

      .. code-block:: bash

         pip install pybamm

   .. grid-item-card::
      :column: col-12 p-3

      In-depth instructions?
      ^^^^^^^^^^^^^^^^^^^^^^
      Installing a specific version? Installing from source? Check the advanced
      installation page.

      .. button-ref:: installation/index
         :classes: btn-secondary stretched-link
         :expand:
         :color: secondary
         :click-parent:

         To detailed installation guide
Using pip
----------

GNU/Linux and Windows
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   pip install pybamm

macOS
~~~~~

.. code:: bash

   brew install sundials && pip install pybamm

Using conda
-----------
PyBaMM is available as a conda package through the conda-forge channel.

.. code:: bash

   conda install -c conda-forge pybamm

Optional solvers
----------------
Following GNU/Linux and macOS solvers are optionally available:

*  `scikits.odes <https://scikits-odes.readthedocs.io/en/latest/>`_ -based solver, see `Optional - scikits.odes solver <https://pybamm.readthedocs.io/en/latest/install/GNU-linux.html#optional-scikits-odes-solver>`_.
*  `jax <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_ -based solver, see `Optional - JaxSolver <https://pybamm.readthedocs.io/en/latest/install/GNU-linux.html#optional-jaxsolver>`_.

.. toctree::
   :maxdepth: 1

   GNU-linux
   windows
   windows-wsl
   install-from-source