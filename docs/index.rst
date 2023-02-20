.. Root of all pybamm docs

.. Remove the right side-bar for the home page

:html_theme.sidebar_secondary.remove:

####################
PyBaMM documentation
####################

.. This TOC defines what goes in the top navbar
.. toctree::
   :maxdepth: 1
   :hidden:

   User Guide <source/user_guide/index>
   source/api/index

**Version**: |version|

**Useful links**:
`Project Home Page <https://www.pybamm.org>`_ |
`Installation <source/user_guide/installation/index.html>`_ |
`Source Repository <https://github.com/pybamm-team/pybamm>`_ |
`Issue Tracker <https://github.com/pybamm-team/pybamm/issues>`_ |
`Discussions <https://github.com/pybamm-team/pybamm/discussions>`_

PyBaMM (Python Battery Mathematical Modelling) is an open-source battery simulation package
written in Python. Our mission is to accelerate battery modelling research by
providing open-source tools for multi-institutional, interdisciplinary collaboration. 
Broadly, PyBaMM consists of

#. a framework for writing and solving systems of differential equations,
#. a library of battery models and parameters, and
#. specialized tools for simulating battery-specific experiments and visualizing the results.

Together, these enable flexible model definitions and fast battery simulations, allowing users to
explore the effect of different battery designs and modeling assumptions under a variety of operating scenarios.

.. grid:: 2

   .. grid-item-card::
      :img-top: source/_static/index-images/getting_started.svg

      User Guide
      ^^^^^^^^^^

      The user guide is the best place to start learning PyBaMM. It contains an installation
      guide, an introduction to the main concepts and links to additional tutorials.

      +++

      .. button-ref:: source/user_guide/index
         :expand:
         :color: secondary
         :click-parent:

         To the user guide
    
   .. grid-item-card::
      :img-top: source/_static/index-images/examples.svg

      Examples
      ^^^^^^^^

      Examples and tutorials can be viewed on the GitHub examples page,
      which also provides a link to run them online through Google Colab.

      +++

      .. button-link:: https://github.com/pybamm-team/PyBaMM/tree/develop/examples/notebooks
         :expand:
         :color: secondary
         :click-parent:

         To the examples

   .. grid-item-card::
      :img-top: source/_static/index-images/api.svg

      API Documentation
      ^^^^^^^^^^^^^^^^^

      The reference guide contains a detailed description of the functions,
      modules, and objects included in PyBaMM. The reference describes how the
      methods work and which parameters can be used.

      +++

      .. button-ref:: source/api/index
         :expand:
         :color: secondary
         :click-parent:

         To the API documentation

   .. grid-item-card::
      :img-top: source/_static/index-images/contributor.svg

      Contributor's Guide
      ^^^^^^^^^^^^^^^^^^^

      Contributions to PyBaMM and its development are welcome! If you have ideas for
      features, bug fixes, models, spatial methods, or solvers, we would love to hear from you.

      +++

      .. button-link:: https://github.com/pybamm-team/PyBaMM/blob/develop/CONTRIBUTING.md
         :expand:
         :color: secondary
         :click-parent:

         To the contributor's guide
