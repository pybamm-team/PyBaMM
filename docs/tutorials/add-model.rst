
.. _CONTRIBUTING.md: https://github.com/pybamm-team/PyBaMM/blob/master/CONTRIBUTING.md


Adding a Model
==============

As with any contribution to PyBaMM, please follow the workflow in CONTRIBUTING.md_.
In particular, start by creating an issue to discuss what you want to do - this is a good way to avoid wasted coding hours!


The role of models
------------------

One of the main motivations for PyBaMM is to allow for new models of batteries to be easily be added, solved, tested, and compared without requiring a detailed knowledge of sophisticated numerical methods. It has therefore been our focus to make the process of adding a new model as simple as possible. To achieve this, all models in PyBaMM are implemented as `expression trees <https://github.com/pybamm-team/PyBaMM/blob/master/examples/notebooks/expression-tree.ipynb>`_, which abstract away the details of computation. 

The fundamental building blocks of a PyBaMM expression tree are Symbols. There are different types of Symbol: Variables, Parameters, Addition, Multiplication, Gradient etc which have been created so that each component of a model written out in PyBaMM mirrors exactly the written mathematics. For example, the expression:

.. math::
    \nabla \cdot \left(D(c) \nabla c \right) + a F j

is simply written as

.. code-block:: python

    div(D(c) * grad(c)) + a * F * j

within PyBaMM. A model basically consists a set of expression trees
which represent the different components of the model: the governing
equations, the boundary conditions, the initial conditions, and useful
outputs (e.g. the terminal voltage).
