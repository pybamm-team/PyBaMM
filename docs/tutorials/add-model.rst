
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

within PyBaMM. A model in PyBaMM is essentially an organised collection of
expression trees. 

Implementing a new model
------------------------

To add a new model (e.g. My New Model), first create a new file (``my_new_model.py``) in ``pybamm/models`` (or the relevant subdirectory).
In this file create a new class which inherits from :class:`pybamm.BaseModel` 
(or :class:`pybamm.LithiumIonBaseModel` if you are modelling a full lithium-ion battery or :class:`pybamm.LeadAcidBaseModel` if you are modelling a full lead acid battery): 

.. code-block:: python

    class MyNewModel(pybamm.BaseModel):

and add the class to ``pybamm/__init__.py``:

.. code-block:: python

    from .models.my_new_model import MyNewModel

(this line will be slightly different if you created your model in a subdirectory of models). Within your new class :class:`MyNewModel`, first create an initialisation function which calls the initialisation function of the parent class

.. code-block:: python

    def __init__(self):
        super().__init__()

Within the initialisation function of :class:`MyNewModel` you must then define the following attributes: 

- ``self.rhs``
- ``self.algebraic``
- ``self.boundary_conditions``
- ``self.initial_conditions``
- ``self.variables``

You may also optionally provide also provide: 

- ``self.default_geometry``
- ``self.default_solver``
- ``self.events``
- ``self.default_spatial_methods``
- ``self.default_submesh_types``
- ``self.default_var_pts``
- ``self.default_parameter_values``

We will go through each of these attributes in turn here for completeness but refer the user to the API documentation or example notebooks (create-model.ipnb) if further details are required. 

Governing equations
~~~~~~~~~~~~~~~~~~~
The governing equations which can either be parabolic or elliptic are entered into the 
``self.rhs`` and ``self.algebraic`` dictionaries, respectively. We associate each governing equation with a subject variable, which is the variable that is found when 
the equation is solved. We use this subject variable as the key of the dictionary. For parabolic equations, we rearrange the equation so that the time derivative of the subject variable is the only term on the left hand side of the equation. We then simply write the resulting right hand side into the `self.rhs` dictionary 
with the key being the variable whi
  




Unit tests for a MyNewModel
---------------------------