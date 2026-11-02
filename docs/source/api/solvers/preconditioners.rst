Preconditioners
===============

A model's default preconditioner is a dictionary of block solvers keyed by state
variable name, :attr:`pybamm.BaseModel.default_preconditioner`, which
:class:`pybamm.Simulation` replaces with its ``preconditioner`` argument, as with
``var_pts``. :class:`pybamm.Discretisation` compiles it against the model's
states into ``model.preconditioner``. A state variable without an entry is solved by
:class:`pybamm.preconditioners.KLU` in the last stage, with that stage's
settings, so the default preconditioner is a single exact sparse LU
factorisation, and a preconditioner only lists the variables that need another
block solver.

Stages are nested Schur eliminations: the variables of the lowest stage are
eliminated first, and each later stage acts on the Schur complement of the ones
before it. Within a stage, variables with an equal block solver form one block.
A :class:`pybamm.preconditioners.Stage` sets how a stage's blocks are combined
and how its Schur complement is formed.

:class:`pybamm.IDAKLUSolver` applies the preconditioner with
``options={"preconditioner": "block"}`` and an iterative linear solver, or as a
direct linear solver with ``options={"linear_solver": "block"}`` when it is
exact.

.. code-block:: python

    P = pybamm.preconditioners
    sim = pybamm.Simulation(
        pybamm.lithium_ion.DFN(),
        solver=pybamm.IDAKLUSolver(options={"linear_solver": "block"}),
        preconditioner={
            "Negative particle concentration [mol.m-3]": P.Tridiagonal(stage=0),
            "Positive particle concentration [mol.m-3]": P.Tridiagonal(stage=0),
        },
    )

Block solvers serialise with ``to_config`` and ``from_config``, by class and
fields, and are saved with a model's default preconditioner.

Block solvers
-------------

.. autoclass:: pybamm.preconditioners.BlockSolver
  :members: specification, to_config, from_config

.. autoclass:: pybamm.preconditioners.Stage

.. autoclass:: pybamm.preconditioners.KLU

.. autoclass:: pybamm.preconditioners.ILU

.. autoclass:: pybamm.preconditioners.Jacobi

.. autoclass:: pybamm.preconditioners.Tridiagonal

External block solvers
----------------------

A block solver can be implemented outside PyBaMM as a plugin: a C library
implementing ``pybamm_block_solver.h`` (in ``pybammsolvers.get_include()``),
exposed to Python as a capsule, and a :class:`pybamm.preconditioners.BlockSolver`
subclass whose :meth:`~pybamm.preconditioners.BlockSolver.specification` returns
that capsule and the plugin's options.

Preconditioner trees
--------------------

.. autofunction:: pybamm.preconditioners.compile_tree

.. autoclass:: pybamm.preconditioners.Leaf

.. autoclass:: pybamm.preconditioners.Additive

.. autoclass:: pybamm.preconditioners.Multiplicative

.. autoclass:: pybamm.preconditioners.Schur

.. autofunction:: pybamm.preconditioners.specification

.. autofunction:: pybamm.preconditioners.describe

.. autofunction:: pybamm.preconditioners.tree_to_config

.. autofunction:: pybamm.preconditioners.tree_from_config

Block preconditioner backend
----------------------------

.. autofunction:: pybamm.preconditioners.analyse

.. autofunction:: pybamm.preconditioners.analyse_jacobian

.. autoclass:: pybamm.preconditioners.analysis.JacobianAnalysis
