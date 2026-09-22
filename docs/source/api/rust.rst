Rust compute core
=================

``pybamm.rust`` exposes the Rust compute core that ships compiled inside the
``pybamm`` wheel. An :class:`~pybamm.rust.ExprGraph` holds expression nodes, and
compiling one of them gives a :class:`~pybamm.rust.CompiledFunction` that evaluates
the expression and its Jacobian-vector products, and a
:class:`~pybamm.rust.CompiledJacobian` that assembles its sparse Jacobian.

.. autoclass:: pybamm.rust.ExprGraph
  :members:

.. autoclass:: pybamm.rust.Expr
  :members:

.. autoclass:: pybamm.rust.CompiledFunction
  :members:

.. autoclass:: pybamm.rust.CompiledJacobian
  :members:
