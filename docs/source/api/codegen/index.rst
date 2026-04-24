Code generation
===============

PyBaMM evaluates CasADi expressions through one of two backends:

* ``compile=False`` (default) -- CasADi's in-process virtual machine.
* ``compile=True`` -- :func:`pybamm.codegen.compilation.aot_compile` emits C
  source for a CasADi ``Function``, compiles it to a shared library and
  returns a ``casadi.external`` wrapper. Results are cached in-process and on
  disk, keyed by a hash of the serialised function.

The backend is selected via the ``compile`` option on a solver, e.g.
``pybamm.IDAKLUSolver(options={"compile": True})``. The setting is forwarded
to :class:`pybamm.Solution` so post-solve variable observation uses the same
backend as the integration.

.. toctree::

  compilation
