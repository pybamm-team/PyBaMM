.. _rust-backend:

Rust Backend
============

PyBaMM ships a Rust compute core (``pybamm.rust``) that compiles a discretised
model's expression trees: right-hand sides, algebraic residuals, Jacobians,
events, and output variables into a form evaluated entirely outside Python.
Two solvers use it:

- :class:`pybamm.IDAKLUSolver` with ``model.convert_to_format = "rust"`` runs
  the usual SUNDIALS IDA integrator with every callback evaluated by the Rust
  core instead of CasADi. Solver options, forward sensitivities,
  ``output_variables``, events, and experiments work as they do with the
  CasADi backend.
- :class:`pybamm.DiffsolSolver` integrates with the pure-Rust BDF
  implementation from the `diffsol <https://github.com/martinjrobins/diffsol>`_
  crate, with no CasADi or SUNDIALS involvement. It supports forward
  sensitivities under solver error control, ``output_variables``-only solves,
  events, and ``t_interp``.

Selecting the backend
---------------------

Which backend a solver uses follows the model's ``convert_to_format``
attribute (documented on :class:`pybamm.BaseModel`):

.. code-block:: python

    import pybamm

    model = pybamm.lithium_ion.SPM()
    model.convert_to_format = "rust"

    sim = pybamm.Simulation(model)  # the default IDAKLUSolver picks up "rust"
    sol = sim.solve([0, 3600])

:class:`pybamm.DiffsolSolver` always uses the Rust backend and converts the
model itself:

.. code-block:: python

    sim = pybamm.Simulation(model, solver=pybamm.DiffsolSolver())
    sol = sim.solve([0, 3600])

Output times with ``DiffsolSolver``
-----------------------------------

Both solvers store the full state trajectory at every output time; they
differ in which times those are. ``IDAKLUSolver`` uses its internal
integrator steps as the output grid when ``t_interp`` is omitted, whereas
diffsol evaluates its error-controlled dense output at the requested times
alone. A bare span — ``solve([t0, tf])`` with no ``t_interp`` — is answered
on a uniform 100-point grid; pass ``t_interp`` (or a ``t_eval`` of three or
more points) to choose the output times exactly.

Off-grid reads — ``sol["Voltage [V]"](t)`` at a ``t`` between output points —
interpolate with cubic Hermite, as they do for ``IDAKLUSolver``: the solver
stores the state time derivatives alongside the solution by default. Pass
``hermite_interpolation=False`` to drop them and halve trajectory memory, at
the cost of off-grid reads falling back to linear interpolation. IDAKLU's
Hermite knots are its internal steps, while diffsol's are the requested output
times, so on a coarse output grid IDAKLU's off-grid reads remain the more
accurate of the two.

Supported expression types
--------------------------

The Rust backend supports the following expression types:

**Leaf nodes:**

- ``Scalar`` — constant scalar values
- ``Array`` / ``Vector`` / ``Matrix`` — dense arrays and matrices
- ``SparseMatrix`` — CSR sparse matrices
- ``StateVector`` / ``StateVectorDot`` — state variable slices
- ``InputParameter`` — named input parameters
- ``Time`` — simulation time

**Binary operators:**

- Arithmetic: ``+``, ``-``, ``*``, ``/``, ``**``
- Matrix: ``@`` (matrix multiplication)
- Comparison: ``minimum``, ``maximum``
- Other: ``modulo``, ``hypot``, ``EqualHeaviside``, ``NotEqualHeaviside``

**Unary operators:**

- ``-`` (negation), ``abs``
- Math functions: ``sqrt``, ``exp``, ``log``, ``sin``, ``cos``, ``tanh``,
  ``sinh``, ``cosh``, ``arcsinh``, ``arctan``, ``erf``, ``sign``, ``floor``,
  ``ceiling``
- Reductions: ``max``, ``min`` (over arrays), differentiated to the
  argmax/argmin subgradient

**Structural:**

- ``Index`` — array slicing
- ``Concatenation`` — combining arrays
- ``Conditional`` — branch selection
- ``VectorField`` — stacked components (read scalar components back with
  ``pybamm.Component``)

**Interpolation:**

- 1D ``linear``, ``cubic``, and ``pchip`` interpolation
- 2D and 3D ``linear`` and ``cubic`` regular-grid interpolation

A symbol the backend cannot convert raises an error naming the symbol at
solver set-up, so an unsupported model fails loudly rather than
mis-evaluating.

Adding new expression types
---------------------------

To add support for a new expression type:

1. **Rust side** (if a new node type is needed):

   - Add a variant to the ``Node`` enum in
     ``packages/pybamm-rust/pybamm-core/src/node.rs``
   - Implement evaluation in ``packages/pybamm-rust/pybamm-core/src/eval.rs``
   - Add a PyO3 binding in ``packages/pybamm-rust/pybamm-python/src/expr.rs``

2. **Python side:**

   - Add a ``_to_rust(self, graph, rust_symbols)`` method to the expression
     class
   - Follow the pattern of existing implementations (e.g.
     ``binary_operators.py``)

3. **Testing:**

   - Add a unit test in
     ``packages/pybamm/tests/unit/test_expression_tree/test_operations/test_convert_to_rust.py``
   - Add a parity test in
     ``packages/pybamm/tests/integration/test_rust_parity.py``

Example ``_to_rust`` implementation:

.. code-block:: python

    def _to_rust(self, graph, rust_symbols):
        """Convert to Rust expression graph."""
        # Convert children first
        converted_children = self._children_to_rust(graph, rust_symbols)
        # Call appropriate ExprGraph method
        return graph.some_method(*converted_children)
