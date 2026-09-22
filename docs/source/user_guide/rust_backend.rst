.. _rust-backend:

Rust Backend
============

PyBaMM ships a Rust compute core (``pybamm.rust``) that compiles a discretised
model's expression trees: right-hand sides, algebraic residuals, Jacobians,
events, and output variables into a form evaluated entirely outside Python.
Two solvers use it:

- :class:`pybamm.IDAKLUSolver` runs the usual SUNDIALS IDA integrator with
  every callback evaluated by the Rust core instead of CasADi. It supports
  forward sensitivities, ``output_variables``, events, and experiments, with
  the differences listed in `Differences from the CasADi backend`_.
- :class:`pybamm.DiffsolSolver` integrates with the pure-Rust BDF
  implementation from the `diffsol <https://github.com/martinjrobins/diffsol>`_
  crate, with no CasADi or SUNDIALS involvement. It supports forward
  sensitivities under solver error control, ``output_variables``-only solves,
  events, and ``t_interp``.

Selecting the backend
---------------------

Which backend a solver uses follows the model's ``convert_to_format``
attribute (documented on :class:`pybamm.BaseModel`), which defaults to
``"rust"``:

.. code-block:: python

    import pybamm

    model = pybamm.lithium_ion.SPM()

    sim = pybamm.Simulation(model)  # the default IDAKLUSolver, on the Rust core
    sol = sim.solve([0, 3600])

Set ``convert_to_format = "casadi"`` before solving to use the CasADi backend
instead:

.. code-block:: python

    model = pybamm.lithium_ion.SPM()
    model.convert_to_format = "casadi"

:class:`pybamm.DiffsolSolver` always uses the Rust backend and converts the
model itself:

.. code-block:: python

    sim = pybamm.Simulation(model, solver=pybamm.DiffsolSolver())
    sol = sim.solve([0, 3600])

Differences from the CasADi backend
-----------------------------------

On the Rust backend, :class:`pybamm.IDAKLUSolver`:

- raises :class:`pybamm.SolverError` for sensitivities with respect to an input
  parameter that holds several values (``pybamm.InputParameter`` with an
  ``expected_size`` above 1);
- runs one solver per thread, so it solves with ``num_solvers`` equal to
  ``num_threads``, with a warning when they differ.

Set ``convert_to_format = "casadi"`` for either.

Output times with ``DiffsolSolver``
-----------------------------------

Without ``output_variables``, both solvers store the full state at every
output time; they differ in which times those are. ``IDAKLUSolver`` uses its
internal integrator steps as the output grid when ``t_interp`` is omitted,
whereas diffsol evaluates its error-controlled dense output at a grid of its
own: ``t_eval`` and the times of any discontinuities, merged with 100 points
uniform over the span. Pass ``t_interp`` to choose the output times exactly;
they are then ``t_interp`` merged with ``t_eval`` and any discontinuity times.

Off-grid reads — ``sol["Voltage [V]"](t)`` at a ``t`` between output points —
interpolate with cubic Hermite, as they do for ``IDAKLUSolver``: the solver
stores the state time derivatives alongside the solution by default. Pass
``hermite_interpolation=False`` to drop them and halve trajectory memory, at
the cost of off-grid reads falling back to linear interpolation. IDAKLU's
Hermite knots are its internal steps, while diffsol's are its output times, so
on a coarse ``t_interp`` IDAKLU's off-grid reads remain the more accurate of
the two.

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

A symbol reaches the Rust core through its ``_to_rust(graph, rust_symbols)``
method, which allocates nodes on a ``pybamm.rust.ExprGraph``. Subclasses of
:class:`pybamm.Function` override ``_rust_evaluate(graph, *children)`` instead,
which receives the children already converted.

1. **If existing graph methods can express it**, only the Python side
   changes: compose them in ``_rust_evaluate`` or ``_to_rust``, as
   ``Arcsinh2`` in ``functions.py`` does.

2. **Otherwise, add a primitive** under ``packages/pybamm-rust``:

   - add a ``Node`` variant in ``pybamm-core/src/node.rs``, the vocabulary of
     the graph Python builds;
   - lower it to the instruction tape in ``pybamm-core/src/ir/mod.rs``, adding
     a ``UnaryOp`` or ``BinaryOp`` variant for an element-wise operation;
   - define that op's value in ``pybamm-core/src/eval.rs``
     (``with_unary_op!`` or ``with_binary_op!``), which every interpreter of
     the tape shares;
   - give it a derivative in ``tangent.rs`` (forward mode) and ``adjoint.rs``
     (reverse mode), and an arm in each pass whose ``match`` the compiler then
     reports as non-exhaustive (``simplify.rs``, ``sparsity.rs``,
     ``zero_propagate.rs``, ``row_extract.rs``, ...);
   - expose it as an ``ExprGraph`` method in ``pybamm-python/src/expr.rs``,
     and declare that method in the hand-written stubs,
     ``packages/pybamm/src/pybamm/rust/_core.pyi``, in the same change
     (``tests/unit/test_rust_stubs.py`` checks them against the extension).

3. **Test** the lowering in
   ``packages/pybamm/tests/unit/test_expression_tree/test_operations/test_convert_to_rust.py``
   and its values against CasADi in
   ``packages/pybamm/tests/integration/test_rust_parity.py``. A new primitive
   also joins the generators in ``pybamm-core/tests/common/cases.rs``, so the
   Rust property tests exercise its values and derivatives.
