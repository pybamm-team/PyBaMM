"""Shared lowering of a discretised model into a Rust ``CompiledModel``."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pybamm


def rust_graph_with_inputs(model: pybamm.BaseModel, input_names):
    """Build an ``ExprGraph`` with ``input_names`` registered in order.

    Graph input indices are positional, and both the stacked-input convention
    shared with CasADi and the sensitivity index mapping rely on that position
    matching the caller's ordering, so registration happens before any lowering.

    Parameters
    ----------
    model : :class:`pybamm.BaseModel`
        Discretised model, used to size vector-valued input parameters.
    input_names : iterable of str
        Input parameter names, in the order the solver will stack them.

    Returns
    -------
    ExprGraph
        Graph with every input registered.
    """
    from pybamm.rust import ExprGraph
    from pybamm.solvers.base_solver import rust_input_parameter_widths

    graph = ExprGraph()
    widths = rust_input_parameter_widths(model)
    for name in input_names:
        graph.input_parameter(name, widths.get(name, 1))
    return graph


class RustModelLowering:
    """Lower one discretised model into a Rust ``CompiledModel``.

    Holds the single ``ExprGraph`` that the state residual, output variables,
    events and algebraic block are all lowered into, so common subexpressions are
    shared and every part sees the same input registration. Callers compose only
    the parts their solver needs, then call :meth:`compile`.

    Parameters
    ----------
    model : :class:`pybamm.BaseModel`
        The discretised model to lower.
    inputs_dict : dict
        Input parameter values; only the key order is used.
    """

    def __init__(self, model: pybamm.BaseModel, inputs_dict: dict):
        self.model = model
        self.input_name_order = list(inputs_dict)
        self.graph = rust_graph_with_inputs(model, self.input_name_order)
        self._symbols: dict = {}
        self._state_residual = None
        self._output_exprs: list = []
        self._output_lens: list[int] = []
        self._event_exprs: list = []
        self._algebraic_expr = None
        self._algebraic_var_indices: list[int] = []
        self._sens_param_indices: list[int] = []

    def lower(self, symbol: pybamm.Symbol):
        """Lower one symbol into this graph, reusing already-lowered subtrees."""
        return symbol.to_rust(self.graph, self._symbols)

    def state_residual(self, algebraic_only: bool = False):
        """Lower the right-hand side, concatenated with the algebraic block.

        A DAE's residual must produce ``len_rhs_and_alg`` outputs to line up with
        the mass matrix, whose algebraic rows are empty.

        Parameters
        ----------
        algebraic_only : bool, optional
            Lower only the algebraic block, for a model with no ``rhs`` (default
            False).

        Returns
        -------
        Expr
            The lowered residual.
        """
        model = self.model
        if algebraic_only:
            residual = model.concatenated_algebraic
        elif model.len_alg > 0:
            residual = pybamm.numpy_concatenation(
                model.concatenated_rhs, model.concatenated_algebraic
            )
        else:
            residual = model.concatenated_rhs
        self._state_residual = self.lower(residual)
        return self._state_residual

    def algebraic_block(self, first_algebraic_index: int | None = None):
        """Lower the algebraic sub-block used by the Newton initialisation.

        Parameters
        ----------
        first_algebraic_index : int, optional
            Global state index the algebraic block starts at. Defaults to
            ``model.len_rhs``.

        Returns
        -------
        tuple
            ``(expr, var_indices)``, both empty-ish for a pure ODE.
        """
        model = self.model
        if model.len_alg == 0:
            return None, []
        start = (
            model.len_rhs if first_algebraic_index is None else first_algebraic_index
        )
        self._algebraic_expr = self.lower(model.concatenated_algebraic)
        self._algebraic_var_indices = list(range(start, model.len_rhs_and_alg))
        return self._algebraic_expr, self._algebraic_var_indices

    def outputs(self, output_variables, time_integral_vars: dict | None = None):
        """Lower the requested output variables and record their lengths.

        Lengths are flattened component counts, which is how the Rust core lays
        out the output rows; slicing by variable ordinal instead would drop a
        vector variable's tail and shift every variable after it.

        Parameters
        ----------
        output_variables : list of str
            Variable names to lower.
        time_integral_vars : dict, optional
            Map of name to :class:`pybamm.ProcessedVariableTimeIntegral`. For a
            listed name the integrand's ``sum_node`` is lowered, leaving the
            postfix summation to run after the solve. Omit to reject such
            variables instead.

        Returns
        -------
        tuple
            ``(exprs, lengths)`` in ``output_variables`` order.

        Raises
        ------
        :class:`pybamm.SolverError`
            If a time-integral variable is requested but unsupported, or if a
            variable is a tensor field.
        """
        time_integral_vars = time_integral_vars or {}
        for name in output_variables:
            time_integral = time_integral_vars.get(name)
            if time_integral is not None:
                symbol = time_integral.sum_node
            else:
                symbol = self.model.get_processed_variable_or_event(name)
            if isinstance(symbol, pybamm.TensorField):
                raise pybamm.SolverError(
                    f"Output variable '{name}' is a tensor field, which cannot be "
                    "read back from a Solution. Request a scalar component with "
                    "pybamm.Component instead."
                )
            self._output_exprs.append(self.lower(symbol))
            shape = getattr(symbol, "shape", ())
            self._output_lens.append(int(np.prod(shape)) if shape else 1)
        return self._output_exprs, self._output_lens

    def termination_events(self):
        """Lower every termination event expression, for root-finding."""
        self._event_exprs = [
            self.lower(event.expression)
            for event in self.model.events
            if event.event_type == pybamm.EventType.TERMINATION
        ]
        return self._event_exprs

    def rhs_evaluator(self):
        """Build an evaluator for the differential block alone, over this graph.

        The consistent initialisation of ``ydot0`` needs the ``len_rhs`` rows that
        the concatenated residual does not expose on a DAE, so the right-hand side
        gets its own root here rather than its own graph -- every node the residual
        already allocated is reused.

        Returns
        -------
        :class:`pybamm.solvers.rust_evaluator.RustEvaluator`
            Callable ``(t, y, p)`` returning the right-hand side as a column.
        """
        from pybamm.solvers.rust_evaluator import RustEvaluator

        compiled = self.graph.compile(
            self.lower(self.model.concatenated_rhs),
            name="RHS",
            n_states=self.model.len_rhs_and_alg,
        )
        return RustEvaluator(compiled, "func")

    def bind_generic_evaluators(self, rust_model) -> None:
        """Serve the model's backend-agnostic evaluator slots from this lowering.

        ``BaseSolver.set_up`` leaves ``rhs_eval`` and ``terminate_events_eval``
        unset on the native path, so the shared helpers that read them
        (consistent initialisation, the event-violation check, event attribution)
        are served from here instead of from a graph per expression. The event
        evaluators are views onto ``rust_model``'s own event tapes -- the roots
        its fused root-finding tape is built from -- so a Python-side event value
        cannot drift from the one the integrator roots on.

        Parameters
        ----------
        rust_model : CompiledModel
            The model compiled from this lowering. Its ``events`` follow
            :meth:`termination_events` order, which is ``model.events`` order.
        """
        from pybamm.solvers.rust_evaluator import RustEvaluator

        self.model.rhs_eval = self.rhs_evaluator()
        self.model.terminate_events_eval = [
            RustEvaluator(compiled, "func") for compiled in rust_model.events
        ]

    def sensitivity_indices(self, sensitivity_names):
        """Map sensitivity parameter names to graph input indices.

        The Rust core differentiates against positions in the global input array,
        so names absent from the solve's inputs are dropped rather than indexed.

        Parameters
        ----------
        sensitivity_names : list of str
            Requested parameter names, typically ``model.calculate_sensitivities``.

        Returns
        -------
        tuple
            ``(indices, names)``, filtered to the inputs actually supplied and
            kept in the requested order.
        """
        indices: list[int] = []
        names: list[str] = []
        for name in sensitivity_names or []:
            if name in self.input_name_order:
                indices.append(self.input_name_order.index(name))
                names.append(name)
        self._sens_param_indices = indices
        return indices, names

    @property
    def n_inputs(self) -> int:
        """Packed width of every input registered, including any added while
        lowering, which may exceed the inputs the solver passes."""
        return self.graph.n_inputs()

    def mass_matrix_csr(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Return the mass matrix as the ``(data, indptr, indices)`` CSR triple."""
        mass = self.model.mass_matrix.entries
        return (
            mass.data.astype(np.float64),
            mass.indptr.astype(np.int64),
            mass.indices.astype(np.int64),
        )

    def compile(self):
        """Compile everything lowered so far into a ``CompiledModel``.

        Returns
        -------
        CompiledModel
            The compiled model.

        Raises
        ------
        :class:`pybamm.SolverError`
            If :meth:`state_residual` has not been called.
        """
        from pybamm.rust import CompiledModel

        if self._state_residual is None:
            raise pybamm.SolverError(
                "Cannot compile a Rust model before lowering its state residual; "
                "call state_residual() first."
            )
        data, indptr, indices = self.mass_matrix_csr()
        return CompiledModel.from_expr(
            self.graph,
            self._state_residual,
            data,
            indptr,
            indices,
            n_inputs=self.n_inputs,
            sens_param_indices=self._sens_param_indices,
            output_exprs=self._output_exprs,
            algebraic_expr=self._algebraic_expr,
            algebraic_variable_indices=self._algebraic_var_indices,
            event_exprs=self._event_exprs,
        )
