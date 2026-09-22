"""Shared lowering of a discretised model into a Rust ``CompiledModel``."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pybamm


def rust_graph_with_inputs(model: pybamm.BaseModel, input_names):
    """Build an ``ExprGraph`` with ``input_names`` registered in stacking order.

    Registration precedes any lowering so graph input indices match that order.
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

    Every part shares one ``ExprGraph``; only the key order of ``inputs_dict`` is used.
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
        """Lower the right-hand side, concatenated with the algebraic block."""
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

    def algebraic_block(self):
        """Lower the algebraic sub-block used by the Newton initialisation."""
        model = self.model
        if model.len_alg == 0:
            return None, []
        self._algebraic_expr = self.lower(model.concatenated_algebraic)
        self._algebraic_var_indices = list(range(model.len_rhs, model.len_rhs_and_alg))
        return self._algebraic_expr, self._algebraic_var_indices

    def outputs(self, output_variables, time_integral_vars: dict | None = None):
        """Lower the requested output variables; lengths are flattened component counts.

        A name in ``time_integral_vars`` lowers its ``sum_node``, summed post-solve.
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
        """Lower every termination event expression, for root-finding.

        Raises
        ------
        pybamm.SolverError
            If an event expression is not scalar.
        """
        events = [
            event
            for event in self.model.events
            if event.event_type == pybamm.EventType.TERMINATION
        ]
        for event in events:
            if event.expression.size != 1:
                raise pybamm.SolverError(
                    f"Termination event '{event.name}' has shape "
                    f"{event.expression.shape}, but a root function must be scalar. "
                    "Reduce it with pybamm.min or pybamm.max."
                )
        self._event_exprs = [self.lower(event.expression) for event in events]
        return self._event_exprs

    def rhs_evaluator(self):
        """Build an evaluator for the differential block alone, over this graph."""
        from pybamm.solvers.rust_evaluator import RustEvaluator

        compiled = self.graph.compile(
            self.lower(self.model.concatenated_rhs),
            name="RHS",
            n_states=self.model.len_rhs_and_alg,
        )
        return RustEvaluator(compiled, "func")

    def algebraic_evaluators(self):
        """Build ``func`` and ``jac`` evaluators for the algebraic block alone."""
        from pybamm.solvers.rust_evaluator import RustEvaluator

        compiled = self.graph.compile(
            self.lower(self.model.concatenated_algebraic),
            name="algebraic",
            n_states=self.model.len_rhs_and_alg,
        )
        return RustEvaluator(compiled, "func"), RustEvaluator(compiled, "jac")

    def bind_generic_evaluators(self, rust_model, root_method=None) -> None:
        """Serve ``rhs_eval`` and ``terminate_events_eval`` from this lowering.

        A DAE also gets ``algebraic_eval`` and ``jac_algebraic_eval`` when
        ``root_method`` will read them. Event evaluators view ``rust_model``'s own
        event tapes, so they match the integrator's roots.
        """
        from pybamm.solvers.rust_evaluator import RustEvaluator

        self.model.rhs_eval = self.rhs_evaluator()
        self.model.terminate_events_eval = [
            RustEvaluator(compiled, "func") for compiled in rust_model.events
        ]
        # NonlinearSolver lowers its own system; other root methods read algebraic_eval
        reads_algebraic = root_method is not None and not isinstance(
            root_method, pybamm.NonlinearSolver
        )
        if reads_algebraic and self.model.len_alg > 0:
            self.model.algebraic_eval, self.model.jac_algebraic_eval = (
                self.algebraic_evaluators()
            )

    def sensitivity_indices(self, sensitivity_names):
        """Map sensitivity names to graph input indices, dropping unsupplied inputs."""
        indices: list[int] = []
        names: list[str] = []
        for name in sensitivity_names or []:
            # The graph numbers the inputs; the solver's input order only filters
            index = self.graph.input_index(name)
            if index is not None and name in self.input_name_order:
                indices.append(index)
                names.append(name)
        self._sens_param_indices = indices
        return indices, names

    @property
    def n_inputs(self) -> int:
        """Packed width of every registered input, including any added by lowering."""
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
        """Compile everything lowered so far into a ``CompiledModel``."""
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
