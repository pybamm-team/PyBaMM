from __future__ import annotations

import numpy as np

import pybamm


def _check_residual(solution, tol):
    tolerance = max(1e-5, 10 * tol)
    solutions = solution if isinstance(solution, list) else [solution]
    for result in solutions:
        for index, (model, times, states) in enumerate(
            zip(result.all_models, result.all_ts, result.all_ys, strict=True)
        ):
            inputs = (
                result.all_inputs_stacked[index]
                if model.uses_stacked_inputs
                else result.all_inputs[index]
            )
            for column, time in enumerate(times):
                residual = model.algebraic_eval(time, states[:, column], inputs)
                max_residual = np.max(np.abs(np.asarray(residual)), initial=0)
                if not np.isfinite(max_residual) or max_residual > tolerance:
                    raise pybamm.SolverError(
                        "Electrode SOH solution does not satisfy its equations "
                        f"(maximum residual {max_residual:.2e}, tolerance {tolerance:.2e})"
                    )


class _ElectrodeSOHAlgebraicSolver(pybamm.AlgebraicSolver):
    def __init__(self, method="lm", tol=1e-6, extra_options=None):
        super().__init__(method=method, tol=tol, extra_options=extra_options)
        if method == "minimize":
            # SciPy's default minimizers do not use the root-finding xtol option.
            self.extra_options.pop("xtol")

    def _integrate_single(self, model, t_eval, inputs_dict, y0):
        solution = super()._integrate_single(model, t_eval, inputs_dict, y0)
        _check_residual(solution, self.tol)
        return solution


class _ElectrodeSOHSimulationSolver(pybamm.BaseSolver):
    """Check the result of a supplied solver before caching it."""

    def __init__(self, solver, tol):
        super().__init__()
        self._solver = solver
        self._tol = (
            solver.tol
            if isinstance(solver, pybamm.AlgebraicSolver | pybamm.CasadiAlgebraicSolver)
            else tol
        )

    def solve(self, *args, **kwargs):
        solution = self._solver.solve(*args, **kwargs)
        _check_residual(solution, self._tol)
        return solution


def _get_esoh_simulation_solver(solver, tol):
    if type(solver) is pybamm.CompositeSolver:
        return pybamm.CompositeSolver(
            [
                _get_esoh_simulation_solver(sub_solver, tol)
                for sub_solver in solver.sub_solvers
            ]
        )
    if type(solver) in (_ElectrodeSOHAlgebraicSolver, _ElectrodeSOHSimulationSolver):
        return solver
    return _ElectrodeSOHSimulationSolver(solver, tol)
