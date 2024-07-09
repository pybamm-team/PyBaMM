import casadi
import pybamm
import numpy as np


class CasadiAlgebraicSolver(pybamm.BaseSolver):
    """Solve a discretised model which contains only (time independent) algebraic
    equations using CasADi's root finding algorithm.
    Note: this solver could be extended for quasi-static models, or models in
    which the time derivative is manually discretised and results in a (possibly
    nonlinear) algebaric system at each time level.

    Parameters
    ----------
    tol : float, optional
        The tolerance for the solver (default is 1e-6).
    extra_options : dict, optional
        Any options to pass to the CasADi rootfinder.
        Please consult `CasADi documentation <https://web.casadi.org/python-api/#rootfinding>`_ for
        details.

    """

    def __init__(self, tol=1e-6, extra_options=None):
        super().__init__()
        self.tol = tol
        self.name = "CasADi algebraic solver"
        self.algebraic_solver = True
        self.extra_options = extra_options or {}
        pybamm.citations.register("Andersson2019")

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, value):
        self._tol = value

    def _integrate_batch(self, model, t_eval, y0, y0S, inputs_list, inputs):
        # The casadi algebraic solver can read rhs equations, but leaves them unchanged
        # i.e. the part of the solution vector that corresponds to the differential
        # equations will be equal to the initial condition provided. This allows this
        # solver to be used for initialising the DAE solvers
        nstates = (
            model.len_rhs + model.len_rhs_sens + model.len_alg + model.len_alg_sens
        )
        len_rhs = model.len_rhs + model.len_rhs_sens
        len_alg = model.len_alg + model.len_alg_sens
        batch_size = len(inputs_list)
        if model.rhs == {}:
            y0_diff_list = [casadi.DM() for _ in range(batch_size)]
            y0_alg = y0
        else:
            y0_diff_list = [
                y0[i * nstates : i * nstates + len_rhs] for i in range(batch_size)
            ]
            y0_alg_list = [
                y0[i * nstates + len_rhs : (i + 1) * nstates] for i in range(batch_size)
            ]
            y0_alg = casadi.vertcat(*y0_alg_list)

        y_sol = None

        # Set up
        t_sym = casadi.MX.sym("t")
        y_alg_sym_list = [
            casadi.MX.sym(f"y_alg{i}", len_alg) for i in range(batch_size)
        ]
        y_alg_sym = casadi.vertcat(*y_alg_sym_list)

        # interleave the differential and algebraic parts
        y_sym = casadi.vertcat(
            *[val for pair in zip(y0_diff_list, y_alg_sym_list) for val in pair]
        )

        alg = model.casadi_algebraic(t_sym, y_sym, inputs)

        # Set constraints vector in the casadi format
        # Constrain the unknowns. 0 (default): no constraint on ui, 1: ui >= 0.0,
        # -1: ui <= 0.0, 2: ui > 0.0, -2: ui < 0.0.
        constraints = np.zeros_like(model.bounds[0], dtype=int)
        # If the lower bound is positive then the variable must always be positive
        constraints[model.bounds[0] >= 0] = 1
        # If the upper bound is negative then the variable must always be negative
        constraints[model.bounds[1] <= 0] = -1

        # Set up rootfinder
        roots = casadi.rootfinder(
            "roots",
            "newton",
            dict(x=y_alg_sym, p=t_sym, g=alg),
            {
                **self.extra_options,
                "abstol": self.tol,
                "constraints": list(constraints[len_rhs:]) * batch_size,
            },
        )

        timer = pybamm.Timer()
        integration_time = 0
        for _, t in enumerate(t_eval):
            # Solve
            try:
                timer.reset()
                y_alg_sol = roots(y0_alg, t)
                integration_time += timer.time()
                success = True
                message = None
                # Check final output
                y_alg_sol_list = casadi.vertsplit(y_alg_sol, len_alg)
                yi_sol = casadi.vertcat(
                    *[val for pair in zip(y0_diff_list, y_alg_sol_list) for val in pair]
                )
                fun = model.casadi_algebraic(t, yi_sol, inputs)
            except RuntimeError as err:
                success = False
                message = err.args[0]
                fun = None

            # If there are no symbolic inputs, check the function is below the tol
            # Skip this check if there are symbolic inputs
            if success and (
                not any(np.isnan(fun)) and np.all(casadi.fabs(fun) < self.tol)
            ):
                # update initial guess for the next iteration
                y0_alg = y_alg_sol
                # update solution array
                if y_sol is None:
                    y_sol = yi_sol
                else:
                    y_sol = casadi.horzcat(y_sol, yi_sol)
            elif not success:
                raise pybamm.SolverError(
                    f"Could not find acceptable solution: {message}"
                )
            elif any(np.isnan(fun)):
                raise pybamm.SolverError(
                    "Could not find acceptable solution: solver returned NaNs"
                )
            else:
                raise pybamm.SolverError(
                    f"""
                    Could not find acceptable solution: solver terminated
                    successfully, but maximum solution error ({casadi.mmax(casadi.fabs(fun))})
                    above tolerance ({self.tol})
                    """
                )

        # Return solution object (no events, so pass None to t_event, y_event)

        try:
            explicit_sensitivities = bool(model.calculate_sensitivities)
        except AttributeError:
            explicit_sensitivities = False

        sols = pybamm.Solution.from_concatenated_state(
            [t_eval],
            y_sol,
            model,
            inputs_list,
            termination="final time",
            sensitivities=explicit_sensitivities,
        )
        for sol in sols:
            sol.integration_time = integration_time
        return sols
