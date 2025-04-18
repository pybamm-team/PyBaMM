import casadi
import pybamm
import numpy as np


class CasadiAlgebraicSolver(pybamm.BaseSolver):
    """Solve a discretised model which contains only (time independent) algebraic
    equations using CasADi's root finding algorithm.

    Note: this solver could be extended for quasi-static models, or models in
    which the time derivative is manually discretised and results in a (possibly
    nonlinear) algebraic system at each time level.

    Parameters
    ----------
    tol : float, optional
        The tolerance for the maximum residual error (default is 1e-6).
    step_tol : float, optional
        The tolerance for the maximum step size (default is 1e-4).
    extra_options : dict, optional
        Any options to pass to the CasADi rootfinder.
        Please consult `CasADi documentation <https://web.casadi.org/python-api/#rootfinding>`_ for
        details. By default:

        .. code-block:: python

            extra_options = {
                # Whether to throw an error if the solver fails to converge.
                "error_on_fail": False,
                # Verbosity level
                "verbose": False,
            }
    """

    def __init__(self, tol=1e-6, step_tol=1e-4, extra_options=None):
        super().__init__()
        default_extra_options = {"error_on_fail": False, "verbose": False}
        extra_options = extra_options or {}
        self.extra_options = extra_options | default_extra_options
        self.step_tol = step_tol
        self.tol = tol
        self.name = "CasADi algebraic solver"
        self._algebraic_solver = True
        pybamm.citations.register("Andersson2019")

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, value):
        self._tol = value

    @property
    def step_tol(self):
        return self._step_tol

    @step_tol.setter
    def step_tol(self, value):
        self._step_tol = value

    def _integrate(self, model, t_eval, inputs_dict=None, t_interp=None):
        """
        Calculate the solution of the algebraic equations through root-finding

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : :class:`numpy.array`, size (k,)
            The times at which to compute the solution
        inputs_dict : dict, optional
            Any input parameters to pass to the model when solving.
        """
        # Record whether there are any symbolic inputs
        inputs_dict = inputs_dict or {}

        # Create casadi objects for the root-finder
        inputs = casadi.vertcat(*[v for v in inputs_dict.values()])

        y0 = model.y0

        # The casadi algebraic solver can read rhs equations, but leaves them unchanged
        # i.e. the part of the solution vector that corresponds to the differential
        # equations will be equal to the initial condition provided. This allows this
        # solver to be used for initialising the DAE solvers
        if model.rhs == {}:
            len_rhs = 0
            y0_diff = casadi.DM()
            y0_alg = y0
        else:
            # Check y0 to see if it includes sensitivities
            if model.len_rhs_and_alg == y0.shape[0]:
                len_rhs = model.len_rhs
            else:
                len_rhs = model.len_rhs + model.len_rhs_sens
            y0_diff = y0[:len_rhs]
            y0_alg = y0[len_rhs:]

        y_alg = None

        # Set up
        t_sym = casadi.MX.sym("t")
        y_alg_sym = casadi.MX.sym("y_alg", y0_alg.shape[0])
        y_sym = casadi.vertcat(y0_diff, y_alg_sym)

        alg = model.casadi_algebraic(t_sym, y_sym, inputs)

        # Set constraints vector in the casadi format
        # Constrain the unknowns. 0 (default): no constraint on ui, 1: ui >= 0.0,
        # -1: ui <= 0.0, 2: ui > 0.0, -2: ui < 0.0.
        model_alg_lb = model.bounds[0][len_rhs:]
        model_alg_ub = model.bounds[1][len_rhs:]
        constraints = np.zeros_like(model_alg_lb, dtype=int)
        # If the lower bound is positive then the variable must always be positive
        constraints[model_alg_lb >= 0] = 1
        # If the upper bound is negative then the variable must always be negative
        constraints[model_alg_ub <= 0] = -1

        # Set up rootfinder
        roots = casadi.rootfinder(
            "roots",
            "newton",
            dict(x=y_alg_sym, p=t_sym, g=alg),
            {
                **self.extra_options,
                "abstol": self.tol,
                "abstolStep": self.step_tol,
                "constraints": list(constraints),
            },
        )
        # As a fallback, if the initial rootfind fails, try again with three
        # Newton iterations of successively larger learning rates

        timer = pybamm.Timer()
        integration_time = 0
        for t in t_eval:
            # Solve
            success = False
            try:
                timer.reset()
                y_alg_sol = roots(y0_alg, t)
                integration_time += timer.time()

                solver_stats = roots.stats()
                # Check final output
                y_sol = casadi.vertcat(y0_diff, y_alg_sol)
                fun = model.casadi_algebraic(t, y_sol, inputs)

                # Casadi does not give us the value of the final residuals or step
                # norm, however, if it returns a success flag and there are no NaNs or Infs
                # in the solution, then it must be successful
                y_is_finite = np.isfinite(max_abs(y_alg_sol))
                fun_is_finite = np.isfinite(max_abs(fun))

                success = solver_stats["success"] and y_is_finite and fun_is_finite
            except RuntimeError as err:
                success = False
                message = err.args[0]
                raise pybamm.SolverError(
                    f"Could not find acceptable solution: {message}"
                ) from err

            if not success:
                if any(np.isnan(fun)):
                    reason = "solver returned NaNs"
                elif any(np.isinf(fun)):
                    reason = "solver returned Infs"
                else:
                    reason = "solver terminated unsuccessfully"

                raise pybamm.SolverError(
                    f"Could not find acceptable solution: {reason} and maximum solution error ({casadi.mmax(casadi.fabs(fun))}) "
                    f"above tolerance ({self.tol})"
                )

            # update initial guess for the next iteration
            y0_alg = y_alg_sol
            y0 = casadi.vertcat(y0_diff, y0_alg)
            # update solution array
            if y_alg is None:
                y_alg = y_alg_sol
            else:
                y_alg = casadi.horzcat(y_alg, y_alg_sol)

        # Concatenate differential part
        y_diff = casadi.horzcat(*[y0_diff] * len(t_eval))
        y_sol = casadi.vertcat(y_diff, y_alg)

        # Return solution object (no events, so pass None to t_event, y_event)

        try:
            explicit_sensitivities = bool(model.calculate_sensitivities)
        except AttributeError:
            explicit_sensitivities = False

        sol = pybamm.Solution(
            [t_eval],
            y_sol,
            model,
            inputs_dict,
            termination="final time",
            all_sensitivities=explicit_sensitivities,
        )
        sol.integration_time = integration_time
        return sol


def max_abs(x):
    return np.linalg.norm(x, ord=np.inf)
