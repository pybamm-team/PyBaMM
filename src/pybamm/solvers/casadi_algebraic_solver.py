import casadi
import pybamm
import numpy as np
import numbers


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
                # Whether to show warnings when evaluating the model
                "show_eval_warnings": False,
            }
    """

    def __init__(self, tol=1e-6, step_tol=1e-4, extra_options=None):
        super().__init__()
        default_extra_options = {
            "error_on_fail": False,
            "verbose": False,
            "show_eval_warnings": False,
        }
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

    def set_up_root_solver(self, model, inputs_dict, t_eval):
        """Create and return a CasADi rootfinder object. The parameter argument to the
        rootfinder is the concatenated time, differential states, and flattened inputs.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        inputs_dict : dict
            Dictionary of inputs.
        t_eval : :class:`numpy.array`, size (k,)
            The times at which to compute the solution (not used).

        Returns
        -------
        None
            The rootfinder function is stored in the model as `algebraic_root_solver`.
        """
        pybamm.logger.info(f"Start building {self.name}")
        y0 = model.y0

        # The casadi algebraic solver can read rhs equations, but leaves them unchanged
        # i.e. the part of the solution vector that corresponds to the differential
        # equations will be equal to the initial condition provided. This allows this
        # solver to be used for initialising the DAE solvers
        if model.rhs == {}:
            len_rhs = 0
        elif model.len_rhs_and_alg == y0.shape[0]:
            len_rhs = model.len_rhs
        else:
            len_rhs = model.len_rhs + model.len_rhs_sens

        len_alg = y0.shape[0] - len_rhs

        # Set up symbolic variables
        t_sym = casadi.MX.sym("t")
        y_diff_sym = casadi.MX.sym("y_diff", len_rhs)
        y_alg_sym = casadi.MX.sym("y_alg", len_alg)
        y_sym = casadi.vertcat(y_diff_sym, y_alg_sym)

        # Create parameter dictionary
        inputs_casadi = {}
        for name, value in inputs_dict.items():
            if isinstance(value, numbers.Number):
                inputs_casadi[name] = casadi.MX.sym(name)
            else:
                inputs_casadi[name] = casadi.MX.sym(name, value.shape[0])
        inputs_sym = casadi.vertcat(*[p for p in inputs_casadi.values()])

        p_sym = casadi.vertcat(
            t_sym,
            y_diff_sym,
            inputs_sym,
        )

        alg = model.casadi_algebraic(t_sym, y_sym, inputs_sym)

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
        model.algebraic_root_solver = casadi.rootfinder(
            "roots",
            "newton",
            dict(x=y_alg_sym, p=p_sym, g=alg),
            {
                **self.extra_options,
                "abstol": self.tol,
                "abstolStep": self.step_tol,
                "constraints": list(constraints),
            },
        )

        pybamm.logger.info(f"Finish building {self.name}")

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
            len_rhs = model.len_rhs
            y0_diff = y0[:len_rhs]
            y0_alg = y0[len_rhs:]

        y_alg = None

        # Set the parameter vector
        p = np.empty(1 + len_rhs + inputs.shape[0])
        # Set the time
        p[0] = 0
        # Set the differential states portion of the parameter vector
        p[1 : 1 + len_rhs] = np.asarray(y0_diff).ravel()
        # Set the inputs portion of the parameter vector
        p[1 + len_rhs :] = np.asarray(inputs).ravel()

        if getattr(model, "algebraic_root_solver", None) is None:
            self.set_up_root_solver(model, inputs_dict, t_eval)
        roots = model.algebraic_root_solver

        timer = pybamm.Timer()
        integration_time = 0
        for t in t_eval:
            # Set the time for the parameter vector
            p[0] = t

            # Solve
            success = False
            try:
                timer.reset()
                y_alg_sol = roots(y0_alg, p)
                integration_time += timer.time()

                # Check final output
                y_sol = casadi.vertcat(y0_diff, y_alg_sol)
                fun = model.casadi_algebraic(t, y_sol, inputs)
                # Casadi does not give us the value of the final residuals or step
                # norm, however, if it returns a success flag and there are no NaNs or Infs
                # in the solution, then it must be successful
                y_is_finite = np.isfinite(max_abs(y_alg_sol))
                fun_is_finite = np.isfinite(max_abs(fun))

                solver_stats = roots.stats()
                success = solver_stats["success"] and y_is_finite and fun_is_finite
            except RuntimeError as err:
                success = False
                message = err.args[0]
                raise pybamm.SolverError(
                    f"Could not find acceptable solution: {message}"
                ) from err

            if not success:
                if any(~np.isfinite(fun)):
                    reason = "solver returned NaNs or Infs"
                else:
                    reason = (
                        f"solver terminated unsuccessfully and maximum solution error ({casadi.mmax(casadi.fabs(fun))}) "
                        f"above tolerance ({self.tol})"
                    )

                raise pybamm.SolverError(
                    f"Could not find acceptable solution: {reason}"
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
        sol = pybamm.Solution(
            [t_eval],
            y_sol,
            model,
            inputs_dict,
            termination="final time",
        )
        sol.integration_time = integration_time
        return sol


def max_abs(x):
    return np.linalg.norm(x, ord=np.inf)
