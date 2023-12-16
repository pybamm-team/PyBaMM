#
# Casadi algebraic solver class
#
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
        Please consult `CasADi documentation <https://tinyurl.com/y7hrxm7d>`_ for
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

    def _integrate(self, model, t_eval, inputs_dict=None):
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
                "constraints": list(constraints[len_rhs:]),
            },
        )

        timer = pybamm.Timer()
        integration_time = 0
        for idx, t in enumerate(t_eval):
            # Solve
            try:
                timer.reset()
                y_alg_sol = roots(y0_alg, t)
                integration_time += timer.time()
                success = True
                message = None
                # Check final output
                y_sol = casadi.vertcat(y0_diff, y_alg_sol)
                fun = model.casadi_algebraic(t, y_sol, inputs)
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
                y0 = casadi.vertcat(y0_diff, y0_alg)
                # update solution array
                if y_alg is None:
                    y_alg = y_alg_sol
                else:
                    y_alg = casadi.horzcat(y_alg, y_alg_sol)
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
                    """
                    Could not find acceptable solution: solver terminated
                    successfully, but maximum solution error ({})
                    above tolerance ({})
                    """.format(
                        casadi.mmax(casadi.fabs(fun)), self.tol
                    )
                )

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
            sensitivities=explicit_sensitivities,
        )
        sol.integration_time = integration_time
        return sol
