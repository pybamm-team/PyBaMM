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
    """

    def __init__(self, method="lm", tol=1e-6):
        super().__init__()
        self.tol = tol
        self.name = "CasADi algebraic solver"
        self.algebraic_solver = True
        pybamm.citations.register("Andersson2019")

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, value):
        self._tol = value

    def _integrate(self, model, t_eval, inputs=None):
        """
        Calculate the solution of the algebraic equations through root-finding

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : :class:`numpy.array`, size (k,)
            The times at which to compute the solution
        inputs : dict, optional
            Any input parameters to pass to the model when solving
        """
        y0 = model.y0

        y = np.empty((len(y0), len(t_eval)))

        # Set up
        u_stacked = casadi.vertcat(*[x for x in inputs.values()])
        t_sym = casadi.MX.sym("t")
        y_sym = casadi.MX.sym("y_alg", y0.shape[0])
        u_sym = casadi.MX.sym("u", u_stacked.shape[0])

        t_u_sym = casadi.vertcat(t_sym, u_sym)
        alg = model.casadi_algebraic(t_sym, y_sym, u_sym)

        # Set up rootfinder
        roots = casadi.rootfinder(
            "roots", "newton", dict(x=y_sym, p=t_u_sym, g=alg), {"abstol": self.tol},
        )
        for idx, t in enumerate(t_eval):
            # Evaluate algebraic with new t and previous y0, if it's already close
            # enough then keep it
            if np.all(abs(model.algebraic_eval(t, y0)) < self.tol):
                pybamm.logger.debug("Keeping same solution at t={}".format(t))
                y[:, idx] = y0
            # Otherwise calculate new y0
            else:
                t_u_stacked = casadi.vertcat(t, u_stacked)
                # Solve
                try:
                    y_sol = roots(y0, t_u_stacked).full().flatten()
                    success = True
                    message = None
                    # Check final output
                    fun = model.casadi_algebraic(t, y_sol, u_stacked)
                except RuntimeError as err:
                    success = False
                    message = err.args[0]
                    fun = None

                if success and np.all(casadi.fabs(fun) < self.tol):
                    # update initial guess for the next iteration
                    y0 = y_sol
                    # update solution array
                    y[:, idx] = y_sol
                elif not success:
                    raise pybamm.SolverError(
                        "Could not find acceptable solution: {}".format(message)
                    )
                else:
                    raise pybamm.SolverError(
                        """
                        Could not find acceptable solution: solver terminated
                        successfully, but maximum solution error ({})
                        above tolerance ({})
                        """.format(
                            casadi.mmax(fun), self.tol
                        )
                    )

        # Return solution object (no events, so pass None to t_event, y_event)
        return pybamm.Solution(t_eval, y, termination="success")

