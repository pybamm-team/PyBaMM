#
# Algebraic solver class
#
import pybamm
import numpy as np
from scipy import optimize


class AlgebraicSolver(pybamm.BaseSolver):
    """Solve a discretised model which contains only (time independent) algebraic
    equations using a root finding algorithm.
    Note: this solver could be extended for quasi-static models, or models in
    which the time derivative is manually discretised and results in a (possibly
    nonlinear) algebaric system at each time level.

    Parameters
    ----------
    method : str, optional
        The method to use to solve the system (default is "lm")
    tolerance : float, optional
        The tolerance for the solver (default is 1e-6).
    """

    def __init__(self, method="lm", tol=1e-6):
        super().__init__(method=method)
        self.tol = tol
        self.name = "Algebraic solver ({})".format(method)
        self.algebraic_solver = True
        pybamm.citations.register("virtanen2020scipy")

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
        algebraic = model.algebraic_eval
        jac = model.jacobian_eval
        y0 = model.y0

        y = np.empty((len(y0), len(t_eval)))

        for idx, t in enumerate(t_eval):

            def root_fun(y):
                "Evaluates algebraic using y"
                out = algebraic(t, y)
                pybamm.logger.debug(
                    "Evaluating algebraic equations, L2-norm is {}".format(
                        np.linalg.norm(out)
                    )
                )
                return out

            if model.jacobian_eval is not None:

                def jac(y):
                    return model.jacobian_eval(t, y)

            else:
                jac = None

            sol = optimize.root(
                root_fun, y0, method=self.method, tol=self.tol, jac=jac,
            )

            if sol.success and np.all(sol.fun < self.tol * len(sol.x)):
                # update initial guess for the next iteration
                y0 = sol.x
                # update solution array
                y[:, idx] = y0
            elif not sol.success:
                raise pybamm.SolverError(
                    "Could not find acceptable solution: {}".format(sol.message)
                )
            else:
                raise pybamm.SolverError(
                    """
                    Could not find acceptable solution: solver terminated
                    successfully, but maximum solution error ({}) above tolerance ({})
                    """.format(
                        np.max(sol.fun), self.tol * len(sol.x)
                    )
                )

        # Return solution object (no events, so pass None to t_event, y_event)
        return pybamm.Solution(t_eval, y, termination="success")

