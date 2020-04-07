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

    def __init__(self, method="lm", tol=1e-6, **extra_options):
        super().__init__()
        self.tol = tol
        self.name = "CasADi algebraic solver"
        self.algebraic_solver = True
        self.extra_options = extra_options
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
            Any input parameters to pass to the model when solving. If any input
            parameters that are present in the model are missing from "inputs", then
            the solution will consist of `ProcessedSymbolicVariable` objects, which must
            be provided with inputs to obtain their value.
        """
        y0 = model.y0

        y = None

        # Set up
        # Record whether there are any symbolic inputs
        has_symbolic_inputs = any(isinstance(v, casadi.MX) for v in inputs.values())

        # Create casadi objects for the root-finder
        inputs = casadi.vertcat(*[x for x in inputs.values()])
        t_sym = casadi.MX.sym("t")
        y_sym = casadi.MX.sym("y_alg", y0.shape[0])
        p_sym = casadi.MX.sym("p", inputs.shape[0])

        t_p_sym = casadi.vertcat(t_sym, p_sym)
        alg = model.casadi_algebraic(t_sym, y_sym, p_sym)

        # Set up rootfinder
        roots = casadi.rootfinder(
            "roots",
            "newton",
            dict(x=y_sym, p=t_p_sym, g=alg),
            {**self.extra_options, "abstol": self.tol},
        )
        for idx, t in enumerate(t_eval):
            # Evaluate algebraic with new t and previous y0, if it's already close
            # enough then keep it
            # We can't do this if there are symbolic inputs
            if has_symbolic_inputs is False and np.all(
                abs(model.casadi_algebraic(t, y0, inputs).full()) < self.tol
            ):
                pybamm.logger.debug(
                    "Keeping same solution at t={}".format(t * model.timescale_eval)
                )
                if y is None:
                    y = y0
                else:
                    y = casadi.horzcat(y, y0)
            # Otherwise calculate new y_sol
            else:
                t_inputs = casadi.vertcat(t, inputs)
                # Solve
                try:
                    y_sol = roots(y0, t_inputs)
                    success = True
                    message = None
                    # Check final output
                    fun = model.casadi_algebraic(t, y_sol, inputs)
                except RuntimeError as err:
                    success = False
                    message = err.args[0]
                    fun = None

                # If there are no symbolic inputs, check the function is below the tol
                # Skip this check if there are symbolic inputs
                if success and (
                    has_symbolic_inputs is True or np.all(casadi.fabs(fun) < self.tol)
                ):
                    # update initial guess for the next iteration
                    y0 = y_sol
                    # update solution array
                    if y is None:
                        y = y_sol
                    else:
                        y = casadi.horzcat(y, y_sol)
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
