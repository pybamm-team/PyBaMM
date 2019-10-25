#
# CasADi Solver class
#
import casadi
import pybamm
import numpy as np


class CasadiSolver(pybamm.DaeSolver):
    """Solve a discretised model, using CasADi.

    Parameters
    ----------
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).

    **Extends**: :class:`pybamm.DaeSolver`
    """

    def __init__(
        self,
        method="idas",
        rtol=1e-6,
        atol=1e-6,
        root_method="lm",
        root_tol=1e-6,
        max_steps=1000,
        **extra_options,
    ):
        super().__init__(method, rtol, atol, root_method, root_tol, max_steps)
        self.extra_options = extra_options

    def compute_solution(self, model, t_eval):
        """Calculate the solution of the model at specified times. In this class, we
        overwrite the behaviour of :class:`pybamm.DaeSolver`, since CasADi requires
        slightly different syntax.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        t_eval : numeric type
            The times at which to compute the solution

        """
        timer = pybamm.Timer()

        solve_start_time = timer.time()
        pybamm.logger.info("Calling DAE solver")
        solution = self.integrate_casadi(
            self.casadi_problem, self.y0, t_eval, mass_matrix=model.mass_matrix.entries
        )
        solve_time = timer.time() - solve_start_time

        # Identify the event that caused termination
        termination = self.get_termination_reason(solution, self.events)

        return solution, solve_time, termination

    def integrate_casadi(self, problem, y0, t_eval, mass_matrix=None):
        """
        Solve a DAE model defined by residuals with initial conditions y0.

        Parameters
        ----------
        residuals : method
            A function that takes in t, y and ydot and returns the residuals of the
            equations
        y0 : numeric type
            The initial conditions
        t_eval : numeric type
            The times at which to compute the solution
        mass_matrix : array_like, optional
            The (sparse) mass matrix for the chosen spatial method. This is only passed
            to check that the mass matrix is diagonal with 1s for the odes and 0s for
            the algebraic equations, as CasADi does not allow to pass mass matrices.
        """
        options = {
            "grid": t_eval,
            "reltol": self.rtol,
            "abstol": self.atol,
            "output_t0": True,
        }
        options.update(self.extra_options)
        if self.method == "idas":
            options["calc_ic"] = True

        # set up and solve
        integrator = casadi.integrator("F", self.method, problem, options)
        try:
            # Try solving
            len_rhs = problem["x"].size()[0]
            y0_diff, y0_alg = np.split(y0, [len_rhs])
            sol = integrator(x0=y0_diff, z0=y0_alg)
            y_values = np.concatenate([sol["xf"].full(), sol["zf"].full()])
            return pybamm.Solution(t_eval, y_values, None, None, "final time")
        except RuntimeError as e:
            # If it doesn't work raise error
            raise pybamm.SolverError(e.args[0])

    def calculate_consistent_initial_conditions(
        self, rhs, algebraic, y0_guess, jac=None
    ):
        "No need to calculate initial conditions separately with this solver"
        return y0_guess
