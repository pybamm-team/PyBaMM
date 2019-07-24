#
# Algebraic solver class
#
import pybamm
import numpy as np
from scipy import optimize


class AlgebraicSolver(object):
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
        self.method = method
        self.tol = tol

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, value):
        self._tol = value

    def solve(self, model):
        """Calculate the solution of the model.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must only contain algebraic
            equations.

        """
        pybamm.logger.info("Start solving {}".format(model.name))

        # Set up
        timer = pybamm.Timer()
        start_time = timer.time()
        concatenated_algebraic, jac = self.set_up(model)
        set_up_time = timer.time() - start_time

        def algebraic(y):
            return concatenated_algebraic.evaluate(0, y, known_evals={})[0][:, 0]

        def jac_fun(y):
            return jac(0, y)

        def root_fun(y0):
            "Evaluates algebraic using y0"
            out = algebraic(y0)
            pybamm.logger.debug(
                "Evaluating algebraic equations, L2-norm is {}".format(
                    np.linalg.norm(out)
                )
            )
            return out

        # Use "initial conditions" set in model as initial guess
        y0_guess = model.concatenated_initial_conditions

        # Solve
        solve_start_time = timer.time()
        pybamm.logger.info("Calling root finding algorithm")
        if jac:
            sol = optimize.root(
                root_fun, y0_guess, method=self.root_method, tol=self.root_tol, jac=jac_fun,
            )
        else:
            sol = optimize.root(
                root_fun, y0_guess, method=self.root_method, tol=self.root_tol
            )

        if sol.success and np.all(sol.fun < self.root_tol * len(sol.x)):
            termination = "success"
            solution = pybamm.Solution(0, sol.x, termination)
        elif not sol.success:
            raise pybamm.SolverError(
                "Could not find solution: {}".format(sol.message)
            )
        else:
            raise pybamm.SolverError(
                """
                Could not find acceptable solution: solver terminated
                successfully, but maximum solution error ({}) above tolerance ({})
                """.format(
                    np.max(sol.fun), self.root_tol * len(sol.x)
                )
            )
        # Assign times
        solution.solve_time = timer.time() - solve_start_time
        solution.total_time = timer.time() - start_time
        solution.set_up_time = set_up_time

        pybamm.logger.info("Finish solving {}".format(model.name))
        return solution

    def set_up(self, model):
        """Unpack model, perform checks, simplify and calculate jacobian.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions

        Returns
        -------
        concatenated_algebraic : :class:`pybamm.Concatenation`
            Algebraic equations, which should evaluate to zero
        jac : :class:`pybamm.SparseStack`
            Jacobian matrix for the differential and algebraic equations

        Raises
        ------
        :class:`pybamm.SolverError`
            If the model contains any time derivatives, i.e. rhs equations (in
            which case an ODE or DAE solver should be used instead)
        """
        if len(model.rhs) > 0:
            raise pybamm.SolverError(
                """Cannot use algebraic solver to solve model with time derivatives"""
            )

        # create simplified algebraic expressions
        concatenated_algebraic = model.concatenated_algebraic

        if model.use_simplify:
            # set up simplification object, for re-use of dict
            simp = pybamm.Simplification()
            pybamm.logger.info("Simplifying algebraic")
            concatenated_algebraic = simp.simplify(concatenated_algebraic)

        if model.use_jacobian:
            y = pybamm.StateVector(
                slice(0, np.size(model.concatenated_initial_conditions))
            )
            pybamm.logger.info("Calculating jacobian")
            jac = concatenated_algebraic.jac(y)

            if model.use_simplify:
                pybamm.logger.info("Simplifying jacobian")
                jac = jac.simplify()

            if model.use_to_python:
                pybamm.logger.info("Converting jacobian to python")
                jac = pybamm.EvaluatorPython(jac)

        else:
            jac = None

        if model.use_to_python:
            pybamm.logger.info("Converting algebraic to python")
            concatenated_algebraic = pybamm.EvaluatorPython(concatenated_algebraic)

        return concatenated_algebraic, jac
