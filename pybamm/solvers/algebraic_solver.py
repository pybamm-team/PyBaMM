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
        self.name = "Algebraic solver ({})".format(method)

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
        pybamm.logger.info("Start solving {} with {}".format(model.name, self.name))

        # Set up
        timer = pybamm.Timer()
        start_time = timer.time()
        concatenated_algebraic, jac = self.set_up(model)
        set_up_time = timer.time() - start_time

        # Create function to evaluate algebraic
        def algebraic(y):
            return concatenated_algebraic.evaluate(0, y, known_evals={})[0][:, 0]

        # Create function to evaluate jacobian
        if jac is not None:

            def jacobian(y):
                # Note: we only use this solver for time independent algebraic
                # systems, so jac is arbitrarily evaluated at t=0. Also, needs
                # to be converted from sparse to dense, so in very large
                # algebraic models it may be best to switch use_jacobian to False
                # by default.
                return jac.evaluate(0, y, known_evals={})[0].toarray()

        else:
            jacobian = None

        # Use "initial conditions" set in model as initial guess
        y0_guess = model.concatenated_initial_conditions

        # Solve
        solve_start_time = timer.time()
        pybamm.logger.info("Calling root finding algorithm")
        solution = self.root(algebraic, y0_guess, jacobian=jacobian)
        solution.model = model

        # Assign times
        solution.solve_time = timer.time() - solve_start_time
        solution.set_up_time = set_up_time

        pybamm.logger.info("Finish solving {}".format(model.name))
        pybamm.logger.info(
            "Set-up time: {}, Solve time: {}, Total time: {}".format(
                timer.format(solution.set_up_time),
                timer.format(solution.solve_time),
                timer.format(solution.total_time),
            )
        )
        return solution

    def root(self, algebraic, y0_guess, jacobian=None):
        """
        Calculate the solution of the algebraic equations through root-finding

        Parameters
        ----------
        algebraic : method
            Function that takes in y and returns the value of the algebraic
            equations
        y0_guess : array-like
            Array of the user's guess for the solution, used to initialise
            the root finding algorithm
        jacobian : method, optional
            A function that takes in t and y and returns the Jacobian. If
            None, the solver will approximate the Jacobian if required.
        """

        def root_fun(y0):
            "Evaluates algebraic using y0"
            out = algebraic(y0)
            pybamm.logger.debug(
                "Evaluating algebraic equations, L2-norm is {}".format(
                    np.linalg.norm(out)
                )
            )
            return out

        if jacobian:
            sol = optimize.root(
                root_fun, y0_guess, method=self.method, tol=self.tol, jac=jacobian
            )
        else:
            sol = optimize.root(root_fun, y0_guess, method=self.method, tol=self.tol)

        if sol.success and np.all(sol.fun < self.tol * len(sol.x)):
            termination = "success"
            # Return solution object (no events, so pass None to t_event, y_event)
            return pybamm.Solution([0], sol.x[:, np.newaxis], termination=termination)
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
            # Create Jacobian from concatenated algebraic
            y = pybamm.StateVector(
                slice(0, np.size(model.concatenated_initial_conditions))
            )
            # set up Jacobian object, for re-use of dict
            jacobian = pybamm.Jacobian()
            pybamm.logger.info("Calculating jacobian")
            jac = jacobian.jac(concatenated_algebraic, y)
            model.jacobian = jac
            model.jacobian_algebraic = jac

            if model.use_simplify:
                pybamm.logger.info("Simplifying jacobian")
                jac = simp.simplify(jac)

            if model.convert_to_format == "python":
                pybamm.logger.info("Converting jacobian to python")
                jac = pybamm.EvaluatorPython(jac)

        else:
            jac = None

        if model.convert_to_format == "python":
            pybamm.logger.info("Converting algebraic to python")
            concatenated_algebraic = pybamm.EvaluatorPython(concatenated_algebraic)

        return concatenated_algebraic, jac
