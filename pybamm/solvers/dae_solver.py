#
# Base solver class
#
import pybamm
import numpy as np
from scipy import optimize


class DaeSolver(pybamm.BaseSolver):
    """Solve a discretised model.

    Parameters
    ----------
    tolerance : float, optional
        The tolerance for the solver (default is 1e-8).
    root_method : str, optional
        The method to use to find initial conditions (default is "lm")
    tolerance : float, optional
        The tolerance for the initial-condition solver (default is 1e-8).
    """

    def __init__(self, tol=1e-8, root_method="lm", root_tol=1e-6):
        super().__init__(tol)
        self.root_method = root_method
        self.root_tol = root_tol

    @property
    def root_method(self):
        return self._root_method

    @root_method.setter
    def root_method(self, method):
        self._root_method = method

    @property
    def root_tol(self):
        return self._root_tol

    @root_tol.setter
    def root_tol(self, tol):
        self._root_tol = tol

    def solve(self, model, t_eval):
        """Calculate the solution of the model at specified times.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        t_eval : numeric type
            The times at which to compute the solution

        """
        pybamm.logger.info("Start solving {}".format(model.name))

        # create simplified rhs algebraic and event expressions
        concatenated_rhs = model.concatenated_rhs
        concatenated_algebraic = model.concatenated_algebraic
        events = model.events
        if model.use_simplify:
            concatenated_rhs = concatenated_rhs.simplify()
            concatenated_algebraic = concatenated_algebraic.simplify()
            events = [event.simplify() for event in events]

        if model.use_jacobian:
            # Create Jacobian from simplified rhs
            y = pybamm.StateVector(
                slice(0, np.size(model.concatenated_initial_conditions)))
            jac_rhs = concatenated_rhs.jac(y)
            jac_algebraic = concatenated_algebraic.jac(y)

            if model.use_simplify:
                jac_rhs = jac_rhs.simplify()
                jac_algebraic = jac_algebraic.simplify()

            jac = pybamm.SparseStack(jac_rhs, jac_algebraic)

            if model.use_to_python:
                jac = pybamm.EvaluatorPython(jac)

            def jacobian(t, y):
                return jac.evaluate(t, y, known_evals={})[0]

        else:
            jacobian = None

        if model.use_to_python:
            concatenated_rhs = pybamm.EvaluatorPython(concatenated_rhs)
            concatenated_algebraic = pybamm.EvaluatorPython(concatenated_algebraic)
            # TODO: events don't work yet
            #events = [pybamm.EvaluatorPython(event) for event in events]

        def residuals(t, y, ydot):
            pybamm.logger.debug(
                "Evaluating residuals for {} at t={}".format(model.name, t)
            )
            y = y[:, np.newaxis]
            rhs_eval, known_evals = concatenated_rhs.evaluate(t, y, known_evals={})
            # reuse known_evals
            alg_eval = concatenated_algebraic.evaluate(t, y, known_evals=known_evals)[0]
            # turn into 1D arrays
            rhs_eval = rhs_eval[:, 0]
            alg_eval = alg_eval[:, 0]
            return np.concatenate((rhs_eval - ydot[: rhs_eval.shape[0]], alg_eval))

        def rhs(t, y):
            return concatenated_rhs.evaluate(t, y, known_evals={})[0][:, 0]

        def algebraic(t, y):
            return concatenated_algebraic.evaluate(t, y, known_evals={})[0][:, 0]

        # Create event-dependent function to evaluate events
        def event_fun(event):
            def eval_event(t, y):
                return event.evaluate(t, y)

            return eval_event

        events = [event_fun(event) for event in events]

        y0 = self.calculate_consistent_initial_conditions(
            rhs, algebraic, model.concatenated_initial_conditions[:, 0]
        )

        self.t, self.y = self.integrate(
            residuals,
            y0,
            t_eval,
            events=events,
            mass_matrix=model.mass_matrix.entries,
            jacobian=jacobian,
        )

        pybamm.logger.info("Finish solving {}".format(model.name))

    def calculate_consistent_initial_conditions(self, rhs, algebraic, y0_guess):
        """
        Calculate consistent initial conditions for the algebraic equations through
        root-finding

        Parameters
        ----------
        rhs : method
            Function that takes in t and y and returns the value of the differential
            equations
        algebraic : method
            Function that takes in t and y and returns the value of the algebraic
            equations
        y0_guess : array-like
            Array of the user's guess for the initial conditions, used to initialise
            the root finding algorithm

        Returns
        -------
        y0_consistent : array-like, same shape as y0_guess
            Initial conditions that are consistent with the algebraic equations (roots
            of the algebraic equations)
        """
        pybamm.logger.info("Start calculating consistent initial conditions")

        # Split y0_guess into differential and algebraic
        len_rhs = rhs(0, y0_guess).shape[0]
        y0_diff, y0_alg_guess = np.split(y0_guess, [len_rhs])

        def root_fun(y0_alg):
            "Evaluates algebraic using y0_diff (fixed) and y0_alg (changed by algo)"
            y0 = np.concatenate([y0_diff, y0_alg])
            out = algebraic(0, y0)
            pybamm.logger.debug(
                "Evaluating algebraic equations at t=0, L2-norm is {}".format(
                    np.linalg.norm(out)
                )
            )
            return out

        # Find the values of y0_alg that are roots of the algebraic equations
        sol = optimize.root(
            root_fun, y0_alg_guess, method=self.root_method, tol=self.root_tol
        )
        # Return full set of consistent initial conditions (y0_diff unchanged)
        y0_consistent = np.concatenate([y0_diff, sol.x])

        if sol.success and np.all(sol.fun < self.root_tol):
            pybamm.logger.info("Finish calculating consistent initial conditions")
            return y0_consistent
        elif not sol.success:
            raise pybamm.SolverError(
                "Could not find consistent initial conditions: {}".format(sol.message)
            )
        else:
            raise pybamm.SolverError(
                "Could not find consistent initial conditions: "
                + "solver terminated successfully, but solution above tolerance"
            )

    def integrate(
        self, residuals, y0, t_eval, events=None, mass_matrix=None, jacobian=None
    ):
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
        events : method, optional
            A function that takes in t and y and returns conditions for the solver to
            stop
        mass_matrix : array_like, optional
            The (sparse) mass matrix for the chosen spatial method.
        jacobian : method, optional
            A function that takes in t, y and ydot and returns the Jacobian
        """
        raise NotImplementedError
