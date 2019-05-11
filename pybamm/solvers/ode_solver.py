#
# Base solver class
#
import pybamm
import numpy as np


class OdeSolver(pybamm.BaseSolver):
    """Solve a discretised model.

    Parameters
    ----------
    tolerance : float, optional
        The tolerance for the solver (default is 1e-8).
    """

    def __init__(self, tol=1e-8):
        super().__init__(tol)

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

        # create simplified rhs and event expressions
        concatenated_rhs = model.concatenated_rhs.simplify()
        events = [event.simplify() for event in model.events]

        def dydt(t, y):
            return concatenated_rhs.evaluate(t, y, known_evals={})[0]

        # Create event-dependent function to evaluate events
        def event_fun(event):
            def eval_event(t, y):
                return event.evaluate(t, y)

            return eval_event

        events = [event_fun(event) for event in events]

        y0 = model.concatenated_initial_conditions

        if model.use_jacobian:
            # Create Jacobian from simplified rhs
            y = pybamm.StateVector(slice(0, np.size(y0)))
            jac_rhs = concatenated_rhs.jac(y).simplify()

            def jacobian(t, y):
                return jac_rhs.evaluate(t, y, known_evals={})[0]

        else:
            jacobian = None

        self.t, self.y = self.integrate(
            dydt,
            y0,
            t_eval,
            events=events,
            mass_matrix=model.mass_matrix.entries,
            jacobian=jacobian,
        )

    def integrate(
        self, derivs, y0, t_eval, events=None, mass_matrix=None, jacobian=None
    ):
        """
        Solve a model defined by dydt with initial conditions y0.

        Parameters
        ----------
        derivs : method
            A function that takes in t and y and returns the time-derivative dydt
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
            A function that takes in t and y and returns the Jacobian
        """
        raise NotImplementedError
