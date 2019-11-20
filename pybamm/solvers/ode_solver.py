#
# Base solver class
#
import casadi
import pybamm
import numpy as np

from .base_solver import add_external


class OdeSolver(pybamm.BaseSolver):
    """Solve a discretised model.

    Parameters
    ----------
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).
    """

    def __init__(self, method=None, rtol=1e-6, atol=1e-6):
        super().__init__(method, rtol, atol)
        self.name = "Base ODE solver"

    def compute_solution(self, model, t_eval):
        """Calculate the solution of the model at specified times.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        t_eval : numeric type
            The times at which to compute the solution

        """
        timer = pybamm.Timer()

        self.dydt.set_pad_ext(self.y_pad, self.y_ext)
        for evnt in self.event_funs:
            evnt.set_pad_ext(self.y_pad, self.y_ext)
        if self.jacobian:
            self.jacobian.set_pad_ext(self.y_pad, self.y_ext)

        solve_start_time = timer.time()
        pybamm.logger.info("Calling ODE solver")
        solution = self.integrate(
            self.dydt,
            self.y0,
            t_eval,
            events=self.event_funs,
            mass_matrix=model.mass_matrix.entries,
            jacobian=self.jacobian,
        )

        solve_time = timer.time() - solve_start_time

        # Identify the event that caused termination
        termination = self.get_termination_reason(solution, self.events)

        return solution, solve_time, termination

    def set_up(self, model):
        """Unpack model, perform checks, simplify and calculate jacobian.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions

        Raises
        ------
        :class:`pybamm.SolverError`
            If the model contains any algebraic equations (in which case a DAE solver
            should be used instead)

        """
        # Check for algebraic equations
        if len(model.algebraic) > 0:
            raise pybamm.SolverError(
                """Cannot use ODE solver to solve model with DAEs"""
            )

        # create simplified rhs and event expressions
        concatenated_rhs = model.concatenated_rhs
        events = model.events

        if model.use_simplify:
            # set up simplification object, for re-use of dict
            simp = pybamm.Simplification()
            # create simplified rhs and event expressions
            pybamm.logger.info("Simplifying RHS")
            concatenated_rhs = simp.simplify(concatenated_rhs)

            pybamm.logger.info("Simplifying events")
            events = {name: simp.simplify(event) for name, event in events.items()}

        y0 = model.concatenated_initial_conditions[:, 0]

        if model.use_jacobian:
            # Create Jacobian from concatenated rhs
            y = pybamm.StateVector(slice(0, np.size(y0)))
            # set up Jacobian object, for re-use of dict
            jacobian = pybamm.Jacobian()
            pybamm.logger.info("Calculating jacobian")
            jac_rhs = jacobian.jac(concatenated_rhs, y)
            model.jacobian = jac_rhs
            model.jacobian_rhs = jac_rhs

            if model.use_simplify:
                pybamm.logger.info("Simplifying jacobian")
                jac_rhs = simp.simplify(jac_rhs)

            if model.convert_to_format == "python":
                pybamm.logger.info("Converting jacobian to python")
                jac_rhs = pybamm.EvaluatorPython(jac_rhs)
        else:
            jac_rhs = None

        if model.convert_to_format == "python":
            pybamm.logger.info("Converting RHS to python")
            concatenated_rhs = pybamm.EvaluatorPython(concatenated_rhs)
            pybamm.logger.info("Converting events to python")
            events = {
                name: pybamm.EvaluatorPython(event) for name, event in events.items()
            }

        # Create event-dependent function to evaluate events
        def get_event_class(event):
            return EvalEvent(event.evaluate)

        # Create function to evaluate jacobian
        if jac_rhs is not None:
            jacobian = Jacobian(jac_rhs.evaluate)
        else:
            jacobian = None

        # Add the solver attributes
        # Note: these are the (possibly) converted to python version rhs, algebraic
        # etc. The expression tree versions of these are attributes of the model
        self.y0 = y0
        self.dydt = Dydt(model, concatenated_rhs.evaluate)
        self.events = events
        self.event_funs = [get_event_class(event) for event in events.values()]
        self.jacobian = jacobian

    def set_up_casadi(self, model):
        """Convert model to casadi format and use their inbuilt functionalities.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions

        Raises
        ------
        :class:`pybamm.SolverError`
            If the model contains any algebraic equations (in which case a DAE solver
            should be used instead)

        """
        # Check for algebraic equations
        if len(model.algebraic) > 0:
            raise pybamm.SolverError(
                """Cannot use ODE solver to solve model with DAEs"""
            )

        y0 = model.concatenated_initial_conditions[:, 0]

        t_casadi = casadi.MX.sym("t")
        y_casadi = casadi.MX.sym("y", len(y0))

        if self.y_pad is not None:
            y_ext = casadi.MX.sym("y_ext", len(self.y_pad))
            y_casadi_w_ext = casadi.vertcat(y_casadi, y_ext)
        else:
            y_casadi_w_ext = y_casadi

        pybamm.logger.info("Converting RHS to CasADi")
        concatenated_rhs = model.concatenated_rhs.to_casadi(t_casadi, y_casadi_w_ext)
        pybamm.logger.info("Converting events to CasADi")
        casadi_events = {
            name: event.to_casadi(t_casadi, y_casadi_w_ext)
            for name, event in model.events.items()
        }

        # Create function to evaluate rhs
        concatenated_rhs_fn = casadi.Function(
            "rhs", [t_casadi, y_casadi_w_ext], [concatenated_rhs]
        )

        # Create event-dependent function to evaluate events
        def get_event_class(event):
            casadi_event_fn = casadi.Function(
                "event", [t_casadi, y_casadi_w_ext], [event]
            )
            return EvalEvent(casadi_event_fn)

        # Create function to evaluate jacobian
        if model.use_jacobian:
            pybamm.logger.info("Calculating jacobian")
            casadi_jac = casadi.jacobian(concatenated_rhs, y_casadi)
            casadi_jac_fn = casadi.Function(
                "jacobian", [t_casadi, y_casadi_w_ext], [casadi_jac]
            )

            jacobian = JacobianCasadi(casadi_jac_fn)

        else:
            jacobian = None

        # Add the solver attributes
        self.y0 = y0
        self.dydt = DydtCasadi(model, concatenated_rhs_fn)
        self.events = model.events
        self.event_funs = [get_event_class(event) for event in casadi_events.values()]
        self.jacobian = jacobian

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


# Set up caller classes outside of the solver object to allow pickling
class Dydt:
    "Returns information about time derivatives at time t and state y"

    def __init__(self, model, concatenated_rhs_fn):
        self.model = model
        self.concatenated_rhs_fn = concatenated_rhs_fn
        self.y_pad = None
        self.y_ext = None

    def set_pad_ext(self, y_pad, y_ext):
        self.y_pad = y_pad
        self.y_ext = y_ext

    def __call__(self, t, y):
        pybamm.logger.debug("Evaluating RHS for {} at t={}".format(self.model.name, t))
        y = y[:, np.newaxis]
        y = add_external(y, self.y_pad, self.y_ext)
        dy = self.concatenated_rhs_fn(t, y, known_evals={})[0]
        return dy[:, 0]


class DydtCasadi(Dydt):
    "Returns information about time derivatives at time t and state y, with CasADi"

    def __call__(self, t, y):
        pybamm.logger.debug("Evaluating RHS for {} at t={}".format(self.model.name, t))
        y = y[:, np.newaxis]
        y = add_external(y, self.y_pad, self.y_ext)
        dy = self.concatenated_rhs_fn(t, y).full()
        return dy[:, 0]


class EvalEvent:
    "Returns information about events at time t and state y"

    def __init__(self, event_fn):
        self.event_fn = event_fn
        self.y_pad = None
        self.y_ext = None

    def set_pad_ext(self, y_pad, y_ext):
        self.y_pad = y_pad
        self.y_ext = y_ext

    def __call__(self, t, y):
        y = y[:, np.newaxis]
        y = add_external(y, self.y_pad, self.y_ext)
        return self.event_fn(t, y)


class Jacobian:
    "Returns information about the jacobian at time t and state y"

    def __init__(self, jac_fn):
        self.jac_fn = jac_fn
        self.y_pad = None
        self.y_ext = None

    def set_pad_ext(self, y_pad, y_ext):
        self.y_pad = y_pad
        self.y_ext = y_ext

    def __call__(self, t, y):
        y = y[:, np.newaxis]
        y = add_external(y, self.y_pad, self.y_ext)
        return self.jac_fn(t, y, known_evals={})[0]


class JacobianCasadi(Jacobian):
    "Returns information about the jacobian at time t and state y, with CasADi"

    def __call__(self, t, y):
        y = y[:, np.newaxis]
        y = add_external(y, self.y_pad, self.y_ext)
        return self.jac_fn(t, y)
