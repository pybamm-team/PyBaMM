#
# Solver class using Scipy's adaptive time stepper
#
import casadi
import pybamm

import scipy.integrate as it
import numpy as np


class ScipySolver(pybamm.BaseSolver):
    """Solve a discretised model, using scipy.integrate.solve_ivp.

    Parameters
    ----------
    method : str, optional
        The method to use in solve_ivp (default is "BDF")
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).
    extrap_tol : float, optional
        The tolerance to assert whether extrapolation occurs or not (default is 0).
    extra_options : dict, optional
        Any options to pass to the solver.
        Please consult `SciPy documentation
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_
        for details.
    """

    def __init__(
        self,
        method="BDF",
        rtol=1e-6,
        atol=1e-6,
        extrap_tol=None,
        extra_options=None,
    ):
        super().__init__(
            method=method,
            rtol=rtol,
            atol=atol,
            extrap_tol=extrap_tol,
        )
        self.ode_solver = True
        self.extra_options = extra_options or {}
        self.name = f"Scipy solver ({method})"
        pybamm.citations.register("Virtanen2020")

    def _integrate_batch(self, model, t_eval, y0, y0S, inputs_list, inputs):
        # Save inputs dictionary, and if necessary convert inputs to a casadi vector
        inputs_list = inputs_list or [{}]
        inputs = pybamm.BaseSolver._inputs_to_stacked_vect(
            inputs_list, model.convert_to_format
        )

        extra_options = {**self.extra_options, "rtol": self.rtol, "atol": self.atol}

        # Initial conditions
        if isinstance(y0, casadi.DM):
            y0 = y0.full()
        y0 = y0.flatten()

        # check for user-supplied Jacobian
        implicit_methods = ["Radau", "BDF", "LSODA"]
        if np.any([self.method in implicit_methods]):
            if model.jac_rhs_eval:

                def jacobian(t, y):
                    return model.jac_rhs_eval(t, y, inputs)

                extra_options.update({"jac": jacobian})

        # rhs equation
        if model.convert_to_format == "casadi":

            def rhs(t, y):
                return model.rhs_eval(t, y, inputs).full().reshape(-1)

        else:

            def rhs(t, y):
                return model.rhs_eval(t, y, inputs).reshape(-1)

        # make events terminal so that the solver stops when they are reached
        if model.terminate_events_eval:

            def event_wrapper(event):
                def event_fn(t, y):
                    return event(t, y, inputs)

                event_fn.terminal = True
                return event_fn

            events = [event_wrapper(event) for event in model.terminate_events_eval]
            extra_options.update({"events": events})

        timer = pybamm.Timer()
        sol = it.solve_ivp(
            rhs,
            (t_eval[0], t_eval[-1]),
            y0,
            t_eval=t_eval,
            method=self.method,
            dense_output=True,
            **extra_options,
        )
        integration_time = timer.time()

        if sol.success:
            # Set the reason for termination
            if sol.message == "A termination event occurred.":
                termination = "event"
                t_event = []
                for time in sol.t_events:
                    if time.size > 0:
                        t_event = np.append(t_event, np.max(time))
                t_event = np.array([np.max(t_event)])
                y_event = sol.sol(t_event)
            elif sol.message.startswith("The solver successfully reached the end"):
                termination = "final time"
                t_event = None
                y_event = np.array(None)
            solns = pybamm.Solution.from_concatenated_state(
                sol.t,
                sol.y,
                model,
                inputs_list,
                t_event,
                y_event,
                termination,
                sensitivities=bool(model.calculate_sensitivities),
            )
            for s in solns:
                s.integration_time = integration_time
            return solns
        else:
            raise pybamm.SolverError(sol.message)
