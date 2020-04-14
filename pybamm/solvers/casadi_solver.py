#
# CasADi Solver class
#
import casadi
import pybamm
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from contextlib import contextmanager
import sys
import os


@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


class CasadiSolver(pybamm.BaseSolver):
    """Solve a discretised model, using CasADi.

    **Extends**: :class:`pybamm.BaseSolver`

    Parameters
    ----------
    method : str, optional
        The method to use for solving the system ('cvodes', for ODEs, or 'idas', for
        DAEs). Default is 'idas'.
    mode : str
            How to solve the model (default is "safe"):

            - "fast": perform direct integration, without accounting for events. \
            Recommended when simulating a drive cycle or other simulation where \
            no events should be triggered.
            - "safe": perform step-and-check integration, checking whether events have \
            been triggered. Recommended for simulations of a full charge or discharge.
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).
    root_method : str, optional
        The method to use for finding consistend initial conditions. Default is 'lm'.
    root_tol : float, optional
        The tolerance for root-finding. Default is 1e-6.
    max_step_decrease_counts : float, optional
        The maximum number of times step size can be decreased before an error is
        raised. Default is 5.
    extra_options : keyword arguments, optional
        Any extra keyword-arguments; these are passed directly to the CasADi integrator.
        Please consult `CasADi documentation <https://tinyurl.com/y5rk76os>`_ for
        details.

    """

    def __init__(
        self,
        mode="safe",
        rtol=1e-6,
        atol=1e-6,
        root_method="casadi",
        root_tol=1e-6,
        max_step_decrease_count=5,
        **extra_options,
    ):
        super().__init__("problem dependent", rtol, atol, root_method, root_tol)
        if mode in ["safe", "fast"]:
            self.mode = mode
        else:
            raise ValueError(
                """
                invalid mode '{}'. Must be either 'safe', for solving with events,
                or 'fast', for solving quickly without events""".format(
                    mode
                )
            )
        self.max_step_decrease_count = max_step_decrease_count
        self.extra_options = extra_options
        self.name = "CasADi solver with '{}' mode".format(mode)

        # Initialize
        self.problems = {}
        self.options = {}
        self.methods = {}

        pybamm.citations.register("Andersson2019")

    def _integrate(self, model, t_eval, inputs=None):
        """
        Solve a DAE model defined by residuals with initial conditions y0.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : numeric type
            The times at which to compute the solution
        inputs : dict, optional
            Any external variables or input parameters to pass to the model when solving
        """
        inputs = inputs or {}
        # convert inputs to casadi format
        inputs = casadi.vertcat(*[x for x in inputs.values()])

        if len(model.rhs) == 0:
            # casadi solver won't allow solving algebraic model so we have to raise an
            # error here
            raise pybamm.SolverError(
                "Cannot use CasadiSolver to solve algebraic model, "
                "use CasadiAlgebraicSolver instead"
            )
        if self.mode == "fast":
            integrator = self.get_integrator(model, t_eval, inputs)
            solution = self._run_integrator(integrator, model, model.y0, inputs, t_eval)
            solution.termination = "final time"
            return solution
        elif not model.events:
            pybamm.logger.info("No events found, running fast mode")
            integrator = self.get_integrator(model, t_eval, inputs)
            solution = self._run_integrator(integrator, model, model.y0, inputs, t_eval)
            solution.termination = "final time"
            return solution
        elif self.mode == "safe":
            # Step-and-check
            t = t_eval[0]
            init_event_signs = np.sign(
                np.concatenate(
                    [
                        event(t, model.y0, inputs)
                        for event in model.terminate_events_eval
                    ]
                )
            )
            pybamm.logger.info("Start solving {} with {}".format(model.name, self.name))
            y0 = model.y0
            # Initialize solution
            solution = pybamm.Solution(np.array([t]), y0[:, np.newaxis])
            solution.solve_time = 0

            # Try to integrate in global steps of size dt_max (arbitrarily taken
            # to be half of the range of t_eval, up to a maximum of 1 hour)
            t_f = t_eval[-1]
            # dt_max must be at least as big as the the biggest step in t_eval
            # to avoid an empty integration window below
            dt_eval_max = np.max(np.diff(t_eval))
            dt_max = np.max([(t_f - t) / 2, dt_eval_max])
            # maximum dt_max corresponds to 1 hour
            t_one_hour = 3600 / model.timescale_eval
            dt_max = np.min([dt_max, t_one_hour])
            while t < t_f:
                # Step
                solved = False
                count = 0
                dt = dt_max
                while not solved:
                    # Get window of time to integrate over (so that we return
                    # all the point in t_eval, not just t and t+dt)
                    t_window = np.concatenate(
                        ([t], t_eval[(t_eval > t) & (t_eval < t + dt)])
                    )
                    integrator = self.get_integrator(model, t_window, inputs)
                    # Try to solve with the current global step, if it fails then
                    # halve the step size and try again.
                    try:
                        # When not in DEBUG mode (level=10), supress the output
                        # from the failed steps in the solver
                        if (
                            pybamm.logger.getEffectiveLevel() == 10
                            or pybamm.settings.debug_mode is True
                        ):
                            current_step_sol = self._run_integrator(
                                integrator, model, y0, inputs, t_window
                            )
                        else:
                            with suppress_stderr():
                                current_step_sol = self._run_integrator(
                                    integrator, model, y0, inputs, t_window
                                )
                        solved = True
                    except pybamm.SolverError:
                        dt /= 2
                        # also reduce maximum step size for future global steps
                        dt_max = dt
                    count += 1
                    if count >= self.max_step_decrease_count:
                        raise pybamm.SolverError(
                            """
                            Maximum number of decreased steps occurred at t={}. Try
                            solving the model up to this time only
                            """.format(
                                t
                            )
                        )
                # Check most recent y to see if any events have been crossed
                new_event_signs = np.sign(
                    np.concatenate(
                        [
                            event(t, current_step_sol.y[:, -1], inputs)
                            for event in model.terminate_events_eval
                        ]
                    )
                )
                # Exit loop if the sign of an event changes
                # Locate the event time using a root finding algorithm and
                # event state using interpolation. The solution is then truncated
                # so that only the times up to the event are returned
                if (new_event_signs != init_event_signs).any():
                    # get the index of the events that have been crossed
                    event_ind = np.where(new_event_signs != init_event_signs)[0]
                    active_events = [model.terminate_events_eval[i] for i in event_ind]

                    # create interpolant to evaluate y in the current integration
                    # window
                    y_sol = interp1d(current_step_sol.t, current_step_sol.y)

                    # loop over events to compute the time at which they were triggered
                    t_events = [None] * len(active_events)
                    for i, event in enumerate(active_events):
                        t_events[i] = brentq(
                            lambda t: event(t, y_sol(t), inputs),
                            current_step_sol.t[0],
                            current_step_sol.t[-1],
                        )
                    # t_event is the earliest event triggered
                    t_event = np.min(t_events)
                    y_event = y_sol(t_event)

                    # return truncated solution
                    t_truncated = current_step_sol.t[current_step_sol.t < t_event]
                    y_trunctaed = current_step_sol.y[:, 0 : len(t_truncated)]
                    truncated_step_sol = pybamm.Solution(t_truncated, y_trunctaed)
                    # assign temporary solve time
                    truncated_step_sol.solve_time = np.nan
                    # append solution from the current step to solution
                    solution.append(truncated_step_sol)

                    solution.termination = "event"
                    solution.t_event = t_event
                    solution.y_event = y_event
                    break
                else:
                    # assign temporary solve time
                    current_step_sol.solve_time = np.nan
                    # append solution from the current step to solution
                    solution.append(current_step_sol)
                    # update time
                    t = t_window[-1]
                    # update y0
                    y0 = solution.y[:, -1]
            return solution

    def get_integrator(self, model, t_eval, inputs):
        # Only set up problem once
        if model not in self.problems:
            y0 = model.y0
            rhs = model.casadi_rhs
            algebraic = model.casadi_algebraic

            options = {
                "grid": t_eval,
                "reltol": self.rtol,
                "abstol": self.atol,
                "output_t0": True,
                "max_num_steps": self.max_steps,
            }

            # set up and solve
            t = casadi.MX.sym("t")
            p = casadi.MX.sym("p", inputs.shape[0])
            y_diff = casadi.MX.sym("y_diff", rhs(t_eval[0], y0, p).shape[0])
            problem = {"t": t, "x": y_diff, "p": p}
            if algebraic(t_eval[0], y0, p).is_empty():
                method = "cvodes"
                problem.update({"ode": rhs(t, y_diff, p)})
            else:
                options["calc_ic"] = True
                method = "idas"
                y_alg = casadi.MX.sym("y_alg", algebraic(t_eval[0], y0, p).shape[0])
                y_full = casadi.vertcat(y_diff, y_alg)
                problem.update(
                    {
                        "z": y_alg,
                        "ode": rhs(t, y_full, p),
                        "alg": algebraic(t, y_full, p),
                    }
                )
            self.problems[model] = problem
            self.options[model] = options
            self.methods[model] = method
        else:
            # problem stays the same
            # just update options
            self.options[model]["grid"] = t_eval
        return casadi.integrator(
            "F", self.methods[model], self.problems[model], self.options[model]
        )

    def _run_integrator(self, integrator, model, y0, inputs, t_eval):
        rhs_size = model.rhs_eval(t_eval[0], y0, inputs).shape[0]
        y0_diff, y0_alg = np.split(y0, [rhs_size])
        try:
            # Try solving
            sol = integrator(x0=y0_diff, z0=y0_alg, p=inputs, **self.extra_options)
            y_values = np.concatenate([sol["xf"].full(), sol["zf"].full()])
            return pybamm.Solution(t_eval, y_values)
        except RuntimeError as e:
            # If it doesn't work raise error
            raise pybamm.SolverError(e.args[0])
