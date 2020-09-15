#
# CasADi Solver class
#
import casadi
import pybamm
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq


class CasadiSolver(pybamm.BaseSolver):
    """Solve a discretised model, using CasADi.

    **Extends**: :class:`pybamm.BaseSolver`

    Parameters
    ----------
    mode : str
            How to solve the model (default is "safe"):

            - "fast": perform direct integration, without accounting for events. \
            Recommended when simulating a drive cycle or other simulation where \
            no events should be triggered.
            - "safe": perform step-and-check integration in global steps of size \
            dt_max, checking whether events have been triggered. Recommended for \
            simulations of a full charge or discharge.
            - "safe without grid": perform step-and-check integration step-by-step. \
            Takes more steps than "safe" mode, but doesn't require creating the grid \
            each time, so may be faster. Experimental only.
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).
    root_method : str or pybamm algebraic solver class, optional
        The method to use to find initial conditions (for DAE solvers).
        If a solver class, must be an algebraic solver class.
        If "casadi",
        the solver uses casadi's Newton rootfinding algorithm to find initial
        conditions. Otherwise, the solver uses 'scipy.optimize.root' with method
        specified by 'root_method' (e.g. "lm", "hybr", ...)
    root_tol : float, optional
        The tolerance for root-finding. Default is 1e-6.
    max_step_decrease_counts : float, optional
        The maximum number of times step size can be decreased before an error is
        raised. Default is 5.
    dt_max : float, optional
        The maximum global step size (in seconds) used in "safe" mode. If None
        the default value corresponds to a non-dimensional time of 0.01
        (i.e. ``0.01 * model.timescale_eval``).
    extra_options_setup : dict, optional
        Any options to pass to the CasADi integrator when creating the integrator.
        Please consult `CasADi documentation <https://tinyurl.com/y5rk76os>`_ for
        details. Some typical options:

        - "max_num_steps": Maximum number of integrator steps

    extra_options_call : dict, optional
        Any options to pass to the CasADi integrator when calling the integrator.
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
        dt_max=None,
        extra_options_setup=None,
        extra_options_call=None,
    ):
        super().__init__("problem dependent", rtol, atol, root_method, root_tol)
        if mode in ["safe", "fast", "safe without grid"]:
            self.mode = mode
        else:
            raise ValueError(
                "invalid mode '{}'. Must be 'safe', for solving with events, "
                "'fast', for solving quickly without events, or 'safe without grid' "
                "(experimental)".format(mode)
            )
        self.max_step_decrease_count = max_step_decrease_count
        self.dt_max = dt_max

        self.extra_options_setup = extra_options_setup or {}
        self.extra_options_call = extra_options_call or {}

        self.name = "CasADi solver with '{}' mode".format(mode)

        # Initialize
        self.integrators = {}
        self.integrator_specs = {}

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
        # Record whether there are any symbolic inputs
        inputs = inputs or {}
        has_symbolic_inputs = any(isinstance(v, casadi.MX) for v in inputs.values())

        # convert inputs to casadi format
        inputs = casadi.vertcat(*[x for x in inputs.values()])

        if has_symbolic_inputs:
            # Create integrator without grid to avoid having to create several times
            self.create_integrator(model, inputs)
            solution = self._run_integrator(model, model.y0, inputs, t_eval)
            solution.termination = "final time"
            return solution
        elif self.mode == "fast" or not model.events:
            if not model.events:
                pybamm.logger.info("No events found, running fast mode")
            # Create an integrator with the grid (we just need to do this once)
            self.create_integrator(model, inputs, t_eval)
            solution = self._run_integrator(model, model.y0, inputs, t_eval)
            solution.termination = "final time"
            return solution
        elif self.mode in ["safe", "safe without grid"]:
            y0 = model.y0
            if isinstance(y0, casadi.DM):
                y0 = y0.full().flatten()
            # Step-and-check
            t = t_eval[0]
            t_f = t_eval[-1]
            init_event_signs = np.sign(
                np.concatenate(
                    [event(t, y0, inputs) for event in model.terminate_events_eval]
                )
            )
            pybamm.logger.info("Start solving {} with {}".format(model.name, self.name))

            if self.mode == "safe without grid":
                # in "safe without grid" mode,
                # create integrator once, without grid,
                # to avoid having to create several times
                self.create_integrator(model, inputs)
                # Initialize solution
                solution = pybamm.Solution(np.array([t]), y0[:, np.newaxis])
                solution.solve_time = 0
            else:
                solution = None

            # Try to integrate in global steps of size dt_max. Note: dt_max must
            # be at least as big as the the biggest step in t_eval (multiplied
            # by some tolerance, here 1.01) to avoid an empty integration window below
            if self.dt_max:
                # Non-dimensionalise provided dt_max
                dt_max = self.dt_max / model.timescale_eval
            else:
                dt_max = 0.01
            dt_eval_max = np.max(np.diff(t_eval)) * 1.01
            dt_max = np.max([dt_max, dt_eval_max])
            while t < t_f:
                # Step
                solved = False
                count = 0
                dt = dt_max
                while not solved:
                    # Get window of time to integrate over (so that we return
                    # all the points in t_eval, not just t and t+dt)
                    t_window = np.concatenate(
                        ([t], t_eval[(t_eval > t) & (t_eval < t + dt)])
                    )
                    # Sometimes near events the solver fails between two time
                    # points in t_eval (i.e. no points t < t_i < t+dt for t_i
                    # in t_eval), so we simply integrate from t to t+dt
                    if len(t_window) == 1:
                        t_window = np.array([t, t + dt])

                    if self.mode == "safe":
                        # update integrator with the grid
                        self.create_integrator(model, inputs, t_window)
                    # Try to solve with the current global step, if it fails then
                    # halve the step size and try again.
                    try:
                        current_step_sol = self._run_integrator(
                            model, y0, inputs, t_window
                        )
                        solved = True
                    except pybamm.SolverError:
                        dt /= 2
                        # also reduce maximum step size for future global steps
                        dt_max = dt
                    count += 1
                    if count >= self.max_step_decrease_count:
                        raise pybamm.SolverError(
                            "Maximum number of decreased steps occurred at t={}. Try "
                            "solving the model up to this time only or reducing dt_max "
                            "(currently, dt_max={})."
                            "".format(
                                t * model.timescale_eval, dt_max * model.timescale_eval
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

                        def event_fun(t):
                            return event(t, y_sol(t), inputs)

                        if np.isnan(event_fun(current_step_sol.t[-1])[0]):
                            # bracketed search fails if f(a) or f(b) is NaN, so we
                            # need to find the times for which we can evaluate the event
                            times = [
                                t
                                for t in current_step_sol.t
                                if event_fun(t)[0] == event_fun(t)[0]
                            ]
                        else:
                            times = current_step_sol.t
                        # skip if sign hasn't changed
                        if np.sign(event_fun(times[0])) != np.sign(
                            event_fun(times[-1])
                        ):
                            t_events[i] = brentq(
                                lambda t: event_fun(t), times[0], times[-1]
                            )
                        else:
                            t_events[i] = np.nan

                    # t_event is the earliest event triggered
                    t_event = np.nanmin(t_events)
                    y_event = y_sol(t_event)

                    # solve again until the event time
                    # See comments above on creating t_window
                    t_window = np.concatenate(
                        ([t], t_eval[(t_eval > t) & (t_eval < t_event)])
                    )
                    if len(t_window) == 1:
                        t_window = np.array([t, t_event])

                    if self.mode == "safe":
                        self.create_integrator(model, inputs, t_window)
                    current_step_sol = self._run_integrator(model, y0, inputs, t_window)

                    # assign temporary solve time
                    current_step_sol.solve_time = np.nan
                    # append solution from the current step to solution
                    solution.append(current_step_sol)
                    solution.termination = "event"
                    solution.t_event = t_event
                    solution.y_event = y_event

                    break
                else:
                    # assign temporary solve time
                    current_step_sol.solve_time = np.nan
                    if solution is None:
                        solution = current_step_sol
                    else:
                        # append solution from the current step to solution
                        solution.append(current_step_sol)
                    # update time
                    t = t_window[-1]
                    # update y0
                    y0 = solution.y[:, -1]
            return solution

    def create_integrator(self, model, inputs, t_eval=None):
        """
        Method to create a casadi integrator object.
        If t_eval is provided, the integrator uses t_eval to make the grid.
        Otherwise, the integrator has grid [0,1].
        """
        # Use grid if t_eval is given
        use_grid = not (t_eval is None)
        # Only set up problem once
        if model in self.integrators:
            # If we're not using the grid, we don't need to change the integrator
            if use_grid is False:
                return self.integrators[model]
            # Otherwise, create new integrator with an updated grid
            else:
                method, problem, options = self.integrator_specs[model]
                options["grid"] = t_eval
                integrator = casadi.integrator("F", method, problem, options)
                self.integrators[model] = (integrator, use_grid)
                return integrator
        else:
            y0 = model.y0
            rhs = model.casadi_rhs
            algebraic = model.casadi_algebraic

            # When not in DEBUG mode (level=10), suppress warnings from CasADi
            if (
                pybamm.logger.getEffectiveLevel() == 10
                or pybamm.settings.debug_mode is True
            ):
                show_eval_warnings = True
            else:
                show_eval_warnings = False

            options = {
                **self.extra_options_setup,
                "reltol": self.rtol,
                "abstol": self.atol,
                "show_eval_warnings": show_eval_warnings,
            }

            # set up and solve
            t = casadi.MX.sym("t")
            p = casadi.MX.sym("p", inputs.shape[0])
            y_diff = casadi.MX.sym("y_diff", rhs(0, y0, p).shape[0])

            if use_grid is False:
                # rescale time
                t_min = casadi.MX.sym("t_min")
                t_max = casadi.MX.sym("t_max")
                t_scaled = t_min + (t_max - t_min) * t
                # add time limits as inputs
                p_with_tlims = casadi.vertcat(p, t_min, t_max)
            else:
                options.update({"grid": t_eval, "output_t0": True})
                # Set dummy parameters for consistency with rescaled time
                t_max = 1
                t_min = 0
                t_scaled = t
                p_with_tlims = p

            problem = {"t": t, "x": y_diff, "p": p_with_tlims}
            if algebraic(0, y0, p).is_empty():
                method = "cvodes"
                # rescale rhs by (t_max - t_min)
                problem.update({"ode": (t_max - t_min) * rhs(t_scaled, y_diff, p)})
            else:
                method = "idas"
                y_alg = casadi.MX.sym("y_alg", algebraic(0, y0, p).shape[0])
                y_full = casadi.vertcat(y_diff, y_alg)
                # rescale rhs by (t_max - t_min)
                problem.update(
                    {
                        "ode": (t_max - t_min) * rhs(t_scaled, y_full, p),
                        "z": y_alg,
                        "alg": algebraic(t_scaled, y_full, p),
                    }
                )
            integrator = casadi.integrator("F", method, problem, options)
            self.integrator_specs[model] = method, problem, options
            self.integrators[model] = (integrator, use_grid)
            return integrator

    def _run_integrator(self, model, y0, inputs, t_eval):
        integrator, use_grid = self.integrators[model]
        len_rhs = model.concatenated_rhs.size
        y0_diff = y0[:len_rhs]
        y0_alg = y0[len_rhs:]
        try:
            # Try solving
            if use_grid is True:
                # Call the integrator once, with the grid
                sol = integrator(
                    x0=y0_diff, z0=y0_alg, p=inputs, **self.extra_options_call
                )
                y_sol = np.concatenate([sol["xf"].full(), sol["zf"].full()])
                return pybamm.Solution(t_eval, y_sol)
            else:
                # Repeated calls to the integrator
                x = y0_diff
                z = y0_alg
                y_diff = x
                y_alg = z
                for i in range(len(t_eval) - 1):
                    t_min = t_eval[i]
                    t_max = t_eval[i + 1]
                    inputs_with_tlims = casadi.vertcat(inputs, t_min, t_max)
                    sol = integrator(
                        x0=x, z0=z, p=inputs_with_tlims, **self.extra_options_call
                    )
                    x = sol["xf"]
                    z = sol["zf"]
                    y_diff = casadi.horzcat(y_diff, x)
                    if not z.is_empty():
                        y_alg = casadi.horzcat(y_alg, z)
                if z.is_empty():
                    return pybamm.Solution(t_eval, y_diff)
                else:
                    y_sol = casadi.vertcat(y_diff, y_alg)
                    return pybamm.Solution(t_eval, y_sol)
        except RuntimeError as e:
            # If it doesn't work raise error
            raise pybamm.SolverError(e.args[0])
