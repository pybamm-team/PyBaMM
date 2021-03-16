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
            - "fast with events": perform direct integration of the whole timespan, \
            then go back and check where events were crossed. Experimental only.
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
    extrap_tol : float, optional
        The tolerance to assert whether extrapolation occurs or not. Default is 0.
    extra_options_setup : dict, optional
        Any options to pass to the CasADi integrator when creating the integrator.
        Please consult `CasADi documentation <https://tinyurl.com/y5rk76os>`_ for
        details. Some useful options:

        - "max_num_steps": Maximum number of integrator steps
        - "print_stats": Print out statistics after integration

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
        extrap_tol=0,
        extra_options_setup=None,
        extra_options_call=None,
    ):
        super().__init__(
            "problem dependent", rtol, atol, root_method, root_tol, extrap_tol
        )
        if mode in ["safe", "fast", "fast with events", "safe without grid"]:
            self.mode = mode
        else:
            raise ValueError(
                "invalid mode '{}'. Must be 'safe', for solving with events, "
                "'fast', for solving quickly without events, or 'safe without grid' or "
                "'fast with events' (both experimental)".format(mode)
            )
        self.max_step_decrease_count = max_step_decrease_count
        self.dt_max = dt_max

        self.extra_options_setup = extra_options_setup or {}
        self.extra_options_call = extra_options_call or {}
        self.extrap_tol = extrap_tol

        self.name = "CasADi solver with '{}' mode".format(mode)

        # Initialize
        self.integrators = {}
        self.integrator_specs = {}

        pybamm.citations.register("Andersson2019")

    def _integrate(self, model, t_eval, inputs_dict=None):
        """
        Solve a DAE model defined by residuals with initial conditions y0.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : numeric type
            The times at which to compute the solution
        inputs_dict : dict, optional
            Any external variables or input parameters to pass to the model when solving
        """
        # Record whether there are any symbolic inputs
        inputs_dict = inputs_dict or {}
        has_symbolic_inputs = any(
            isinstance(v, casadi.MX) for v in inputs_dict.values()
        )

        # convert inputs to casadi format
        inputs = casadi.vertcat(*[x for x in inputs_dict.values()])

        # Calculate initial event signs needed for some of the modes
        if (
            has_symbolic_inputs is False
            and self.mode != "fast"
            and model.terminate_events_eval
        ):
            init_event_signs = np.sign(
                np.concatenate(
                    [
                        event(t_eval[0], model.y0, inputs)
                        for event in model.terminate_events_eval
                    ]
                )
            )
        else:
            init_event_signs = np.sign([])

        if has_symbolic_inputs:
            # Create integrator without grid to avoid having to create several times
            self.create_integrator(model, inputs)
            solution = self._run_integrator(
                model, model.y0, inputs_dict, inputs, t_eval
            )
            solution.termination = "final time"
            return solution
        elif self.mode in ["fast", "fast with events"] or not model.events:
            if not model.events:
                pybamm.logger.info("No events found, running fast mode")
            if self.mode == "fast with events":
                # Create the integrator with an event switch that will set the rhs to
                # zero when voltage limits are crossed
                use_event_switch = True
            else:
                use_event_switch = False
            # Create an integrator with the grid (we just need to do this once)
            self.create_integrator(
                model, inputs, t_eval, use_event_switch=use_event_switch
            )
            solution = self._run_integrator(
                model, model.y0, inputs_dict, inputs, t_eval
            )
            # Check if the sign of an event changes, if so find an accurate
            # termination point and exit
            solution = self._solve_for_event(solution, init_event_signs)
            return solution
        elif self.mode in ["safe", "safe without grid"]:
            y0 = model.y0
            # Step-and-check
            t = t_eval[0]
            t_f = t_eval[-1]

            pybamm.logger.debug(
                "Start solving {} with {}".format(model.name, self.name)
            )

            if self.mode == "safe without grid":
                # in "safe without grid" mode,
                # create integrator once, without grid,
                # to avoid having to create several times
                self.create_integrator(model, inputs)
                # Initialize solution
                solution = pybamm.Solution(np.array([t]), y0, model, inputs_dict)
                solution.solve_time = 0
                solution.integration_time = 0
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
                            model, y0, inputs_dict, inputs, t_window
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
                # Check if the sign of an event changes, if so find an accurate
                # termination point and exit
                current_step_sol = self._solve_for_event(
                    current_step_sol, init_event_signs
                )
                # assign temporary solve time
                current_step_sol.solve_time = np.nan
                # append solution from the current step to solution
                solution = solution + current_step_sol
                if current_step_sol.termination == "event":
                    break
                else:
                    # update time
                    t = t_window[-1]
                    # update y0
                    y0 = solution.all_ys[-1][:, -1]
            return solution

    def _solve_for_event(self, coarse_solution, init_event_signs):
        """
        Check if the sign of an event changes, if so find an accurate
        termination point and exit

        Locate the event time using a root finding algorithm and
        event state using interpolation. The solution is then truncated
        so that only the times up to the event are returned
        """
        model = coarse_solution.all_models[-1]
        inputs_dict = coarse_solution.all_inputs[-1]
        inputs = casadi.vertcat(*[x for x in inputs_dict.values()])

        # Check for interpolant extrapolations
        if model.interpolant_extrapolation_events_eval:
            extrap_event = [
                event(t, coarse_solution.y[:, -1], inputs=inputs)
                for event in model.interpolant_extrapolation_events_eval
            ]

            if extrap_event:
                if (np.concatenate(extrap_event) < self.extrap_tol).any():
                    extrap_event_names = []
                    for event in model.events:
                        if (
                            event.event_type
                            == pybamm.EventType.INTERPOLANT_EXTRAPOLATION
                            and (
                                event.expression.evaluate(
                                    t,
                                    coarse_solution.y[:, -1].full(),
                                    inputs=inputs,
                                )
                                < self.extrap_tol
                            ).any()
                        ):
                            extrap_event_names.append(event.name[12:])

                    raise pybamm.SolverError(
                        "CasADI solver failed because the following "
                        "interpolation bounds were exceeded: {}. You may need "
                        "to provide additional interpolation points outside "
                        "these bounds.".format(extrap_event_names)
                    )

        def find_t_event(sol, interpolant_kind):

            # Check most recent y to see if any events have been crossed
            if model.terminate_events_eval:
                y_last = sol.all_ys[-1][:, -1]
                new_event_signs = np.sign(
                    np.concatenate(
                        [
                            event(sol.t[-1], y_last, inputs)
                            for event in model.terminate_events_eval
                        ]
                    )
                )
            else:
                new_event_signs = np.sign([])

            # Return None if no events have been triggered
            if (new_event_signs == init_event_signs).all():
                return None, None

            # get the index of the events that have been crossed
            event_ind = np.where(new_event_signs != init_event_signs)[0]
            active_events = [model.terminate_events_eval[i] for i in event_ind]

            # create interpolant to evaluate y in the current integration
            # window
            y_sol = interp1d(sol.t, sol.y, kind=interpolant_kind)

            # loop over events to compute the time at which they were triggered
            t_events = [None] * len(active_events)
            for i, event in enumerate(active_events):
                init_event_sign = init_event_signs[event_ind[i]][0]

                def event_fun(t):
                    # We take away 1e-5 to deal with the case where the event sits
                    # exactly on zero, as can happen when the event switch is used
                    # (fast with events mode)
                    return init_event_sign * event(t, y_sol(t), inputs) - 1e-5

                if np.isnan(event_fun(sol.t[-1])[0]):
                    # bracketed search fails if f(a) or f(b) is NaN, so we
                    # need to find the times for which we can evaluate the event
                    times = [t for t in sol.t if event_fun(t)[0] == event_fun(t)[0]]
                else:
                    times = sol.t
                # skip if sign hasn't changed
                if np.sign(event_fun(times[0])) != np.sign(event_fun(times[-1])):
                    t_events[i] = brentq(lambda t: event_fun(t), times[0], times[-1])
                else:
                    t_events[i] = np.nan

            # t_event is the earliest event triggered
            t_event = np.nanmin(t_events)
            y_event = y_sol(t_event)

            return t_event, y_event

        # Find the interval in which the event was triggered
        t_event_coarse, _ = find_t_event(coarse_solution, "linear")

        # Return the existing solution if no events have been triggered
        if t_event_coarse is None:
            # Flag "final time" for termination
            coarse_solution.termination = "final time"
            return coarse_solution

        # If events have been triggered, we solve for a dense window in the interval
        # where the event was triggered, then find the precise location of the event
        event_idx = np.where(coarse_solution.t > t_event_coarse)[0][0] - 1
        t_window_event = coarse_solution.t[event_idx : event_idx + 2]

        # Solve again with a more dense t_window, starting from the start of the
        # window where the event was triggered
        t_window_event_dense = np.linspace(t_window_event[-2], t_window_event[-1], 100)
        if self.mode in ["safe", "fast with events"]:
            self.create_integrator(model, inputs, t_window_event_dense)

        y0 = coarse_solution.y[:, event_idx]
        dense_step_sol = self._run_integrator(
            model, y0, inputs_dict, inputs, t_window_event_dense
        )

        # Find the exact time at which the event was triggered
        t_event, y_event = find_t_event(dense_step_sol, "cubic")

        # Return solution truncated at the first coarse event time
        # Also assign t_event
        t_sol = coarse_solution.t[: event_idx + 1]
        y_sol = coarse_solution.y[:, : event_idx + 1]
        solution = pybamm.Solution(
            t_sol,
            y_sol,
            model,
            inputs_dict,
            np.array([t_event]),
            y_event[:, np.newaxis],
            "event",
        )
        solution.integration_time = coarse_solution.integration_time

        # Flag "True" for termination
        return solution

    def create_integrator(self, model, inputs, t_eval=None, use_event_switch=False):
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
                return self.integrators[model][0]
            # Otherwise, create new integrator with an updated grid
            # We don't need to update the grid if reusing the same t_eval
            else:
                method, problem, options = self.integrator_specs[model]
                t_eval_old = options["grid"]
                if np.array_equal(t_eval_old, t_eval):
                    return self.integrators[model][0]
                else:
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
            y_alg = casadi.MX.sym("y_alg", algebraic(0, y0, p).shape[0])
            y_full = casadi.vertcat(y_diff, y_alg)

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

            # define the event switch as the point when an event is crossed
            # we don't do this for ODE models
            # see #1082
            event_switch = 1
            if use_event_switch is True and not algebraic(0, y0, p).is_empty():
                for event in model.casadi_terminate_events:
                    event_switch *= event(t_scaled, y_full, p)

            problem = {
                "t": t,
                "x": y_diff,
                # rescale rhs by (t_max - t_min)
                "ode": (t_max - t_min) * rhs(t_scaled, y_full, p) * event_switch,
                "p": p_with_tlims,
            }
            if algebraic(0, y0, p).is_empty():
                method = "cvodes"
            else:
                method = "idas"
                problem.update(
                    {
                        "z": y_alg,
                        "alg": algebraic(t_scaled, y_full, p),
                    }
                )
            integrator = casadi.integrator("F", method, problem, options)
            self.integrator_specs[model] = method, problem, options
            self.integrators[model] = (integrator, use_grid)
            return integrator

    def _run_integrator(self, model, y0, inputs_dict, inputs, t_eval):
        integrator, use_grid = self.integrators[model]
        len_rhs = model.concatenated_rhs.size
        y0_diff = y0[:len_rhs]
        y0_alg = y0[len_rhs:]
        try:
            # Try solving
            if use_grid is True:
                # Call the integrator once, with the grid
                timer = pybamm.Timer()
                casadi_sol = integrator(
                    x0=y0_diff, z0=y0_alg, p=inputs, **self.extra_options_call
                )
                integration_time = timer.time()
                y_sol = casadi.vertcat(casadi_sol["xf"], casadi_sol["zf"])
                sol = pybamm.Solution(t_eval, y_sol, model, inputs_dict)
                sol.integration_time = integration_time
                return sol
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
                    timer = pybamm.Timer()
                    casadi_sol = integrator(
                        x0=x, z0=z, p=inputs_with_tlims, **self.extra_options_call
                    )
                    integration_time = timer.time()
                    x = casadi_sol["xf"]
                    z = casadi_sol["zf"]
                    y_diff = casadi.horzcat(y_diff, x)
                    if not z.is_empty():
                        y_alg = casadi.horzcat(y_alg, z)
                if z.is_empty():
                    sol = pybamm.Solution(t_eval, y_diff, model, inputs_dict)
                else:
                    y_sol = casadi.vertcat(y_diff, y_alg)
                    sol = pybamm.Solution(t_eval, y_sol, model, inputs_dict)

                sol.integration_time = integration_time
                return sol
        except RuntimeError as e:
            # If it doesn't work raise error
            raise pybamm.SolverError(e.args[0])
