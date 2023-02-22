#
# CasADi Solver class
#
import casadi
import pybamm
import numpy as np
import warnings
from scipy.interpolate import interp1d


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
        the default value is 600 seconds.
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
    return_solution_if_failed_early : bool, optional
        Whether to return a Solution object if the solver fails to reach the end of
        the simulation, but managed to take some successful steps. Default is False.
    perturb_algebraic_initial_conditions : bool, optional
        Whether to perturb algebraic initial conditions to avoid a singularity. This
        can sometimes slow down the solver, but is kept True as default for "safe" mode
        as it seems to be more robust (False by default for other modes).
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
        extrap_tol=None,
        extra_options_setup=None,
        extra_options_call=None,
        return_solution_if_failed_early=False,
        perturb_algebraic_initial_conditions=None,
    ):
        super().__init__(
            "problem dependent",
            rtol,
            atol,
            root_method,
            root_tol,
            extrap_tol,
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
        self.dt_max = dt_max or 600

        self.extra_options_setup = extra_options_setup or {}
        self.extra_options_call = extra_options_call or {}
        self.return_solution_if_failed_early = return_solution_if_failed_early

        self._on_extrapolation = "error"

        # Decide whether to perturb algebraic initial conditions, True by default for
        # "safe" mode, False by default for other modes
        if perturb_algebraic_initial_conditions is None:
            if mode == "safe":
                self.perturb_algebraic_initial_conditions = True
            else:
                self.perturb_algebraic_initial_conditions = False
        else:
            self.perturb_algebraic_initial_conditions = (
                perturb_algebraic_initial_conditions
            )
        self.name = "CasADi solver with '{}' mode".format(mode)

        # Initialize
        self.integrators = {}
        self.integrator_specs = {}
        self.y_sols = {}

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
            Any input parameters to pass to the model when solving
        """

        # Record whether there are any symbolic inputs
        inputs_dict = inputs_dict or {}

        # convert inputs to casadi format
        inputs = casadi.vertcat(*[x for x in inputs_dict.values()])

        if self.mode in ["fast", "fast with events"] or not model.events:
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
            solution = self._solve_for_event(solution)
            solution.check_ys_are_not_too_large()
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
                solution = pybamm.Solution(
                    np.array([t]),
                    y0,
                    model,
                    inputs_dict,
                    sensitivities=False,
                )
                solution.solve_time = 0
                solution.integration_time = 0
                use_grid = False
            else:
                solution = None
                use_grid = True

            # Try to integrate in global steps of size dt_max. Note: dt_max must
            # be at least as big as the the biggest step in t_eval (multiplied
            # by some tolerance, here 1.01) to avoid an empty integration window below
            dt_max = self.dt_max
            dt_eval_max = np.max(np.diff(t_eval)) * 1.01
            if dt_max < dt_eval_max:
                pybamm.logger.debug(
                    "Setting dt_max to be as big as the largest step in "
                    f"t_eval ({dt_eval_max})"
                )
                dt_max = dt_eval_max
            termination_due_to_small_dt = False
            first_ts_solved = False
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
                        pybamm.logger.debug(
                            "Running integrator for "
                            f"{t_window[0]:.2f} < t < {t_window[-1]:.2f}"
                        )
                        current_step_sol = self._run_integrator(
                            model,
                            y0,
                            inputs_dict,
                            inputs,
                            t_window,
                            use_grid=use_grid,
                            extract_sensitivities_in_solution=False,
                        )
                        first_ts_solved = True
                        solved = True
                    except pybamm.SolverError as error:
                        pybamm.logger.debug("Failed, halving step size")
                        dt /= 2
                        # also reduce maximum step size for future global steps,
                        # but skip them in the beginning
                        # sometimes, for the first integrator smaller timesteps are
                        # needed, but this won't affect the global timesteps. The
                        # global timestep will only be reduced after the first timestep.
                        if first_ts_solved:
                            dt_max = dt
                        if count >= self.max_step_decrease_count:
                            message = (
                                "Maximum number of decreased steps occurred at "
                                f"t={t} (final SolverError: '{error}'). "
                                "For a full solution try reducing dt_max (currently, "
                                f"dt_max={dt_max}) and/or reducing the size of the "
                                "time steps or period of the experiment."
                            )
                            if first_ts_solved and self.return_solution_if_failed_early:
                                warnings.warn(message, pybamm.SolverWarning)
                                termination_due_to_small_dt = True
                                break
                            else:
                                raise pybamm.SolverError(
                                    message
                                    + " Set `return_solution_if_failed_early=True` to "
                                    "return the solution object up to the point where "
                                    "failure occured."
                                )
                    count += 1
                if termination_due_to_small_dt:
                    break
                # Check if the sign of an event changes, if so find an accurate
                # termination point and exit
                current_step_sol = self._solve_for_event(current_step_sol)
                # assign temporary solve time
                current_step_sol.solve_time = np.nan
                # append solution from the current step to solution
                solution = solution + current_step_sol
                if current_step_sol.termination == "event":
                    break
                else:
                    # update time as time
                    # from which to start the new casadi integrator
                    t = t_window[-1]
                    # update y0 as initial_values
                    # from which to start the new casadi integrator
                    y0 = solution.all_ys[-1][:, -1]

            # now we extract sensitivities from the solution
            if bool(model.calculate_sensitivities):
                solution.sensitivities = True

            solution.check_ys_are_not_too_large()
            return solution

    def _solve_for_event(self, coarse_solution):
        """
        Check if the sign of an event changes, if so find an accurate
        termination point and exit

        Locate the event time using a root finding algorithm and
        event state using interpolation. The solution is then truncated
        so that only the times up to the event are returned
        """
        pybamm.logger.debug("Solving for events")

        model = coarse_solution.all_models[-1]
        inputs_dict = coarse_solution.all_inputs[-1]
        inputs = casadi.vertcat(*[x for x in inputs_dict.values()])

        def find_t_event(sol, typ):
            # Check most recent y to see if any events have been crossed
            if model.terminate_events_eval:
                y_last = sol.all_ys[-1][:, -1]
                crossed_events = np.sign(
                    np.concatenate(
                        [
                            event(sol.t[-1], y_last, inputs)
                            for event in model.terminate_events_eval
                        ]
                    )
                    - 1e-5
                )
            else:
                crossed_events = np.sign([])

            # Return None if no events have been triggered
            if (crossed_events == 1).all():
                return None, None, None

            # get the index of the events that have been crossed
            event_idx = np.where(crossed_events != 1)[0]
            active_events = [model.terminate_events_eval[i] for i in event_idx]

            # loop over events to compute the time at which they were triggered
            t_events = [None] * len(active_events)
            event_idcs_lower = [None] * len(active_events)
            for i, event in enumerate(active_events):
                # Implement our own bisection algorithm for speed
                # This is used to find the time range in which the event is triggered
                # Evaluations of the "event" function are (relatively) expensive
                f_eval = {}

                def f(idx):
                    try:
                        return f_eval[idx]
                    except KeyError:
                        # We take away 1e-5 to deal with the case where the event sits
                        # exactly on zero, as can happen when the event switch is used
                        # (fast with events mode)
                        f_eval[idx] = event(sol.t[idx], sol.y[:, idx], inputs) - 1e-5
                        return f_eval[idx]

                def integer_bisect():
                    a_n = 0
                    b_n = len(sol.t) - 1
                    for _ in range(len(sol.t)):
                        if a_n + 1 == b_n:
                            return a_n
                        m_n = (a_n + b_n) // 2
                        f_m_n = f(m_n)
                        if np.isnan(f_m_n):
                            a_n = a_n
                            b_n = m_n
                        elif f_m_n < 0:
                            a_n = a_n
                            b_n = m_n
                        elif f_m_n > 0:
                            a_n = m_n
                            b_n = b_n

                event_idx_lower = integer_bisect()
                if typ == "window":
                    event_idcs_lower[i] = event_idx_lower
                elif typ == "exact":
                    # Linear interpolation between the two indices to find the root time
                    # We could do cubic interpolation here instead but it would be
                    # slower
                    t_lower = sol.t[event_idx_lower]
                    t_upper = sol.t[event_idx_lower + 1]
                    event_lower = abs(f(event_idx_lower))
                    event_upper = abs(f(event_idx_lower + 1))

                    t_events[i] = (event_lower * t_upper + event_upper * t_lower) / (
                        event_lower + event_upper
                    )

            if typ == "window":
                event_idx_lower = np.nanmin(event_idcs_lower)
                return event_idx_lower, None, None
            elif typ == "exact":
                # t_event is the earliest event triggered
                t_event = np.nanmin(t_events)
                # create interpolant to evaluate y in the current integration
                # window
                y_sol = interp1d(sol.t, sol.y, kind="linear")
                y_event = y_sol(t_event)

                closest_event_idx = event_idx[np.nanargmin(t_events)]

                return t_event, y_event, closest_event_idx

        # Find the interval in which the event was triggered
        event_idx_lower, _, _ = find_t_event(coarse_solution, "window")

        # Return the existing solution if no events have been triggered
        if event_idx_lower is None:
            # Flag "final time" for termination
            coarse_solution.termination = "final time"
            return coarse_solution

        # If events have been triggered, we solve for a dense window in the interval
        # where the event was triggered, then find the precise location of the event
        # Solve again with a more dense idx_window, starting from the start of the
        # window where the event was triggered
        t_window_event_dense = np.linspace(
            coarse_solution.t[event_idx_lower],
            coarse_solution.t[event_idx_lower + 1],
            100,
        )

        if self.mode == "safe without grid":
            use_grid = False
        else:
            self.create_integrator(model, inputs, t_window_event_dense)
            use_grid = True

        y0 = coarse_solution.y[:, event_idx_lower]
        dense_step_sol = self._run_integrator(
            model,
            y0,
            inputs_dict,
            inputs,
            t_window_event_dense,
            use_grid=use_grid,
            extract_sensitivities_in_solution=False,
        )

        # Find the exact time at which the event was triggered
        t_event, y_event, closest_event_idx = find_t_event(dense_step_sol, "exact")
        # If this returns None, no event was crossed in dense_step_sol. This can happen
        # if the event crossing was right at the end of the interval in the coarse
        # solution. In this case, return the t and y from the end of the interval
        # (i.e. next point in the coarse solution)
        if y_event is None:  # pragma: no cover
            # This is extremely rare, it's difficult to find a test that triggers this
            # hence no coverage check
            t_event = coarse_solution.t[event_idx_lower + 1]
            y_event = coarse_solution.y[:, event_idx_lower + 1].full().flatten()

        # Return solution truncated at the first coarse event time
        # Also assign t_event
        t_sol = coarse_solution.t[: event_idx_lower + 1]
        y_sol = coarse_solution.y[:, : event_idx_lower + 1]
        solution = pybamm.Solution(
            t_sol,
            y_sol,
            model,
            inputs_dict,
            np.array([t_event]),
            y_event[:, np.newaxis],
            "event",
            sensitivities=bool(model.calculate_sensitivities),
        )
        solution.integration_time = (
            coarse_solution.integration_time + dense_step_sol.integration_time
        )

        solution.closest_event_idx = closest_event_idx

        return solution

    def create_integrator(self, model, inputs, t_eval=None, use_event_switch=False):
        """
        Method to create a casadi integrator object.
        If t_eval is provided, the integrator uses t_eval to make the grid.
        Otherwise, the integrator has grid [0,1].
        """
        pybamm.logger.debug("Creating CasADi integrator")

        # Use grid if t_eval is given
        use_grid = t_eval is not None
        if use_grid is True:
            t_eval_shifted = t_eval - t_eval[0]
            t_eval_shifted_rounded = np.round(t_eval_shifted, decimals=12).tobytes()
        # Only set up problem once
        if model in self.integrators:
            # If we're not using the grid, we don't need to change the integrator
            if use_grid is False:
                return self.integrators[model]["no grid"]
            # Otherwise, create new integrator with an updated grid
            # We don't need to update the grid if reusing the same t_eval
            # (up to a shift by a constant)
            else:
                if t_eval_shifted_rounded in self.integrators[model]:
                    return self.integrators[model][t_eval_shifted_rounded]
                else:
                    method, problem, options = self.integrator_specs[model]
                    options["grid"] = t_eval_shifted
                    integrator = casadi.integrator("F", method, problem, options)
                    self.integrators[model][t_eval_shifted_rounded] = integrator
                    return integrator
        else:
            rhs = model.casadi_rhs
            algebraic = model.casadi_algebraic

            options = {
                "show_eval_warnings": False,
                **self.extra_options_setup,
                "reltol": self.rtol,
                "abstol": self.atol,
            }

            # set up and solve
            t = casadi.MX.sym("t")
            p = casadi.MX.sym("p", inputs.shape[0])
            y0 = model.y0

            y_diff = casadi.MX.sym("y_diff", rhs(0, y0, p).shape[0])
            y_alg = casadi.MX.sym("y_alg", algebraic(0, y0, p).shape[0])
            y_full = casadi.vertcat(y_diff, y_alg)

            if use_grid is False:
                # rescale time
                t_min = casadi.MX.sym("t_min")
                t_max = casadi.MX.sym("t_max")
                t_max_minus_t_min = t_max - t_min
                t_scaled = t_min + (t_max - t_min) * t
                # add time limits as inputs
                p_with_tlims = casadi.vertcat(p, t_min, t_max)
            else:
                options.update({"grid": t_eval_shifted, "output_t0": True})
                # rescale time
                t_min = casadi.MX.sym("t_min")
                # Set dummy parameters for consistency with rescaled time
                t_max_minus_t_min = 1
                t_scaled = t_min + t
                p_with_tlims = casadi.vertcat(p, t_min)

            # define the event switch as the point when an event is crossed
            # we don't do this for ODE models
            # see #1082
            event_switch = 1
            if use_event_switch is True and not algebraic(0, y0, p).is_empty():
                for event in model.casadi_switch_events:
                    event_switch *= event(t_scaled, y_full, p)

            problem = {
                "t": t,
                "x": y_diff,
                # rescale rhs by (t_max - t_min)
                "ode": (t_max_minus_t_min) * rhs(t_scaled, y_full, p) * event_switch,
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
            if use_grid is False:
                self.integrators[model] = {"no grid": integrator}
            else:
                self.integrators[model] = {t_eval_shifted_rounded: integrator}

            return integrator

    def _run_integrator(
        self,
        model,
        y0,
        inputs_dict,
        inputs,
        t_eval,
        use_grid=True,
        extract_sensitivities_in_solution=None,
    ):
        """
        Run the integrator.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        y0:
            casadi vector of initial conditions
        inputs_dict : dict, optional
            Any input parameters to pass to the model when solving
        inputs:
            Casadi vector of inputs
        t_eval : numeric type
            The times at which to compute the solution
        use_grid: bool, optional
            Determines whether the casadi solver uses a grid or rescales time to (0,1)
        extract_sensitivities_in_solution: bool or None
            If None, then the sensitivities are extracted within the
            :class:`pybamm.Solution` object returned, only if present in the solution.
            Setting to True or False will override this behaviour, forcing the
            sensitivities to be extracted or not (it is up to the caller to determine if
            the sensitivities are in fact present)
        """

        pybamm.logger.debug("Running CasADi integrator")

        # are we solving explicit forward equations?
        explicit_sensitivities = bool(model.calculate_sensitivities)
        # by default we extract sensitivities in the solution if we
        # are calculating the sensitivities
        if extract_sensitivities_in_solution is None:
            extract_sensitivities_in_solution = explicit_sensitivities

        if use_grid is True:
            pybamm.logger.spam("Calculating t_eval_shifted")
            t_eval_shifted = t_eval - t_eval[0]
            t_eval_shifted_rounded = np.round(t_eval_shifted, decimals=12).tobytes()
            pybamm.logger.spam("Finished calculating t_eval_shifted")
            integrator = self.integrators[model][t_eval_shifted_rounded]
        else:
            integrator = self.integrators[model]["no grid"]

        len_rhs = model.concatenated_rhs.size
        len_alg = model.concatenated_algebraic.size

        # Check y0 to see if it includes sensitivities
        if explicit_sensitivities:
            num_parameters = model.len_rhs_sens // model.len_rhs
            len_rhs = len_rhs * (num_parameters + 1)
            len_alg = len_alg * (num_parameters + 1)

        y0_diff = y0[:len_rhs]
        y0_alg = y0[len_rhs:]
        if self.perturb_algebraic_initial_conditions and len_alg > 0:
            # Add a tiny perturbation to the algebraic initial conditions
            # For some reason this helps with convergence
            # The actual value of the initial conditions for the algebraic variables
            # doesn't matter
            y0_alg = y0_alg * (1 + 1e-6 * casadi.DM(np.random.rand(len_alg)))
        pybamm.logger.spam("Finished preliminary setup for integrator run")

        # Solve
        # Try solving
        if use_grid is True:
            t_min = t_eval[0]
            inputs_with_tmin = casadi.vertcat(inputs, t_min)
            # Call the integrator once, with the grid
            timer = pybamm.Timer()
            pybamm.logger.debug("Calling casadi integrator")
            try:
                casadi_sol = integrator(
                    x0=y0_diff, z0=y0_alg, p=inputs_with_tmin, **self.extra_options_call
                )
            except RuntimeError as error:
                # If it doesn't work raise error
                pybamm.logger.debug(f"Casadi integrator failed with error {error}")
                raise pybamm.SolverError(error.args[0])
            pybamm.logger.debug("Finished casadi integrator")
            integration_time = timer.time()
            y_sol = casadi.vertcat(casadi_sol["xf"], casadi_sol["zf"])
            sol = pybamm.Solution(
                t_eval,
                y_sol,
                model,
                inputs_dict,
                sensitivities=extract_sensitivities_in_solution,
                check_solution=False,
            )
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
                try:
                    casadi_sol = integrator(
                        x0=x, z0=z, p=inputs_with_tlims, **self.extra_options_call
                    )
                except RuntimeError as error:
                    # If it doesn't work raise error
                    pybamm.logger.debug(f"Casadi integrator failed with error {error}")
                    raise pybamm.SolverError(error.args[0])
                integration_time = timer.time()
                x = casadi_sol["xf"]
                z = casadi_sol["zf"]
                y_diff = casadi.horzcat(y_diff, x)
                if not z.is_empty():
                    y_alg = casadi.horzcat(y_alg, z)
            if z.is_empty():
                y_sol = y_diff
            else:
                y_sol = casadi.vertcat(y_diff, y_alg)

            sol = pybamm.Solution(
                t_eval,
                y_sol,
                model,
                inputs_dict,
                sensitivities=extract_sensitivities_in_solution,
                check_solution=False,
            )
            sol.integration_time = integration_time
            return sol
