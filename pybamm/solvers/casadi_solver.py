#
# CasADi Solver class
#
import casadi
import pybamm
import numpy as np


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
    extra_options_setup : dict, optional
        Any options to pass to the CasADi integrator when creating the integrator.
        Please consult `CasADi documentation <https://tinyurl.com/y5rk76os>`_ for
        details.
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
        extra_options_setup=None,
        extra_options_call=None,
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
        self.extra_options_setup = extra_options_setup or {}
        self.extra_options_call = extra_options_call or {}
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
            y0 = model.y0
            if isinstance(y0, casadi.DM):
                y0 = y0.full().flatten()
            # Step-and-check
            t = t_eval[0]
            init_event_signs = np.sign(
                np.concatenate(
                    [event(t, y0, inputs) for event in model.terminate_events_eval]
                )
            )
            pybamm.logger.info("Start solving {} with {}".format(model.name, self.name))

            # Initialize solution
            solution = pybamm.Solution(np.array([t]), y0[:, np.newaxis])
            solution.solve_time = 0
            for dt in np.diff(t_eval):
                # Step
                solved = False
                count = 0
                while not solved:
                    integrator = self.get_integrator(
                        model, np.array([t, t + dt]), inputs
                    )
                    # Try to solve with the current step, if it fails then halve the
                    # step size and try again. This will make solution.t slightly
                    # different to t_eval, but shouldn't matter too much as it should
                    # only happen near events.
                    try:
                        current_step_sol = self._run_integrator(
                            integrator, model, y0, inputs, np.array([t, t + dt])
                        )
                        solved = True
                    except pybamm.SolverError:
                        dt /= 2
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
                # Check most recent y
                new_event_signs = np.sign(
                    np.concatenate(
                        [
                            event(t, current_step_sol.y[:, -1], inputs)
                            for event in model.terminate_events_eval
                        ]
                    )
                )
                # Exit loop if the sign of an event changes
                if (new_event_signs != init_event_signs).any():
                    solution.termination = "event"
                    solution.t_event = solution.t[-1]
                    solution.y_event = solution.y[:, -1]
                    break
                else:
                    # assign temporary solve time
                    current_step_sol.solve_time = np.nan
                    # append solution from the current step to solution
                    solution.append(current_step_sol)
                    # update time
                    t += dt
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
                **self.extra_options_setup,
                "grid": t_eval,
                "reltol": self.rtol,
                "abstol": self.atol,
                "output_t0": True,
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
        rhs_size = model.concatenated_rhs.size
        y0_diff, y0_alg = np.split(y0, [rhs_size])
        try:
            # Try solving
            sol = integrator(x0=y0_diff, z0=y0_alg, p=inputs, **self.extra_options_call)
            y_values = np.concatenate([sol["xf"].full(), sol["zf"].full()])
            return pybamm.Solution(t_eval, y_values)
        except RuntimeError as e:
            # If it doesn't work raise error
            raise pybamm.SolverError(e.args[0])
