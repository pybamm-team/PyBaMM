#
# CasADi Solver class
#
import casadi
import pybamm
import numpy as np


class CasadiSolver(pybamm.DaeSolver):
    """Solve a discretised model, using CasADi.

    **Extends**: :class:`pybamm.DaeSolver`

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
        method="idas",
        mode="safe",
        rtol=1e-6,
        atol=1e-6,
        root_method="lm",
        root_tol=1e-6,
        max_step_decrease_count=5,
        **extra_options,
    ):
        super().__init__(method, rtol, atol, root_method, root_tol)
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
        self.name = "CasADi solver ({}) with '{}' mode".format(method, mode)

    def solve(self, model, t_eval):
        """
        Execute the solver setup and calculate the solution of the model at
        specified times.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        t_eval : numeric type
            The times at which to compute the solution

        Raises
        ------
        :class:`pybamm.ValueError`
            If an invalid mode is passed.
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic={}`)

        """
        if self.mode == "fast":
            # Solve model normally by calling the solve method from parent class
            return super().solve(model, t_eval)
        elif model.events == {}:
            pybamm.logger.info("No events found, running fast mode")
            # Solve model normally by calling the solve method from parent class
            return super().solve(model, t_eval)
        elif self.mode == "safe":
            # Step-and-check
            timer = pybamm.Timer()
            self.set_up_casadi(model)
            set_up_time = timer.time()
            init_event_signs = np.sign(
                np.concatenate([event(0, self.y0) for event in self.event_funs])
            )
            solution = None
            pybamm.logger.info(
                "Start solving {} with {} in 'safe' mode".format(model.name, self.name)
            )
            self.t = 0.0
            for dt in np.diff(t_eval):
                # Step
                solved = False
                count = 0
                while not solved:
                    # Try to solve with the current step, if it fails then halve the
                    # step size and try again. This will make solution.t slightly
                    # different to t_eval, but shouldn't matter too much as it should
                    # only happen near events.
                    try:
                        current_step_sol = self.step(model, dt)
                        solved = True
                    except pybamm.SolverError:
                        dt /= 2
                    count += 1
                    if count >= self.max_step_decrease_count:
                        if solution is None:
                            t = 0
                        else:
                            t = solution.t[-1]
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
                            event(0, current_step_sol.y[:, -1])
                            for event in self.event_funs
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
                    if not solution:
                        # create solution object on first step
                        solution = current_step_sol
                        solution.set_up_time = set_up_time
                    else:
                        # append solution from the current step to solution
                        solution.append(current_step_sol)

            # Calculate more exact termination reason
            solution.termination = self.get_termination_reason(solution, self.events)
            pybamm.logger.info(
                "Finish solving {} ({})".format(model.name, solution.termination)
            )
            pybamm.logger.info(
                "Set-up time: {}, Solve time: {}, Total time: {}".format(
                    timer.format(solution.set_up_time),
                    timer.format(solution.solve_time),
                    timer.format(solution.total_time),
                )
            )
            return solution

    def compute_solution(self, model, t_eval):
        """Calculate the solution of the model at specified times. In this class, we
        overwrite the behaviour of :class:`pybamm.DaeSolver`, since CasADi requires
        slightly different syntax.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        t_eval : numeric type
            The times at which to compute the solution

        """
        timer = pybamm.Timer()

        solve_start_time = timer.time()
        pybamm.logger.debug("Calling DAE solver")
        solution = self.integrate_casadi(
            self.casadi_problem, self.y0, t_eval, mass_matrix=model.mass_matrix.entries
        )
        solve_time = timer.time() - solve_start_time

        # Events not implemented, termination is always 'final time'
        termination = "final time"

        return solution, solve_time, termination

    def integrate_casadi(self, problem, y0, t_eval, mass_matrix=None):
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
        mass_matrix : array_like, optional
            The (sparse) mass matrix for the chosen spatial method. This is only passed
            to check that the mass matrix is diagonal with 1s for the odes and 0s for
            the algebraic equations, as CasADi does not allow to pass mass matrices.
        """
        options = {
            "grid": t_eval,
            "reltol": self.rtol,
            "abstol": self.atol,
            "output_t0": True,
            "max_num_steps": self.max_steps,
        }
        options.update(self.extra_options)
        if self.method == "idas":
            options["calc_ic"] = True

        # set up and solve
        integrator = casadi.integrator("F", self.method, problem, options)
        try:
            # Try solving
            len_rhs = problem["x"].size()[0]
            y0_diff, y0_alg = np.split(y0, [len_rhs])
            sol = integrator(x0=y0_diff, z0=y0_alg)
            y_values = np.concatenate([sol["xf"].full(), sol["zf"].full()])
            return pybamm.Solution(t_eval, y_values, None, None, "final time")
        except RuntimeError as e:
            # If it doesn't work raise error
            raise pybamm.SolverError(e.args[0])

