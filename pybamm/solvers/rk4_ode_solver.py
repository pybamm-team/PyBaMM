#
# Solver class using fixed step Runge-Kutta 4 integrator
#
import casadi
import pybamm
import numpy as np


class RK4Solver(pybamm.BaseSolver):
    """Solve a discretised model, using RK4.

    **Extends**: :class:`pybamm.BaseSolver`

    Parameters
    ----------
    N : float
        The number integration steps within a time step.

    """

    def __init__(
        self,
        N,  # number of control intervals
        rtol=1e-6,
        atol=1e-6,
        root_method="casadi",
        root_tol=1e-6,
        max_step_decrease_count=20,
    ):
        super().__init__("problem dependent", rtol, atol, root_method, root_tol)
        self.N = N
        self.name = "RK4 ODE solver"

        self.max_step_decrease_count = max_step_decrease_count

    def _integrate(self, model, t_eval, inputs=None):
        """
        Solve a model defined by dydt with initial conditions y0.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : numeric type
            The times at which to compute the solution
        inputs : dict, optional
            Any input parameters to pass to the model when solving

        """
        inputs = inputs or {}
        # convert inputs to casadi format
        inputs = casadi.vertcat(*[x for x in inputs.values()])

        integrator = self.get_integrator(model, t_eval, inputs)
        solution = self._run_integrator(integrator, model, model.y0, inputs, t_eval)
        return solution

    def get_integrator(self, model, t_eval, inputs):
        # Only set up problem once
        y0 = model.y0
        if isinstance(y0, casadi.DM):
            y0 = y0.full().flatten()
        rhs = model.casadi_rhs
        # algebraic = model.casadi_algebraic
        if not model.events:
            init_event_signs = np.array([1, 1])
        else:
            # Step-and-check
            t0 = t_eval[0]
            init_event_signs = np.sign(
                np.concatenate(
                    [event(t0, y0, inputs) for event in model.terminate_events_eval]
                )
            )
        pybamm.logger.info("Start solving {} with {}".format(model.name, self.name))

        # set up and solve
        U = inputs
        DT = casadi.MX.sym("DT")
        t0 = casadi.MX.sym("t0")
        t = t0

        Y0 = casadi.MX.sym("Y0", y0.shape[0])
        Y = Y0

        # Formulate discrete time dynamics
        # Fixed step Runge-Kutta 4 integrator
        for j in range(self.N):
            k1 = rhs(t, Y, U)
            k2 = rhs(t + DT / 2 / self.N, Y + DT / 2 / self.N * k1, U)
            k3 = rhs(t + DT / 2 / self.N, Y + DT / 2 / self.N * k2, U)
            k4 = rhs(t + DT / self.N, Y + DT / self.N * k3, U)
            Y = Y + DT / 6 / self.N * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + DT / self.N
        F = casadi.Function("F", [t0, Y0, DT], [Y], ["t0", "y0", "dt"], ["yf"])

        # Try solving
        # Get the trajectory
        xk = y0
        y_sol = casadi.DM.zeros(y0.shape[0], t_eval.shape[0])
        y_sol[:, 0] = y0
        for k in range(t_eval.shape[0] - 1):
            dt = t_eval[k + 1] - t_eval[k]
            xkp1 = F(t0=t_eval[k], y0=xk, dt=dt)["yf"]
            y_sol[:, k + 1] = xkp1

            xk = xkp1

            # Check most recent y to see if any events have been crossed
            if not model.events:
                new_event_signs = np.array([1, 1])
            else:
                new_event_signs = np.sign(
                    np.concatenate(
                        [
                            event(t_eval[k + 1], xk.full().flatten(), inputs)
                            for event in model.terminate_events_eval
                        ]
                    )
                )
            # Exit loop if the sign of an event changes
            # Locate the event time using a root finding algorithm and
            # event state using interpolation. The solution is then truncated
            # so that only the times up to the event are returned
            i = 1
            if (new_event_signs != init_event_signs).any():
                t_event = t_eval[k]
                y_event = y_sol[:, k]
                pre_event_signs = init_event_signs
                s = 1
                dt = dt / 2
                while (i < self.max_step_decrease_count) and (dt > 1e-6):
                    i += 1
                    xk = F(t0=t_event, y0=y_event, dt=dt * s)["yf"]
                    t_event = t_event + s * dt
                    y_event = xk.full().flatten()

                    new_event_signs = np.sign(
                        np.concatenate(
                            [
                                event(t_event, y_event, inputs)
                                for event in model.terminate_events_eval
                            ]
                        )
                    )
                    if (new_event_signs != pre_event_signs).any():
                        s = -1 * s
                    dt = dt / 2
                    pre_event_signs = new_event_signs
                y_sol[:, k + 1] = y_event
                break

        if i > 1:
            # return truncated solution
            t_truncated = np.append(t_eval[: k + 1], t_event)
            y_trunctaed = y_sol[:, : k + 2].full()
            truncated_step_sol = pybamm.Solution(t_truncated, y_trunctaed)
            # assign temporary solve time
            truncated_step_sol.solve_time = np.nan
            # append solution from the current step to solution
            solution = truncated_step_sol

            solution.termination = "event"
            solution.t_event = t_event
            solution.y_event = y_event
        else:
            # return full solution
            step_sol = pybamm.Solution(t_eval, y_sol.full())
            # assign temporary solve time
            step_sol.solve_time = np.nan
            # append solution from the current step to solution
            solution = step_sol

            solution.termination = "final"

        return solution

    def _run_integrator(self, integrator, model, y0, inputs, t_eval):
        try:
            sol = integrator
            y_values = sol.y
            return pybamm.Solution(sol.t, y_values)
        except RuntimeError as e:
            # If it doesn't work raise error
            raise pybamm.SolverError(e.args[0])
