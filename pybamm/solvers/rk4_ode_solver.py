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


    """

    def __init__(
        self,
        N,  # number of control intervals
        rtol=1e-6,
        atol=1e-6,
        root_method="casadi",
        root_tol=1e-6,
        # extra_options_setup=None,
        # extra_options_call=None,
    ):
        super().__init__("problem dependent", rtol, atol, root_method, root_tol)
        self.N = N
        self.name = "RK4 ODE solver"

        # self.extra_options_setup = extra_options_setup or {}
        # self.extra_options_call = extra_options_call or {}

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
        solution.termination = "final time"
        return solution

    def get_integrator(self, model, t_eval, inputs):
        # Only set up problem once
        y0 = model.y0
        rhs = model.casadi_rhs
        # algebraic = model.casadi_algebraic

        # When not in DEBUG mode (level=10), suppress warnings from CasADi
        if (
            pybamm.logger.getEffectiveLevel() == 10
            or pybamm.settings.debug_mode is True
        ):
            show_eval_warnings = True
        else:
            show_eval_warnings = False

        # set up and solve
        U = inputs  # casadi.MX.sym("U", inputs.shape[0])
        t = casadi.MX.sym("t")
        # y_diff = casadi.MX.sym("y_diff", rhs(t_eval[0], y0, U).shape[0])

        # Formulate discrete time dynamics
        # Fixed step Runge-Kutta 4 integrator
        M = self.N  # RK4 steps per interval
        DT = casadi.MX.sym("DT")  # t_eval[-1] / M
        # f = Function("f", [model.states, inputs], [derivs])
        Y0 = casadi.MX.sym("Y0", y0.shape[0])
        # U = MX.sym("U")
        Y = Y0

        for j in range(M):
            k1 = rhs(t, Y, U)
            k2 = rhs(t + DT / 2, Y + DT / 2 * k1, U)
            k3 = rhs(t + DT / 2, Y + DT / 2 * k2, U)
            k4 = rhs(t + DT, Y + DT * k3, U)
            Y = Y + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        F = casadi.Function("F", [t, Y0, DT], [Y], ["t", "y0", "dt"], ["yf"])

        # Try solving
        # Get the trajectory
        xk = y0
        sol = casadi.DM.zeros(y0.shape[0], t_eval.shape[0])
        sol[:, 0] = y0
        for k in range(t_eval.shape[0] - 1):
            dt = (t_eval[k + 1] - t_eval[k]) / M
            xk = F(t=t_eval[k], y0=xk, dt=dt)["yf"]
            sol[:, k + 1] = xk

        return sol

    def _run_integrator(self, integrator, model, y0, inputs, t_eval):
        # rhs_size = model.concatenated_rhs.size
        # y0_diff, y0_alg = np.split(y0, [rhs_size])
        try:
            sol = integrator
            # y_values = np.concatenate([sol["xf"].full(), sol["zf"].full()])
            # y_values = np.reshape(np.array(sol), (y0.shape[0], t_eval.shape[0]))
            y_values = sol.full()
            return pybamm.Solution(t_eval, y_values)
        except RuntimeError as e:
            # If it doesn't work raise error
            raise pybamm.SolverError(e.args[0])
