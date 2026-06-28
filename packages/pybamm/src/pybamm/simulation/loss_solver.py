from enum import Enum

import casadi
import numpy as np
import pydiffsol as ds

import pybamm


class LossSolver:
    """
    A solver defined by a PyBaMM time-series model (i.e. dy/dt = f(y, t, p))
    and a scalar loss function L(p) = g(y(t), p). Has the functionality to:
    (a) calculate the loss function for a given set of parameters p, by solving the ODE and evaluating L at the solution;
    (c) calculate the solution of the ODE y(t) for a given set of parameters p, by solving the ODE;
    (b) calculate the gradient of the loss function with respect to the parameters p, by:
       (i) finitely-differencing the loss function with respect to p, which requires multiple ODE solves; or
       (ii) solving the forward sensitivity equations, which requires a single ODE solve of an augmented system of equations.
       (iii) solving the adjoint equations, which requires a single ODE solve of an augmented system of equations backwards in time.
    (d) batching the above calculations across multiple parameter sets p
    (c) pickle and unpickle the solver so it can be saved and loaded in multiple contexts (e.g. training and inference workflows)
    By definition, the loss function can only vary with the parameters p, and it must depend on the solution y(t) for the loss to be well-defined,
    so there are some restrictions on the form of the loss function g(y(t), p):
    (1) g must include either a pybamm.DiscreteSumSum or a pybamm.ExplicitTimeIntegral node in the expression tree that integrates/sums over time and the solution y(t).
    (2) any instances of the state variables or time in g must be contained within the scope of the aforementioned time-sum/integral node
    (3) the output shape of g must be a scalar
    For convenience, a sum-of-squared-error loss function factory method is provided to create a LossSolver from data in BDF format
    """

    INNER_LOSS_FUNCTION_NAME = "inner loss function"

    def __init__(
        self, sim: pybamm.Simulation, loss_function: pybamm.Symbol, final_time: float
    ):
        self._sim = sim
        self._processed_loss = pybamm.ProcessedVariableTimeIntegral.from_pybamm_var(
            loss_function, final_time
        )
        if self._processed_loss is None:
            raise ValueError(
                "Loss function must contain either a DiscreteSum or an ExplicitTimeIntegral node"
            )
        self._final_time = final_time

        # add inner function as the output of the model
        if self.INNER_LOSS_FUNCTION_NAME in self._sim.model.variables:
            raise ValueError(
                f"Model already contains a variable named {self.INNER_LOSS_FUNCTION_NAME}. "
                "Please rename the inner loss function to avoid conflicts."
            )
        self._sim.model.variables[self.INNER_LOSS_FUNCTION_NAME] = self._inner

        self._exporter = pybamm.DiffSLExport(self._sim)
        code = self._exporter.to_diffeq([self.INNER_LOSS_FUNCTION_NAME])
        self._ode = ds.Ode(
            code,
            matrix_type=ds.faer_sparse,
            scalar_type=ds.f64,
            linear_solver=ds.lu,
            ode_solver=ds.bdf,
        )
        self._ode.integrate_out = self._processed_loss.method == "continuous"
        self._ode.rtol = self._sim.solver.rtol
        self._ode.atol = self._sim.solver.atol

        # generate casadi functions for post-sum node and its sensitivities
        inputs = self._exporter.default_inputs()
        post_sum_node = self._processed_loss.post_sum_node
        (self._post_sum, self._post_sum_sens) = self._post_sum(
            post_sum_node, inputs, self._exporter.input_names()
        )

    def _post_sum(self, var_pybamm, inputs, input_names):
        t_casadi = casadi.MX.sym("t")
        sum_casadi = casadi.MX.sym("sum", 1)
        p_casadi = {name: casadi.MX.sym(name, value) for name, value in inputs.items()}
        post_sum_casadi = var_pybamm.to_casadi(t_casadi, sum_casadi, inputs=p_casadi)

        p_casadi_stacked = casadi.vertcat(*[p_casadi[name] for name, _ in input_names])
        sens_casadi = casadi.MX.sym("sens", len(input_names))
        dpost_dy = casadi.jacobian(post_sum_casadi, sum_casadi)
        dpost_dp = casadi.jacobian(post_sum_casadi, p_casadi_stacked)
        sens = dpost_dy * sens_casadi + dpost_dp
        post_sum_sens_casadi = casadi.Function(
            "sens_fun",
            [t_casadi, sum_casadi, p_casadi, sens_casadi],
            [sens],
        )
        return post_sum_casadi, post_sum_sens_casadi

    def inputs_to_parameters(self, inputs: list[dict]) -> np.ndarray:
        """Converts a standard set of pybamm input dictionaries to a 2D parameter array (n_batch, n_params) for use in the functions below."""
        # TODO: add batching
        return self._exporter.map_inputs(
            inputs, outputs=[self.INNER_LOSS_FUNCTION_NAME]
        )

    def parameters_to_inputs(self, p: np.ndarray) -> list[dict]:
        """Converts a 2D parameter array (n_batch, n_params) to a standard set of pybamm input dictionaries."""
        # TODO: implement inverse_map_inputs in DiffSLExporter to allow this to work
        # TODO: add batching
        return self._exporter.inverse_map_inputs(
            p, outputs=[self.INNER_LOSS_FUNCTION_NAME]
        )

    def predict(self, p: np.ndarray) -> list[pybamm.Solution]:
        """Calculate the solution of the ODE for each set of parameters in inputs."""
        # TODO: add batching
        inputs = self.parameters_to_inputs(p)
        return self._sim.solve(inputs=inputs)

    def _discrete_sum_to_loss(self, sol: ds.Solution, inputs: dict) -> np.ndarray:
        """Calculate the loss function for a discrete sum loss function."""
        the_integral = np.sum(sol.ys, axis=1)
        if self.post_sum_node is None:
            ret = the_integral
        else:
            ret = self._post_sum(0.0, the_integral, inputs).full()
        return ret

    def _explicit_time_integral_to_loss(
        self, sol: ds.Solution, inputs: dict
    ) -> np.ndarray:
        """Calculate the loss function for an explicit time integral loss function."""
        the_integral = sol.ys[:, -1]
        if self.post_sum_node is None:
            ret = the_integral
        else:
            ret = self._post_sum(0.0, the_integral, inputs).full()
        return ret

    def loss(self, p: np.ndarray) -> np.ndarray:
        """
        Calculate the loss function for each set of parameters in inputs.
        Returns a 1D array of loss values of length n_batch.
        """
        # TODO: add batching
        if self._processed_loss.method == "discrete":
            sol = self._ode.solve_dense(p, self._processed_loss.discrete_times)
            return self._discrete_sum_to_loss(sol, self.parameters_to_inputs(p))
        elif self._processed_loss.method == "continuous":
            sol = self._ode.solve(p, self._final_time)
            return self._explicit_time_integral_to_loss(
                sol, self.parameters_to_inputs(p)
            )

    def finite_difference_gradient(self, p: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        Calculate the gradient of the loss function with respect to the parameters for each set of parameters in inputs using finite differencing.
        Returns a 2D array of gradients, with shape (n_batch, n_params), where each row corresponds to the gradient for a given input parameter set.
        """
        raise NotImplementedError("LossSolver is not yet implemented")

    def _discrete_sum_to_gradient(
        self, sol: ds.Solution, inputs: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        ys_sum = np.sum(sol.ys, axis=1)
        sens_sum = np.array([np.sum(s, axis=1) for s in sol.sens])
        if self.post_sum_node is None:
            return ys_sum, sens_sum
        else:
            loss = self.post_sum_node.evaluate(0.0, ys_sum, None, inputs)
            gradient = self._post_sum_sens(0.0, ys_sum, inputs, sens_sum)
            return loss, gradient

    def loss_and_gradient(
        self, p: np.ndarray, mode: "LossSolverGradientMode"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the loss and gradient of the loss function with respect to the parameters for each set of parameters in inputs.
        Returns a tuple of arrays, where the first contains the loss values as a 1D array of length n_batch and the second contains the gradients as a 2D array of shape (n_batch, n_params)
        """
        # TODO: add batching
        if self._processed_loss.method == "discrete":
            times = self._processed_loss.discrete_times
            if mode == self.LossSolverGradientMode.FORWARD_SENSITIVITY:
                sol = self._ode.solve_fwd_sens(p, times)
                return self._discrete_sum_to_gradient(sol, self.parameters_to_inputs(p))
            else:
                raise NotImplementedError(
                    "Adjoint sensitivity for discrete sum is not yet implemented"
                )

        elif self._processed_loss.method == "continuous":
            if mode == self.LossSolverGradientMode.FORWARD_SENSITIVITY:
                raise NotImplementedError(
                    "Forward sensitivity for explicit time integral is not yet implemented"
                )
            else:
                integral, integral_sens = self._ode.solve_continuous_adjoint(
                    p, self._final_time
                )
                if self._post_sum is None:
                    return integral, integral_sens
                else:
                    inputs = self.parameters_to_inputs(p)
                    loss = self._post_sum(0.0, integral, inputs)
                    gradient = self._post_sum_sens(0.0, integral, inputs, integral_sens)
                    return loss, gradient

    class LossSolverGradientMode(Enum):
        FORWARD_SENSITIVITY = "forward_sensitivity"
        ADJOINT_SENSITIVITY = "adjoint_sensitivity"
