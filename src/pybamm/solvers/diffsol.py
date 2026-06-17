from enum import Enum

import numpy as np
import pandas as pd

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

    def __init__(self, sim: pybamm.Simulation, loss_function: pybamm.Symbol):
        raise NotImplementedError("LossSolver is not yet implemented")

    @staticmethod
    def sum_of_squared_error_loss_function(data: pd.DataFrame) -> "LossSolver":
        """
        A factory method to create a LossSolver with a sum-of-squared-error loss function, given some data in BDF format (https://battery-data-alliance.github.io/battery-data-format/) to fit against.
        """
        raise NotImplementedError("LossSolver is not yet implemented")

    def inputs_to_parameters(self, inputs: list[dict]) -> np.ndarray:
        """Converts a standard set of pybamm input dictionaries to a 2D parameter array (n_batch, n_params) for use in the functions below."""
        raise NotImplementedError("LossSolver is not yet implemented")

    def parameters_to_inputs(self, p: np.ndarray) -> list[dict]:
        """Converts a 2D parameter array (n_batch, n_params) to a standard set of pybamm input dictionaries."""
        raise NotImplementedError("LossSolver is not yet implemented")

    def predict(self, p: np.ndarray) -> list[pybamm.Solution]:
        """Calculate the solution of the ODE for each set of parameters in inputs."""
        raise NotImplementedError("LossSolver is not yet implemented")

    def loss(self, p: np.ndarray) -> np.ndarray:
        """
        Calculate the loss function for each set of parameters in inputs.

        Returns a 1D array of loss values of length n_batch.
        """
        raise NotImplementedError("LossSolver is not yet implemented")

    def finite_difference_gradient(self, p: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        Calculate the gradient of the loss function with respect to the parameters for each set of parameters in inputs using finite differencing.

        Returns a 2D array of gradients, with shape (n_batch, n_params), where each row corresponds to the gradient for a given input parameter set.
        """
        raise NotImplementedError("LossSolver is not yet implemented")

    def loss_and_gradient(
        self, p: np.ndarray, mode: "LossSolverGradientMode"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the loss and gradient of the loss function with respect to the parameters for each set of parameters in inputs.

        Returns a tuple of arrays, where the first contains the loss values as a 1D array of length n_batch and the second contains the gradients as a 2D array of shape (n_batch, n_params)
        """
        raise NotImplementedError("LossSolver is not yet implemented")

    class LossSolverGradientMode(Enum):
        FORWARD_SENSITIVITY = "forward_sensitivity"
        ADJOINT_SENSITIVITY = "adjoint_sensitivity"
