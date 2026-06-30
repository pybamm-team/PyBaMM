import concurrent.futures
import multiprocessing
import pickle
from enum import Enum

import casadi
import numpy as np

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
    (1) g must include either a pybamm.DiscreteTimeSum or a pybamm.ExplicitTimeIntegral node in the expression tree that integrates/sums over time and the solution y(t).
    (2) any instances of the state variables or time in g must be contained within the scope of the aforementioned time-sum/integral node
    (3) the output shape of g must be a scalar
    For convenience, a sum-of-squared-error loss function factory method is provided to create a LossSolver from data in BDF format

    Parameters
    ----------
    sim : pybamm.Simulation
        The simulation wrapping the time-series model.
    loss_function : pybamm.Symbol
        The scalar loss function expression.
    final_time : float
        The final time to integrate the model to.
    max_workers : int or None, optional
        If greater than 1, the batched methods (``loss``, ``loss_and_gradient``,
        ``finite_difference_gradient``) distribute the per-parameter-set solves
        across a process pool of this many workers. Defaults to ``None``
        (sequential). ``predict`` is unaffected; its parallelism comes from the
        wrapped solver's ``num_threads`` option.
    """

    INNER_LOSS_FUNCTION_NAME = "inner loss function"

    def __init__(
        self,
        sim: pybamm.Simulation,
        loss_function: pybamm.Symbol,
        final_time: float,
        max_workers: int | None = None,
    ):
        ds = pybamm.import_optional_dependency("pydiffsol")

        self._sim = sim
        self._max_workers = max_workers
        self._pool = None
        self._processed_loss = pybamm.ProcessedVariableTimeIntegral.from_pybamm_var(
            loss_function, final_time
        )
        if self._processed_loss is None:
            raise ValueError(
                "Loss function must contain either a DiscreteSum or an ExplicitTimeIntegral node"
            )
        self._final_time = final_time

        # the inner function is the integrand that is summed/integrated over time
        self._inner = self._processed_loss.sum_node.orphans[0]

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

        # generate casadi functions for the post-sum node and its sensitivities
        self._post_sum, self._post_sum_sens = self._make_post_sum_functions(
            self._processed_loss.post_sum_node
        )

        # set up the process pool for the batched methods, if requested
        if self._max_workers is not None and self._max_workers > 1:
            self._pool = self._start_pool()

    def inputs_to_parameters(self, inputs: list[dict]) -> np.ndarray:
        """Converts a standard set of pybamm input dictionaries to a 2D parameter array (n_batch, n_params) for use in the functions below."""
        return np.array([self._exporter.map_inputs(single) for single in inputs])

    def parameters_to_inputs(self, p: np.ndarray) -> list[dict]:
        """Converts a 2D parameter array (n_batch, n_params) to a standard set of pybamm input dictionaries."""
        return [self._exporter.inverse_map_inputs(row) for row in np.atleast_2d(p)]

    def predict(self, p: np.ndarray) -> list[pybamm.Solution]:
        """Calculate the solution of the ODE for each set of parameters in inputs.

        The parameter sets are solved together in a single batched solve, so any
        ``num_threads`` parallelism configured on the wrapped solver is used.
        """
        solutions = self._sim.solve(
            t_eval=[0, self._final_time], inputs=self.parameters_to_inputs(p)
        )
        # a single-element batch is returned as a bare Solution
        if not isinstance(solutions, list):
            solutions = [solutions]
        return solutions

    def loss(self, p: np.ndarray) -> np.ndarray:
        """
        Calculate the loss function for each set of parameters in inputs.
        Returns a 1D array of loss values of length n_batch.
        """
        rows = list(np.atleast_2d(p))
        if self._pool is not None:
            values = list(self._pool.map(_worker_loss, rows))
        else:
            values = [self._single_loss(row) for row in rows]
        return np.array(values)

    def loss_and_gradient(
        self, p: np.ndarray, mode: "LossSolverGradientMode"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the loss and gradient of the loss function with respect to the parameters for each set of parameters in inputs.
        Returns a tuple of arrays, where the first contains the loss values as a 1D array of length n_batch and the second contains the gradients as a 2D array of shape (n_batch, n_params)
        """
        rows = list(np.atleast_2d(p))
        if self._pool is not None:
            items = [(row, mode.value) for row in rows]
            results = list(self._pool.map(_worker_loss_and_gradient, items))
        else:
            results = [self._single_loss_and_gradient(row, mode) for row in rows]
        losses = np.array([loss for loss, _ in results])
        gradients = np.array([gradient for _, gradient in results])
        return losses, gradients

    def finite_difference_gradient(self, p: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        Calculate the gradient of the loss function with respect to the parameters for each set of parameters in inputs using finite differencing.
        Returns a 2D array of gradients, with shape (n_batch, n_params), where each row corresponds to the gradient for a given input parameter set.
        """
        p = np.atleast_2d(p).astype(float)
        n_batch, n_params = p.shape
        gradient = np.zeros((n_batch, n_params))
        for j in range(n_params):
            p_plus = p.copy()
            p_plus[:, j] += h
            p_minus = p.copy()
            p_minus[:, j] -= h
            gradient[:, j] = (self.loss(p_plus) - self.loss(p_minus)) / (2 * h)
        return gradient

    def close(self) -> None:
        """Shut down the process pool, if one was created."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> "LossSolver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # the live process pool and casadi functions are not picklable; the pool
        # is recreated on unpickle (except inside worker processes) and the
        # casadi functions are rebuilt
        state["_pool"] = None
        state["_post_sum_node_present"] = self._post_sum is not None
        state["_post_sum"] = None
        state["_post_sum_sens"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        rebuild = state.pop("_post_sum_node_present", False)
        self.__dict__.update(state)
        if rebuild:
            self._post_sum, self._post_sum_sens = self._make_post_sum_functions(
                self._processed_loss.post_sum_node
            )
        # recreate the pool on unpickle, but never inside a worker process,
        # which would spawn nested pools
        if not _IN_WORKER and self._max_workers and self._max_workers > 1:
            self._pool = self._start_pool()

    def _make_post_sum_functions(
        self, post_sum_node: pybamm.Symbol | None
    ) -> tuple[casadi.Function | None, casadi.Function | None]:
        """Build casadi functions for the post-sum node and its sensitivities."""
        if post_sum_node is None:
            return None, None

        input_names = self._exporter.input_names()
        sum_size = self._processed_loss.sum_node.evaluate_for_shape().shape[0]
        n_params = len(input_names)

        t_casadi = casadi.MX.sym("t")
        sum_casadi = casadi.MX.sym("sum", sum_size)
        p_casadi = {
            original: casadi.MX.sym(variable, 1) for original, variable in input_names
        }
        post_sum_casadi = post_sum_node.to_casadi(t_casadi, sum_casadi, inputs=p_casadi)
        p_casadi_stacked = casadi.vertcat(
            *[p_casadi[original] for original, _ in input_names]
        )

        post_sum = casadi.Function(
            "post_sum",
            [t_casadi, sum_casadi, p_casadi_stacked],
            [post_sum_casadi],
        )

        sens_casadi = casadi.MX.sym("sens", sum_size, n_params)
        dpost_dy = casadi.jacobian(post_sum_casadi, sum_casadi)
        dpost_dp = casadi.jacobian(post_sum_casadi, p_casadi_stacked)
        sens = dpost_dy @ sens_casadi + dpost_dp
        post_sum_sens = casadi.Function(
            "post_sum_sens",
            [t_casadi, sum_casadi, p_casadi_stacked, sens_casadi],
            [sens],
        )
        return post_sum, post_sum_sens

    def _start_pool(self) -> concurrent.futures.ProcessPoolExecutor:
        # spawn (not fork) avoids deadlocks with the native threadpools used by
        # the matrix backend; pickling self ships the compiled Ode cheaply.
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(pickle.dumps(self),),
        )

    def _apply_post_sum(
        self, the_integral: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        """Apply the post-sum node (if any) to the summed/integrated inner output.

        ``params`` is the ordered parameter row, which is exactly the stacked
        input vector expected by the post-sum casadi function.
        """
        if self._post_sum is None:
            return the_integral
        return self._post_sum(0.0, the_integral, params).full().reshape(-1)

    def _single_loss(self, params: np.ndarray) -> float:
        """Calculate the scalar loss for a single parameter set."""
        params = np.asarray(params, dtype=float)
        if self._processed_loss.method == "discrete":
            sol = self._ode.solve_dense(params, self._processed_loss.discrete_times)
            the_integral = np.sum(sol.ys, axis=1)
        else:
            sol = self._ode.solve(params, self._final_time)
            the_integral = sol.ys[:, -1]
        value = self._apply_post_sum(the_integral, params)
        return float(np.asarray(value).reshape(-1)[0])

    def _single_loss_and_gradient(
        self, params: np.ndarray, mode: "LossSolverGradientMode"
    ) -> tuple[float, np.ndarray]:
        """Calculate the scalar loss and (n_params,) gradient for a single parameter set."""
        params = np.asarray(params, dtype=float)
        if self._processed_loss.method == "discrete":
            if mode == self.LossSolverGradientMode.FORWARD_SENSITIVITY:
                sol = self._ode.solve_fwd_sens(
                    params, self._processed_loss.discrete_times
                )
                return self._discrete_sum_to_gradient(sol, params)
            raise NotImplementedError(
                "Adjoint sensitivity for discrete sum is not yet implemented"
            )

        if mode == self.LossSolverGradientMode.FORWARD_SENSITIVITY:
            raise NotImplementedError(
                "Forward sensitivity for explicit time integral is not yet implemented"
            )
        integral, integral_sens = self._ode.solve_continuous_adjoint(
            params, self._final_time
        )
        if self._post_sum is None:
            loss = float(np.asarray(integral).reshape(-1)[0])
            gradient = np.asarray(integral_sens).reshape(-1)
            return loss, gradient
        loss = float(self._post_sum(0.0, integral, params).full().reshape(-1)[0])
        gradient = (
            self._post_sum_sens(0.0, integral, params, integral_sens).full().reshape(-1)
        )
        return loss, gradient

    def _discrete_sum_to_gradient(
        self, sol, params: np.ndarray
    ) -> tuple[float, np.ndarray]:
        ys_sum = np.sum(sol.ys, axis=1)
        sens_sum = np.array([np.sum(s, axis=1) for s in sol.sens])
        if self._post_sum is None:
            loss = float(np.asarray(ys_sum).reshape(-1)[0])
            return loss, sens_sum.reshape(-1)
        loss = float(self._post_sum(0.0, ys_sum, params).full().reshape(-1)[0])
        gradient = (
            self._post_sum_sens(0.0, ys_sum, params, sens_sum.T).full().reshape(-1)
        )
        return loss, gradient

    class LossSolverGradientMode(Enum):
        FORWARD_SENSITIVITY = "forward_sensitivity"
        ADJOINT_SENSITIVITY = "adjoint_sensitivity"


_WORKER_SOLVER = None
_IN_WORKER = False


def _init_worker(solver_bytes: bytes) -> None:
    global _WORKER_SOLVER, _IN_WORKER
    _IN_WORKER = True
    _WORKER_SOLVER = pickle.loads(solver_bytes)


def _worker_loss(params: np.ndarray) -> float:
    return _WORKER_SOLVER._single_loss(params)


def _worker_loss_and_gradient(item: tuple) -> tuple[float, np.ndarray]:
    params, mode_value = item
    mode = type(_WORKER_SOLVER).LossSolverGradientMode(mode_value)
    return _WORKER_SOLVER._single_loss_and_gradient(params, mode)
