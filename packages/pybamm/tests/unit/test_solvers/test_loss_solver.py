#
# Tests for the LossSolver class
#
import importlib.util
import pickle
import sys

import numpy as np
import pytest

import pybamm
from pybamm.simulation.loss_solver import LossSolver

has_pydiffsol = importlib.util.find_spec("pydiffsol") is not None
is_windows = sys.platform == "win32"

K_TRUE = 0.5
K_OTHER = 0.8
FINAL_TIME = 2.0
N_DATA = 11
Y0 = 1.0


def _decay_model():
    """dy/dt = -k * y, y(0) = 1, with analytic solution y(t) = exp(-k * t)."""
    model = pybamm.BaseModel("exponential decay")
    y = pybamm.Variable("y")
    k = pybamm.InputParameter("k")
    model.rhs = {y: -k * y}
    model.initial_conditions = {y: pybamm.Scalar(Y0)}
    model.variables = {"y": y}
    return model


def _data_times():
    return np.linspace(0, FINAL_TIME, N_DATA)


def _analytic_solution(k, t):
    return Y0 * np.exp(-k * np.asarray(t, dtype=float))


def _discrete_loss(k):
    t = _data_times()
    residual = _analytic_solution(k, t) - _analytic_solution(K_TRUE, t)
    return np.sum(residual**2)


def _discrete_loss_gradient(k):
    t = _data_times()
    residual = _analytic_solution(k, t) - _analytic_solution(K_TRUE, t)
    dydk = -t * _analytic_solution(k, t)
    return np.sum(2 * residual * dydk)


def _continuous_loss(k):
    return (1 - np.exp(-2 * k * FINAL_TIME)) / (2 * k)


def _continuous_loss_gradient(k):
    e = np.exp(-2 * k * FINAL_TIME)
    return (2 * k * FINAL_TIME * e - (1 - e)) / (2 * k**2)


def _discrete_loss_function():
    t = _data_times()
    data = pybamm.DiscreteTimeData(t, _analytic_solution(K_TRUE, t), "decay data")
    return pybamm.DiscreteTimeSum((data - pybamm.Variable("y")) ** 2)


def _continuous_loss_function():
    return pybamm.ExplicitTimeIntegral(pybamm.Variable("y") ** 2, pybamm.Scalar(0))


def _make_loss_solver(loss_function, max_workers=None):
    sim = pybamm.Simulation(
        _decay_model(), solver=pybamm.IDAKLUSolver(rtol=1e-9, atol=1e-9)
    )
    return LossSolver(sim, loss_function, FINAL_TIME, max_workers=max_workers)


@pytest.mark.skipif(not has_pydiffsol, reason="pydiffsol is not installed")
class TestLossSolver:
    @pytest.fixture
    def continuous_solver(self):
        return _make_loss_solver(_continuous_loss_function())

    def test_init_raises_without_time_integral(self):
        sim = pybamm.Simulation(_decay_model())
        with pytest.raises(ValueError, match=r"DiscreteSum or an ExplicitTimeIntegral"):
            LossSolver(sim, pybamm.Variable("y"), FINAL_TIME)

    def test_init_raises_on_duplicate_inner_name(self):
        sim = pybamm.Simulation(_decay_model())
        sim.model.variables[LossSolver.INNER_LOSS_FUNCTION_NAME] = pybamm.Scalar(0)
        with pytest.raises(ValueError, match=r"already contains a variable named"):
            LossSolver(sim, _continuous_loss_function(), FINAL_TIME)

    def test_inputs_to_parameters(self, continuous_solver):
        p = continuous_solver.inputs_to_parameters([{"k": K_TRUE}])
        assert p.shape == (1, 1)
        np.testing.assert_allclose(p, [[K_TRUE]])

    def test_parameters_to_inputs_round_trip(self, continuous_solver):
        p = continuous_solver.inputs_to_parameters([{"k": K_TRUE}])
        assert continuous_solver.parameters_to_inputs(p) == [{"k": K_TRUE}]

    def test_predict_matches_analytic(self, continuous_solver):
        p = continuous_solver.inputs_to_parameters([{"k": K_TRUE}])
        solution = continuous_solver.predict(p)[0]
        t = _data_times()
        np.testing.assert_allclose(
            solution["y"](t), _analytic_solution(K_TRUE, t), atol=1e-4
        )

    def test_loss_continuous_matches_analytic(self, continuous_solver):
        p = continuous_solver.inputs_to_parameters([{"k": K_TRUE}])
        np.testing.assert_allclose(
            continuous_solver.loss(p), [_continuous_loss(K_TRUE)], atol=1e-6
        )

    def test_finite_difference_gradient_matches_analytic(self, continuous_solver):
        p = continuous_solver.inputs_to_parameters([{"k": K_TRUE}])
        gradient = continuous_solver.finite_difference_gradient(p)
        np.testing.assert_allclose(
            gradient, [[_continuous_loss_gradient(K_TRUE)]], rtol=3e-5, atol=1e-6
        )

    @pytest.mark.skipif(
        is_windows, reason="adjoint sensitivity not available on Windows"
    )
    def test_loss_and_gradient_continuous_adjoint(self, continuous_solver):
        p = continuous_solver.inputs_to_parameters([{"k": K_TRUE}])
        loss, gradient = continuous_solver.loss_and_gradient(
            p, LossSolver.LossSolverGradientMode.ADJOINT_SENSITIVITY
        )
        np.testing.assert_allclose(loss, [_continuous_loss(K_TRUE)], atol=1e-6)
        np.testing.assert_allclose(
            gradient, [[_continuous_loss_gradient(K_TRUE)]], rtol=1e-3, atol=1e-5
        )

    @pytest.mark.skipif(
        is_windows, reason="forward sensitivity not available on Windows"
    )
    def test_loss_and_gradient_continuous_forward_not_implemented(
        self, continuous_solver
    ):
        p = continuous_solver.inputs_to_parameters([{"k": K_TRUE}])
        with pytest.raises(NotImplementedError):
            continuous_solver.loss_and_gradient(
                p, LossSolver.LossSolverGradientMode.FORWARD_SENSITIVITY
            )

    def test_inputs_to_parameters_batch(self, continuous_solver):
        p = continuous_solver.inputs_to_parameters([{"k": K_TRUE}, {"k": K_OTHER}])
        assert p.shape == (2, 1)
        np.testing.assert_allclose(p, [[K_TRUE], [K_OTHER]])

    def test_parameters_to_inputs_batch_round_trip(self, continuous_solver):
        p = continuous_solver.inputs_to_parameters([{"k": K_TRUE}, {"k": K_OTHER}])
        assert continuous_solver.parameters_to_inputs(p) == [
            {"k": K_TRUE},
            {"k": K_OTHER},
        ]

    def test_predict_batch_matches_analytic(self, continuous_solver):
        p = continuous_solver.inputs_to_parameters([{"k": K_TRUE}, {"k": K_OTHER}])
        solutions = continuous_solver.predict(p)
        assert len(solutions) == 2
        t = _data_times()
        for solution, k in zip(solutions, (K_TRUE, K_OTHER), strict=True):
            np.testing.assert_allclose(
                solution["y"](t), _analytic_solution(k, t), atol=1e-4
            )

    def test_loss_batch_matches_analytic(self, continuous_solver):
        p = continuous_solver.inputs_to_parameters([{"k": K_TRUE}, {"k": K_OTHER}])
        loss = continuous_solver.loss(p)
        assert loss.shape == (2,)
        np.testing.assert_allclose(
            loss, [_continuous_loss(K_TRUE), _continuous_loss(K_OTHER)], atol=1e-6
        )

    def test_finite_difference_gradient_batch(self, continuous_solver):
        p = continuous_solver.inputs_to_parameters([{"k": K_TRUE}, {"k": K_OTHER}])
        gradient = continuous_solver.finite_difference_gradient(p)
        assert gradient.shape == (2, 1)
        np.testing.assert_allclose(
            gradient,
            [[_continuous_loss_gradient(K_TRUE)], [_continuous_loss_gradient(K_OTHER)]],
            rtol=3e-5,
            atol=1e-6,
        )

    @pytest.mark.skipif(
        is_windows, reason="adjoint sensitivity not available on Windows"
    )
    def test_loss_and_gradient_batch_adjoint(self, continuous_solver):
        p = continuous_solver.inputs_to_parameters([{"k": K_TRUE}, {"k": K_OTHER}])
        loss, gradient = continuous_solver.loss_and_gradient(
            p, LossSolver.LossSolverGradientMode.ADJOINT_SENSITIVITY
        )
        assert loss.shape == (2,)
        assert gradient.shape == (2, 1)
        np.testing.assert_allclose(
            loss, [_continuous_loss(K_TRUE), _continuous_loss(K_OTHER)], atol=1e-6
        )
        np.testing.assert_allclose(
            gradient,
            [[_continuous_loss_gradient(K_TRUE)], [_continuous_loss_gradient(K_OTHER)]],
            rtol=1e-3,
            atol=1e-5,
        )

    @pytest.mark.skipif(is_windows, reason="pickling not supported on Windows")
    def test_pickle_round_trip(self, continuous_solver):
        p = continuous_solver.inputs_to_parameters([{"k": K_TRUE}, {"k": K_OTHER}])
        expected = continuous_solver.loss(p)
        restored = pickle.loads(pickle.dumps(continuous_solver))
        np.testing.assert_allclose(restored.loss(p), expected)

    @pytest.mark.skipif(is_windows, reason="pickling not supported on Windows")
    def test_pickle_round_trip_restores_pool(self):
        sequential = _make_loss_solver(_continuous_loss_function())
        p = sequential.inputs_to_parameters([{"k": K_TRUE}, {"k": K_OTHER}])
        expected = sequential.loss(p)

        parallel = _make_loss_solver(_continuous_loss_function(), max_workers=2)
        blob = pickle.dumps(parallel)
        parallel.close()

        restored = pickle.loads(blob)
        try:
            assert restored._pool is not None
            np.testing.assert_allclose(restored.loss(p), expected)
        finally:
            restored.close()

    @pytest.mark.skipif(
        is_windows, reason="adjoint sensitivity not available on Windows"
    )
    def test_parallel_matches_sequential(self):
        inputs = [{"k": k} for k in (0.4, 0.6, 0.8, 1.0)]
        sequential = _make_loss_solver(_continuous_loss_function())
        parallel = _make_loss_solver(_continuous_loss_function(), max_workers=2)
        mode = LossSolver.LossSolverGradientMode.ADJOINT_SENSITIVITY
        try:
            p = sequential.inputs_to_parameters(inputs)
            np.testing.assert_allclose(parallel.loss(p), sequential.loss(p))
            seq_loss, seq_grad = sequential.loss_and_gradient(p, mode)
            par_loss, par_grad = parallel.loss_and_gradient(p, mode)
            np.testing.assert_allclose(par_loss, seq_loss)
            np.testing.assert_allclose(par_grad, seq_grad)
            np.testing.assert_allclose(
                parallel.finite_difference_gradient(p),
                sequential.finite_difference_gradient(p),
            )
        finally:
            parallel.close()

    def test_loss_discrete_zero_at_true(self):
        solver = _make_loss_solver(_discrete_loss_function())
        p = solver.inputs_to_parameters([{"k": K_TRUE}])
        np.testing.assert_allclose(solver.loss(p), [0.0], atol=1e-5)

    def test_loss_discrete_matches_analytic_off_true(self):
        solver = _make_loss_solver(_discrete_loss_function())
        p_true = solver.inputs_to_parameters([{"k": K_TRUE}])
        p_other = solver.inputs_to_parameters([{"k": K_OTHER}])
        loss_other = solver.loss(p_other)
        np.testing.assert_allclose(loss_other, [_discrete_loss(K_OTHER)], atol=1e-6)
        assert loss_other[0] > solver.loss(p_true)[0]

    @pytest.mark.skipif(
        is_windows, reason="forward sensitivity not available on Windows"
    )
    def test_loss_and_gradient_discrete_forward(self):
        solver = _make_loss_solver(_discrete_loss_function())
        p = solver.inputs_to_parameters([{"k": K_OTHER}])
        loss, gradient = solver.loss_and_gradient(
            p, LossSolver.LossSolverGradientMode.FORWARD_SENSITIVITY
        )
        np.testing.assert_allclose(loss, solver.loss(p), atol=1e-6)
        np.testing.assert_allclose(
            gradient, [[_discrete_loss_gradient(K_OTHER)]], rtol=3e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            gradient, solver.finite_difference_gradient(p), rtol=3e-5, atol=1e-6
        )

    @pytest.mark.skipif(
        is_windows, reason="forward sensitivity not available on Windows"
    )
    def test_loss_and_gradient_discrete_forward_zero_at_true(self):
        solver = _make_loss_solver(_discrete_loss_function())
        p = solver.inputs_to_parameters([{"k": K_TRUE}])
        _, gradient = solver.loss_and_gradient(
            p, LossSolver.LossSolverGradientMode.FORWARD_SENSITIVITY
        )
        np.testing.assert_allclose(gradient, [[0.0]], atol=5e-4)

    @pytest.mark.skipif(
        is_windows, reason="adjoint sensitivity not available on Windows"
    )
    def test_loss_and_gradient_discrete_adjoint(self):
        solver = _make_loss_solver(_discrete_loss_function())
        p = solver.inputs_to_parameters([{"k": K_OTHER}])
        loss, gradient = solver.loss_and_gradient(
            p, LossSolver.LossSolverGradientMode.ADJOINT_SENSITIVITY
        )
        np.testing.assert_allclose(loss, solver.loss(p), atol=1e-6)
        np.testing.assert_allclose(
            gradient, [[_discrete_loss_gradient(K_OTHER)]], rtol=1e-3, atol=1e-5
        )
        np.testing.assert_allclose(
            gradient,
            solver.finite_difference_gradient(p),
            rtol=1e-3,
            atol=1e-5,
        )

    @pytest.mark.skipif(
        is_windows, reason="adjoint sensitivity not available on Windows"
    )
    def test_loss_and_gradient_discrete_adjoint_zero_at_true(self):
        solver = _make_loss_solver(_discrete_loss_function())
        p = solver.inputs_to_parameters([{"k": K_TRUE}])
        _, gradient = solver.loss_and_gradient(
            p, LossSolver.LossSolverGradientMode.ADJOINT_SENSITIVITY
        )
        np.testing.assert_allclose(gradient, [[0.0]], atol=5e-4)
