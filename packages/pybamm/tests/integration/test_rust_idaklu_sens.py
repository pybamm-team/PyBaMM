"""Parity test: Rust IDAKLU forward sensitivities vs CasADi."""

import numpy as np
import pytest

import pybamm

pytest.importorskip("casadi")


def _build_decay_model():
    """ODE: dy/dt = -k * y, with k as InputParameter (sensitivity target)."""
    model = pybamm.BaseModel()
    y = pybamm.Variable("y")
    k = pybamm.InputParameter("k")
    model.rhs = {y: -k * y}
    model.initial_conditions = {y: 1.0}
    return model


DECOUPLED_RATES = {"p0": 0.3, "p1": 1.7, "p2": 0.9}


def _build_decoupled_decay_model(names):
    """Decoupled decays dy_i/dt = -p_i * y_i, so dy_i/dp_j = 0 for i != j."""
    model = pybamm.BaseModel()
    states = [pybamm.Variable(f"y{i}") for i in range(len(names))]
    model.rhs = {
        y: -pybamm.InputParameter(name) * y
        for y, name in zip(states, names, strict=True)
    }
    model.initial_conditions = dict.fromkeys(states, pybamm.Scalar(1.0))
    return model


def _build_coupled_model():
    """Every parameter drives every state, so no sensitivity column is sparse."""
    model = pybamm.BaseModel()
    u = pybamm.Variable("u")
    v = pybamm.Variable("v")
    a = pybamm.InputParameter("a")
    b = pybamm.InputParameter("b")
    c = pybamm.InputParameter("c")
    model.rhs = {u: -(a + b + c) * u + b * v, v: -(a + 2 * b + 3 * c) * v + c * u}
    model.initial_conditions = {u: 1.0, v: 2.0}
    return model


class TestRustIDAKLUSensitivities:
    def test_sensitivity_parity_decay(self):
        t_eval = np.linspace(0, 5, 30)
        inputs = {"k": 0.5}

        model_casadi = _build_decay_model()
        model_casadi.convert_to_format = "casadi"
        sol_casadi = pybamm.IDAKLUSolver().solve(
            model_casadi,
            t_eval,
            inputs=inputs,
            calculate_sensitivities=["k"],
        )
        model_rust = _build_decay_model()
        model_rust.convert_to_format = "rust"
        sol_rust = pybamm.IDAKLUSolver().solve(
            model_rust,
            t_eval,
            inputs=inputs,
            calculate_sensitivities=["k"],
        )

        np.testing.assert_allclose(sol_rust.y, sol_casadi.y, rtol=1e-5, atol=1e-8)

        casadi_sens = sol_casadi.sensitivities["k"]
        rust_sens = sol_rust.sensitivities["k"]
        np.testing.assert_allclose(rust_sens, casadi_sens, rtol=1e-4, atol=1e-7)

    def test_multi_param_sensitivity_columns_land_in_their_own_parameter(self):
        """A stride or ordering slip in the df/dp scatter would swap parameter columns."""
        names = list(DECOUPLED_RATES)
        n_states = len(names)
        t_interp = np.linspace(0, 2, 9)

        model = _build_decoupled_decay_model(names)
        model.convert_to_format = "rust"
        pybamm.Discretisation().process_model(model)
        sol = pybamm.IDAKLUSolver(atol=1e-10, rtol=1e-10).solve(
            model,
            [0, 2],
            inputs=DECOUPLED_RATES,
            calculate_sensitivities=names,
            t_interp=t_interp,
        )

        # Pin the state order the sensitivity assertions below index against.
        for i, name in enumerate(names):
            np.testing.assert_allclose(
                sol.y[i, :],
                np.exp(-DECOUPLED_RATES[name] * sol.t),
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"state {i} is not the decay driven by '{name}'",
            )

        for i, name in enumerate(names):
            expected = np.zeros((len(sol.t), n_states))
            expected[:, i] = -sol.t * np.exp(-DECOUPLED_RATES[name] * sol.t)
            np.testing.assert_allclose(
                np.asarray(sol.sensitivities[name]).reshape(len(sol.t), n_states),
                expected,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"'{name}' sensitivity landed on the wrong state",
            )

    def test_multi_param_dense_sensitivities_match_casadi(self):
        """With no structural zeros, only a value comparison catches a wrong stride."""
        inputs = {"a": 0.4, "b": 0.25, "c": 0.15}
        t_interp = np.linspace(0, 3, 13)

        sols = {}
        for backend in ("casadi", "rust"):
            model = _build_coupled_model()
            model.convert_to_format = backend
            pybamm.Discretisation().process_model(model)
            sols[backend] = pybamm.IDAKLUSolver(atol=1e-10, rtol=1e-10).solve(
                model,
                [0, 3],
                inputs=inputs,
                calculate_sensitivities=list(inputs),
                t_interp=t_interp,
            )

        np.testing.assert_allclose(
            sols["rust"].y, sols["casadi"].y, rtol=1e-6, atol=1e-9
        )

        per_param = {}
        for name in inputs:
            rust_sens = np.asarray(sols["rust"].sensitivities[name])
            per_param[name] = rust_sens
            np.testing.assert_allclose(
                rust_sens,
                np.asarray(sols["casadi"].sensitivities[name]),
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"sensitivity mismatch for parameter '{name}'",
            )

        # Parity only discriminates if the columns actually differ from each other.
        for lhs, rhs in (("a", "b"), ("a", "c"), ("b", "c")):
            assert not np.allclose(
                per_param[lhs], per_param[rhs], rtol=1e-3, atol=1e-6
            ), f"columns '{lhs}' and '{rhs}' are too alike to detect a swap"


class TestRustInitialSensitivities:
    """Algebraic-state sensitivities at t0, at tolerances near roundoff."""

    INPUT = "Current function [A]"

    @staticmethod
    def _model():
        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values.update({"Current function [A]": "[input]"})
        simulation = pybamm.Simulation(
            pybamm.lithium_ion.DFN({"contact resistance": "true"}),
            parameter_values=parameter_values,
            var_pts={"x_n": 6, "x_s": 5, "x_p": 6, "r_n": 8, "r_p": 8},
        )
        simulation.build()
        model = simulation.built_model
        model.convert_to_format = "rust"
        return model

    @pytest.mark.parametrize("newton_mode", ["auto", "full"])
    def test_algebraic_sensitivities_at_t0_match_finite_differences(self, newton_mode):
        model = self._model()
        solver = pybamm.IDAKLUSolver(
            rtol=1e-11, atol=1e-13, options={"newton_mode": newton_mode}
        )
        n_states = model.len_rhs_and_alg
        algebraic = slice(model.len_rhs, n_states)
        for current in (4.5, 4.875, 5.0, 5.125, 5.375, 5.5):
            solution = solver.solve(
                model,
                [0, 1],
                inputs={self.INPUT: current},
                calculate_sensitivities=True,
            )
            # Stored time-major, so the first rows are the states at t0
            sensitivity = solution.sensitivities[self.INPUT][:n_states, 0]

            step = 1e-3 * current
            plus, minus = (
                solver.solve(model, [0, 1], inputs={self.INPUT: current + sign * step})
                for sign in (1, -1)
            )
            expected = (plus.y[algebraic, 0] - minus.y[algebraic, 0]) / (2 * step)
            np.testing.assert_allclose(
                sensitivity[algebraic],
                expected,
                rtol=0,
                atol=1e-5 * np.abs(expected).max(),
                err_msg=f"current {current}",
            )
