"""Coloured Jacobian assembly batches its tangent sweeps without changing values.

The Rust core evaluates the tangent tape once per colour; batching runs several
colours per sweep to amortise the sparse-operator gather. The rust-side property
tests pin bitwise equality of the sweep itself, so these pin the parts visible
from Python: that a real battery model actually engages batching, that a model
too small to benefit does not, and that the assembled matrix still matches
CasADi's Jacobian of the same discretised model.
"""

import numpy as np
import pytest

import pybamm
from tests.shared import (
    POUCH_OPTIONS,
    POUCH_PTS,
    build_rust_model,
    dense_casadi_jacobian,
    dense_rust_jacobian,
)

pytest.importorskip("casadi")


class TestLaneWidth:
    def test_a_dfn_batches_its_colour_sweeps(self):
        _, rust_model = build_rust_model(pybamm.lithium_ion.DFN())
        stats = rust_model.jacobian_stats()
        assert stats["n_colors"] > 1
        assert stats["jac_lane_width"] > 1

    def test_a_single_colour_model_stays_scalar(self):
        # Nonlinear, so the one entry is genuinely swept rather than folded.
        model = pybamm.BaseModel()
        model.convert_to_format = "rust"
        var = pybamm.Variable("var")
        model.rhs = {var: -(var**2)}
        model.initial_conditions = {var: 1.0}
        solver = pybamm.IDAKLUSolver()
        solver.set_up(model, inputs=[{}])
        stats = solver._setup["rust_model"].jacobian_stats()
        assert stats["n_colors"] == 1
        assert stats["jac_lane_width"] == 1

    def test_a_linear_model_needs_no_colour_at_all(self):
        # Every entry folds at compile time, so assembly is a table write.
        model = pybamm.BaseModel()
        model.convert_to_format = "rust"
        var = pybamm.Variable("var")
        model.rhs = {var: -var}
        model.initial_conditions = {var: 1.0}
        solver = pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-10)
        solver.set_up(model, inputs=[{}])
        stats = solver._setup["rust_model"].jacobian_stats()
        assert stats["n_colors"] == 0
        assert stats["n_swept_columns"] == 0
        assert stats["n_constant_entries"] == 1

        t = np.linspace(0, 1, 5)
        solution = solver.solve(model, t_eval=[0, 1], t_interp=t)
        np.testing.assert_allclose(solution["var"](t), np.exp(-t), rtol=1e-7, atol=1e-7)


class TestAssembledValues:
    def test_batched_assembly_matches_casadi(self):
        """The batched sweep must reproduce CasADi's Jacobian of the same model."""
        built, rust_model = build_rust_model(pybamm.lithium_ion.DFN())
        assert rust_model.jacobian_stats()["jac_lane_width"] > 1

        y, got = dense_rust_jacobian(built, rust_model)
        want = dense_casadi_jacobian(pybamm.lithium_ion.DFN(), y)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)

    def test_pouch_assembly_matches_casadi(self):
        """The pouch is the model whose colouring the constant split changes."""
        var_pts = POUCH_PTS | {"y": 4, "z": 4}
        built, rust_model = build_rust_model(
            pybamm.lithium_ion.DFN(POUCH_OPTIONS), var_pts
        )
        stats = rust_model.jacobian_stats()
        assert stats["n_constant_entries"] > 0
        assert stats["n_swept_columns"] < built.len_rhs_and_alg

        y, got = dense_rust_jacobian(built, rust_model)
        want = dense_casadi_jacobian(pybamm.lithium_ion.DFN(POUCH_OPTIONS), y, var_pts)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
