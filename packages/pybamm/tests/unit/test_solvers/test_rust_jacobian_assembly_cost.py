"""Budgets on the tangent-tape walks one Jacobian assembly costs.

This count is fixed at compile time, so unlike wall times and solver counters it
survives contention and trajectory chaos. Budgets allow a lane-width halving;
raising one means confirming the model grew or that assembly got cheaper.
"""

import numpy as np
import pytest

import pybamm
from tests.shared import POUCH_OPTIONS, POUCH_PTS, build_rust_model


def sweeps_per_assembly(stats):
    """Batched colour sweeps plus one reverse pass per split dense row."""
    batched = -(-stats["n_colors"] // stats["jac_lane_width"])
    return batched + stats["n_dense_rows"]


def _stats_for_model(model, var_pts=None):
    """`(jacobian_stats, n_states)` for a plainly built model."""
    built, rust_model = build_rust_model(model, var_pts)
    return rust_model.jacobian_stats(), built.len_rhs_and_alg


def _stats_for_step(model, step, var_pts, key_prefix):
    """`(jacobian_stats, n_states)` for one experiment step's own model."""
    model.convert_to_format = "rust"
    sim = pybamm.Simulation(
        model, var_pts=var_pts, experiment=pybamm.Experiment([step])
    )
    sim.build_for_experiment()
    built = next(
        m for key, m in sim.steps_to_built_models.items() if key.startswith(key_prefix)
    )
    solver = pybamm.IDAKLUSolver()
    solver.set_up(built, inputs={}, t_eval=np.array([0.0, 1.0]))
    return solver._setup["rust_model"].jacobian_stats(), built.len_rhs_and_alg


PLAIN_CASES = [
    ("SPM", lambda: pybamm.lithium_ion.SPM(), None, 1),
    ("SPMe", lambda: pybamm.lithium_ion.SPMe(), None, 3),
    (
        "SPMe-voltage-as-a-state",
        lambda: pybamm.lithium_ion.SPMe({"voltage as a state": "true"}),
        None,
        4,
    ),
    ("DFN", lambda: pybamm.lithium_ion.DFN(), None, 3),
    (
        "DFN-fine",
        lambda: pybamm.lithium_ion.DFN(),
        {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 30, "r_p": 30},
        3,
    ),
]

STEP_CASES = [
    ("pouch-2plus1D-CC", POUCH_OPTIONS, "Discharge at 1C until 2.7 V", "CRate", 3),
    ("pouch-2plus1D-CV", POUCH_OPTIONS, "Hold at 4.1 V until C/20", "Voltage", 4),
    (
        "pouch-1plus1D-CV",
        {"current collector": "potential pair", "dimensionality": 1},
        "Hold at 4.1 V until C/20",
        "Voltage",
        3,
    ),
]


class TestAssemblySweepBudget:
    @pytest.mark.parametrize(
        ("name", "build", "var_pts", "budget"),
        PLAIN_CASES,
        ids=[case[0] for case in PLAIN_CASES],
    )
    def test_plain_model_stays_within_sweep_budget(self, name, build, var_pts, budget):
        stats, _ = _stats_for_model(build(), var_pts)
        assert sweeps_per_assembly(stats) <= budget, (
            f"{name} assembles in {sweeps_per_assembly(stats)} sweeps "
            f"(budget {budget}): {stats}"
        )

    @pytest.mark.parametrize(
        ("name", "options", "step", "prefix", "budget"),
        STEP_CASES,
        ids=[case[0] for case in STEP_CASES],
    )
    def test_experiment_step_stays_within_sweep_budget(
        self, name, options, step, prefix, budget
    ):
        var_pts = POUCH_PTS if options["dimensionality"] == 2 else None
        if options["dimensionality"] == 1:
            var_pts = {"x_n": 4, "x_s": 4, "x_p": 4, "r_n": 4, "r_p": 4, "z": 10}
        stats, _ = _stats_for_step(
            pybamm.lithium_ion.DFN(options), step, var_pts, prefix
        )
        assert sweeps_per_assembly(stats) <= budget, (
            f"{name} assembles in {sweeps_per_assembly(stats)} sweeps "
            f"(budget {budget}): {stats}"
        )


class TestNoWideRowEscapesTheSplit:
    def test_a_constraint_row_never_sets_the_colour_count(self):
        """A row far wider than the rest must be split, not coloured against."""
        stats, _ = _stats_for_step(
            pybamm.lithium_ion.DFN(POUCH_OPTIONS),
            "Hold at 4.1 V until C/20",
            POUCH_PTS,
            "Voltage",
        )
        assert stats["n_dense_rows"] == 1
        assert stats["dense_row_entries"] > 10 * stats["n_colors"]

    def test_batching_engages_on_a_multi_colour_model(self):
        """Batching turning itself off multiplies assembly cost by the lane width."""
        for stats, _ in (
            _stats_for_model(pybamm.lithium_ion.DFN()),
            _stats_for_step(
                pybamm.lithium_ion.DFN(POUCH_OPTIONS),
                "Hold at 4.1 V until C/20",
                POUCH_PTS,
                "Voltage",
            ),
        ):
            assert stats["n_colors"] > 1
            assert stats["jac_lane_width"] > 1


class TestConstantEntriesCarryTheirWeight:
    """The classifier silently proving nothing would cost only sweeps, so the
    budgets above would still pass. These pin that it fires."""

    def test_a_dfn_proves_most_of_its_entries_constant(self):
        stats, n_states = _stats_for_model(pybamm.lithium_ion.DFN())
        assert stats["n_constant_entries"] > stats["nnz"] // 2
        assert stats["n_swept_columns"] < n_states

    def test_the_pouch_stops_colouring_against_its_constant_rows(self):
        stats, n_states = _stats_for_step(
            pybamm.lithium_ion.DFN(POUCH_OPTIONS),
            "Discharge at 1C until 2.7 V",
            POUCH_PTS,
            "CRate",
        )
        assert stats["n_constant_entries"] > stats["nnz"] // 10
        # The current-collector rows are wholly constant, so their columns are
        # never seeded and the coloring does not have to separate them.
        assert stats["n_swept_columns"] < n_states
