"""Keep completed state and callbacks aligned after a failed experiment step."""

from __future__ import annotations

import pytest

import pybamm


class Audit(pybamm.callbacks.Callback):
    def __init__(self):
        self.completed = []
        self.errors = []

    def on_cycle_end(self, logs):
        self.completed.append(logs["cycle number"][0])

    def on_experiment_error(self, logs):
        self.errors.append(logs["cycle number"][0])


def controlled_solver(fail_at):
    class ControlledSolver(pybamm.IDAKLUSolver):
        calls = 0

        def step(self, *args, **kwargs):
            type(self).calls += 1
            if type(self).calls == fail_at:
                raise pybamm.SolverError("controlled step failure")
            result = super().step(*args, **kwargs)
            result.solver_statistics = pybamm.SolverStatistics(number_of_steps=7)
            return result

    return ControlledSolver()


class TestFailureRetention:
    @pytest.mark.parametrize("save_at_cycles", [[1], None, 5])
    @pytest.mark.parametrize("fail_at", [None, 1, 3])
    def test_failure_keeps_latest_completed_state(self, save_at_cycles, fail_at):
        audit = Audit()
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=pybamm.Experiment(["Rest for 1 second"] * 4),
            solver=controlled_solver(fail_at),
        )
        if fail_at == 1:
            with pytest.raises(pybamm.SolverError, match="controlled step failure"):
                sim.solve(
                    save_at_cycles=save_at_cycles, calc_esoh=False, callbacks=[audit]
                )
            assert audit.completed == []
            assert audit.errors == [1]
            return
        sol = sim.solve(
            save_at_cycles=save_at_cycles, calc_esoh=False, callbacks=[audit]
        )
        expected = 4 if fail_at is None else fail_at - 1
        assert sol.t[-1] == pytest.approx(expected)
        assert sol.last_state.t[-1] == pytest.approx(expected)
        assert sol.all_summary_variables[-1].last_state.t[-1] == pytest.approx(expected)
        assert len(sol.cycles) == len(sol.all_first_states) == expected
        assert audit.completed == list(range(1, expected + 1))
        assert audit.errors == ([] if fail_at is None else [fail_at])
        assert sol.solver_statistics.number_of_steps == 7 * expected

    @pytest.mark.parametrize("save_at_cycles", [[1], None])
    def test_resume_after_unsaved_cycle_failure(self, save_at_cycles):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(
            model,
            experiment=pybamm.Experiment(["Rest for 1 second"] * 4),
            solver=controlled_solver(3),
        )
        starting = sim.solve(save_at_cycles=save_at_cycles, calc_esoh=False)
        resumed = pybamm.Simulation(
            model,
            experiment=pybamm.Experiment(["Rest for 1 second"]),
            solver=controlled_solver(None),
        ).solve(starting_solution=starting, calc_esoh=False)
        assert resumed.t[-1] == pytest.approx(3)
        assert resumed.all_summary_variables[-1].last_state.t[-1] == pytest.approx(3)
        assert len(resumed.cycles) == len(resumed.all_first_states) == 3
        assert resumed.solver_statistics.number_of_steps == 21

    def test_partial_cycle_failure_keeps_completed_step(self):
        audit = Audit()
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=pybamm.Experiment(
                [("Rest for 1 second", "Rest for 1 second")] * 4
            ),
            solver=controlled_solver(4),
        )
        sol = sim.solve(save_at_cycles=[1], calc_esoh=False, callbacks=[audit])
        assert sol.t[-1] == pytest.approx(3)
        assert len(sol.cycles) == len(sol.all_first_states) == 2
        assert sol.cycles[-1] is None
        assert sol.all_summary_variables[-1].last_state.t[-1] == pytest.approx(3)
        assert audit.completed == [1, 2]
        assert audit.errors == [2]
        assert sol.solver_statistics.number_of_steps == 21

    @pytest.mark.parametrize(
        "var_pts",
        [
            {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5},
            {"x_n": 9, "x_s": 9, "x_p": 9, "r_n": 9, "r_p": 9},
        ],
    )
    def test_unsaved_failure_retention_is_mesh_independent(self, var_pts):
        """The retained endpoint and callback contract must not depend on mesh size."""
        audit = Audit()
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=pybamm.Experiment(["Rest for 1 second"] * 4),
            solver=controlled_solver(3),
            var_pts=var_pts,
        )
        sol = sim.solve(save_at_cycles=[1], calc_esoh=False, callbacks=[audit])

        assert sol.t[-1] == pytest.approx(2)
        assert sol.last_state.t[-1] == pytest.approx(2)
        assert sol.all_summary_variables[-1].last_state.t[-1] == pytest.approx(2)
        assert len(sol.cycles) == len(sol.all_first_states) == 2
        assert audit.completed == [1, 2]
        assert audit.errors == [3]
        assert sol.solver_statistics.number_of_steps == 14
