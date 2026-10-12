"""Sparse exports preserve retained sample provenance and numerical data."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pybamm


class TestSparseExport:
    @staticmethod
    def solve(save_at_cycles, skipped=False):
        steps = (
            ("Charge at 1 A until 2 V", "Rest for 1 second")
            if skipped
            else ("Discharge at 1 A for 1 second", "Rest for 1 second")
        )
        return pybamm.Simulation(
            pybamm.lithium_ion.SPM(), experiment=pybamm.Experiment([steps] * 4)
        ).solve(save_at_cycles=save_at_cycles, calc_esoh=False)

    @pytest.mark.parametrize("save_at_cycles", [[1], 2, None])
    def test_sparse_export(self, save_at_cycles):
        solution = self.solve(save_at_cycles)
        expected = solution.get_data_dict(
            ["Time [s]", "Voltage [V]"], cycles_and_steps=False
        )
        data = solution.get_data_dict(["Time [s]", "Voltage [V]"])
        csv = pd.read_csv(
            io.StringIO(
                solution.save_data(
                    variables=["Time [s]", "Voltage [V]"], to_format="csv"
                )
            )
        )
        for name in expected:
            np.testing.assert_array_equal(data[name], expected[name])
            np.testing.assert_allclose(csv[name], expected[name], rtol=0, atol=1e-12)
        assert len(data["Cycle"]) == len(data["Step"]) == solution.t.size
        for index, cycle in enumerate(solution.cycles):
            if cycle is not None:
                mask = np.isin(solution.t, cycle.t)
                np.testing.assert_array_equal(data["Cycle"][mask], index)
        assert data["Cycle"][-1] == pytest.approx(3)
        assert data["Step"][-1] == pytest.approx(1)

    def test_terminal_only_sample(self):
        solution = self.solve([1])
        endpoint = solution.all_summary_variables[2].last_state
        prefix = solution.cycles[0] + endpoint
        prefix.cycles = solution.cycles[:3]
        prefix.all_summary_variables = solution.all_summary_variables[:3]
        data = prefix.get_data_dict(["Time [s]", "Voltage [V]"])
        assert data["Cycle"][-1] == pytest.approx(2)
        assert data["Step"][-1] == pytest.approx(-1)
        np.testing.assert_allclose(data["Time [s]"], prefix.t, rtol=0, atol=1e-12)
        assert len(data["Step"]) == prefix.t.size

    def test_skipped_initial_step(self):
        solution = self.solve([1], skipped=True)
        data = solution.get_data_dict(["Time [s]", "Voltage [V]"])
        endpoint = solution.all_summary_variables[2].last_state.t[-1]
        expected = np.ones_like(solution.t)
        expected[solution.t == endpoint] = -1
        np.testing.assert_array_equal(data["Step"], expected)
        np.testing.assert_array_equal(data["Cycle"][solution.t == endpoint], 2)
        assert len(data["Cycle"]) == solution.t.size

    def test_resumed_sparse_history(self):
        solution = self.solve([1])
        resumed = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=pybamm.Experiment(["Rest for 1 second"]),
        ).solve(starting_solution=solution, calc_esoh=False)
        data = resumed.get_data_dict(["Time [s]"])
        assert data["Cycle"][-1] == pytest.approx(4)
        assert data["Step"][-1] == pytest.approx(0)
        assert len(data["Cycle"]) == len(data["Step"]) == resumed.t.size

    def test_shared_step_boundary_keeps_first_owner(self):
        model = pybamm.BaseModel()
        state = pybamm.Variable("state")
        model.rhs = {state: -state}
        model.initial_conditions = {state: 1}
        model.variables = {"state": state}
        solver = pybamm.IDAKLUSolver()
        first = solver.solve(model, [0, 1], t_interp=[0, 0.5, 1])
        second = solver.solve(model, [1, 2], t_interp=[1, 1.5, 2])
        cycle = first + second
        cycle.steps = [first, second]
        solution = cycle.copy()
        solution.cycles = [cycle, None]
        data = solution.get_data_dict("state")
        np.testing.assert_array_equal(data["Step"], [0, 0, 0, 1, 1])
        np.testing.assert_array_equal(data["Cycle"], 0)

    @pytest.mark.parametrize("endpoints", [[1, 1], [2], []])
    def test_unknown_or_ambiguous_provenance_raises(self, endpoints):
        model = pybamm.BaseModel()
        solution = pybamm.Solution(np.array([1]), np.ones((1, 1)), model, {})
        solution.cycles = [None] * len(endpoints)
        solution.all_summary_variables = [
            SimpleNamespace(last_state=SimpleNamespace(t=np.array([time])))
            for time in endpoints
        ]
        with pytest.raises(pybamm.ModelError, match="cycle provenance"):
            solution._get_sparse_cycle_and_step_labels()
