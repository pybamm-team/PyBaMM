"""
Memory leak tests for PyBaMM simulations.

These tests use pytest-memray to detect memory leaks in long-running simulations.
Run locally (Linux/macOS only): nox -s memory

Leak thresholds are set tight (100 KB) to catch real leaks. If tests fail,
investigate the reported allocation sites - common culprits:
- casadi function serialization caching
- idaklu_solver._integrate allocations not being freed
"""

import pytest

import pybamm

pytest.importorskip("memray", reason="pytest-memray required for memory tests")


class TestSimulationMemoryLeaks:
    @pytest.mark.limit_leaks("100 KB")
    def test_spm_gitt_no_leak(self):
        experiment = pybamm.Experiment(
            ["Discharge at C/20 for 1 hour", "Rest for 1 hour"] * 5,
            period="6 minutes",
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.solve()

    @pytest.mark.limit_leaks("100 KB")
    def test_dfn_gitt_no_leak(self):
        experiment = pybamm.Experiment(
            ["Discharge at C/20 for 1 hour", "Rest for 1 hour"] * 5,
            period="6 minutes",
        )
        model = pybamm.lithium_ion.DFN()
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
        sim.solve()

    @pytest.mark.limit_leaks("100 KB")
    def test_spm_cccv_cycling_no_leak(self):
        cycle = [
            "Discharge at C/5 for 10 hours or until 3.3 V",
            "Rest for 1 hour",
            "Charge at 1 A until 4.1 V",
            "Hold at 4.1 V until 10 mA",
            "Rest for 1 hour",
        ]
        experiment = pybamm.Experiment(cycle * 3)
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.solve()

    @pytest.mark.limit_leaks("100 KB")
    def test_spme_long_cycling_no_leak(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at 1C until 3.0 V",
                "Charge at 0.5C until 4.2 V",
                "Hold at 4.2 V until C/50",
            ]
            * 5
        )
        model = pybamm.lithium_ion.SPMe()
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
        sim.solve()


class TestSolverMemoryLeaks:
    @pytest.mark.limit_leaks("100 KB")
    def test_repeated_solve_no_leak(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        for _ in range(5):
            sim.solve([0, 3600])

    @pytest.mark.limit_leaks("100 KB")
    def test_repeated_simulation_creation_no_leak(self):
        for _ in range(5):
            model = pybamm.lithium_ion.SPM()
            sim = pybamm.Simulation(model)
            sim.solve([0, 3600])


class TestSolutionMemoryLeaks:
    @pytest.mark.limit_leaks("100 KB")
    def test_solution_variable_access_no_leak(self):
        experiment = pybamm.Experiment(
            ["Discharge at C/20 for 1 hour", "Rest for 1 hour"] * 3
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve()

        for _ in range(10):
            _ = sol["Voltage [V]"].entries
            _ = sol["Current [A]"].entries
            _ = sol["Time [h]"].entries

    @pytest.mark.limit_leaks("100 KB")
    def test_cycle_access_no_leak(self):
        experiment = pybamm.Experiment(
            ["Discharge at C/20 for 1 hour", "Rest for 1 hour"] * 5
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve()

        for _ in range(5):
            for cycle in sol.cycles:
                _ = cycle["Voltage [V]"].entries
