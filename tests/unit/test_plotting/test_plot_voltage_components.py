import pytest
import pybamm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use

use("Agg")


class TestPlotVoltageComponents:
    def test_plot_with_solution(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 3600])
        for split in [True, False]:
            _, ax = pybamm.plot_voltage_components(
                sol, show_plot=False, split_by_electrode=split
            )
            t, V = ax.get_lines()[0].get_data()
            np.testing.assert_array_equal(t, sol["Time [h]"].data)
            np.testing.assert_array_equal(V, sol["Battery voltage [V]"].data)

        _, ax = plt.subplots()
        _, ax_out = pybamm.plot_voltage_components(sol, ax=ax, show_legend=True)
        assert ax_out == ax

    def test_plot_with_simulation(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sim.solve([0, 3600])

        for split in [True, False]:
            _, ax = pybamm.plot_voltage_components(
                sim, show_plot=False, split_by_electrode=split
            )
            t, V = ax.get_lines()[0].get_data()
            np.testing.assert_array_equal(t, sim.solution["Time [h]"].data)
            np.testing.assert_array_equal(V, sim.solution["Battery voltage [V]"].data)

        _, ax = plt.subplots()
        _, ax_out = pybamm.plot_voltage_components(sim, ax=ax, show_legend=True)
        assert ax_out == ax

    def test_plot_from_solution(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 3600])
        for split in [True, False]:
            _, ax = sol.plot_voltage_components(
                show_plot=False, split_by_electrode=split
            )
            t, V = ax.get_lines()[0].get_data()
            np.testing.assert_array_equal(t, sol["Time [h]"].data)
            np.testing.assert_array_equal(V, sol["Battery voltage [V]"].data)

        _, ax = plt.subplots()
        _, ax_out = sol.plot_voltage_components(ax=ax, show_legend=True)
        assert ax_out == ax

    def test_plot_from_simulation(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sim.solve([0, 3600])

        for split in [True, False]:
            _, ax = sim.plot_voltage_components(
                show_plot=False, split_by_electrode=split
            )
            t, V = ax.get_lines()[0].get_data()
            np.testing.assert_array_equal(t, sim.solution["Time [h]"].data)
            np.testing.assert_array_equal(V, sim.solution["Battery voltage [V]"].data)

        _, ax = plt.subplots()
        _, ax_out = sim.plot_voltage_components(ax=ax, show_legend=True)
        assert ax_out == ax

    def test_plot_without_solution(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        with pytest.raises(ValueError) as error:
            sim.plot_voltage_components()
            assert str(error.exception) == "The simulation has not been solved yet."
