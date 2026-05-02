"""
Regression tests for historical plotting bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import numpy as np
from matplotlib import use

import pybamm

use("Agg")


class TestPlotVoltageComponentsHistoricalFixes:
    """Guards for historical plot_voltage_components bug fixes."""

    def test_time_not_starting_at_zero_with_experiment(self):
        """
        Guards against: 4bad8f38f - fix plot_voltage_components to work when time
        doesn't start at 0

        The bug occurred when plotting voltage components for experiments where
        the time array doesn't start at exactly 0 due to floating point or
        experiment step transitions.
        """
        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment(
            [
                "Discharge at 1C for 5 minutes",
                "Rest for 1 minute",
                "Discharge at 0.5C for 5 minutes",
            ]
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve()

        _fig, ax = sol.plot_voltage_components(show_plot=False)

        assert ax is not None
        assert len(ax.get_lines()) > 0

        line_data = ax.get_lines()[0].get_data()
        time_hours = line_data[0]
        assert time_hours[0] >= 0

        pybamm.close_plots()

    def test_half_cell_model_overpotential_variable(self):
        """
        Guards against: 7e4c5ffdf - fix non-existent overpotential variable for
        half cell models

        Half-cell models don't have certain overpotential variables that
        full-cell models have. The plotting function must handle this gracefully.
        """
        model = pybamm.lithium_ion.SPM({"working electrode": "positive"})
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 3600])

        _fig, ax = sol.plot_voltage_components(show_plot=False)

        assert ax is not None
        assert len(ax.get_lines()) > 0

        pybamm.close_plots()


class TestQuickPlotHistoricalFixes:
    """Guards for historical QuickPlot bug fixes."""

    def test_extrapolation_time_clipping(self):
        """
        Guards against: c2bf6ec16 - Fix `QuickPlot` extrapolation bug (#4991)

        When a time value beyond the solution range is requested, it should
        be clipped to the valid range instead of causing extrapolation errors.
        """
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 100])

        quick_plot = pybamm.QuickPlot(sol, ["Voltage [V]"])

        quick_plot.plot(200)
        assert quick_plot.fig is not None

        quick_plot.plot(-50)
        assert quick_plot.fig is not None

        pybamm.close_plots()

    def test_solution_starting_nonzero_time(self):
        """
        Test QuickPlot handles solutions that don't start at t=0.
        """
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        sol = sim.solve([50, 150])

        quick_plot = pybamm.QuickPlot(sol, ["Voltage [V]"])
        quick_plot.plot(100)

        assert quick_plot.fig is not None
        assert quick_plot.min_t_unscaled == 50
        assert quick_plot.max_t_unscaled == 150

        pybamm.close_plots()

    def test_spatial_variable_interpolation_no_nan(self):
        """
        Guards against: 9ec472df4 - Fix plotting interpolation bug with spatial
        variables (#4841)

        Spatial variables should be correctly interpolated without producing NaN.
        """
        model = pybamm.lithium_ion.SPMe()
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 100])

        quick_plot = pybamm.QuickPlot(sol, ["Electrolyte concentration [mol.m-3]"])
        quick_plot.plot(50)

        plot_data = quick_plot.plots[("Electrolyte concentration [mol.m-3]",)][0][0]
        ydata = plot_data.get_ydata()

        assert not np.any(np.isnan(ydata))
        assert not np.any(np.isinf(ydata))

        pybamm.close_plots()
