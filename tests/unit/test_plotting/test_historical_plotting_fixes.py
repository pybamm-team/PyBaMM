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
        # Use an experiment with multiple steps - the second step won't have
        # time starting at exactly 0
        experiment = pybamm.Experiment(
            [
                "Discharge at 1C for 5 minutes",
                "Rest for 1 minute",
                "Discharge at 0.5C for 5 minutes",
            ]
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve()

        # This should not raise an error - the fix ensures ocv(time[0]) works
        # even when time[0] is not exactly 0
        _fig, ax = sol.plot_voltage_components(show_plot=False)

        # Verify plot was created successfully
        assert ax is not None
        assert len(ax.get_lines()) > 0

        # Verify time axis starts near 0
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

        # This should not raise a KeyError for missing overpotential variables
        _fig, ax = sol.plot_voltage_components(show_plot=False)

        # Verify plot was created successfully
        assert ax is not None
        assert len(ax.get_lines()) > 0

        pybamm.close_plots()

    def test_single_phase_electrode_handling(self):
        """
        Guards against: 3da8048b1 - fix single phase number option bug

        When particle phases option is "1" (single phase), the phase name
        should be handled correctly without adding extra spaces or failing.
        """
        # Single phase model (default)
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 3600])

        # Should work with split_by_electrode for single phase
        _fig, ax = sol.plot_voltage_components(show_plot=False, split_by_electrode=True)

        assert ax is not None
        assert len(ax.get_lines()) > 0

        pybamm.close_plots()

    def test_composite_electrode_phases(self):
        """
        Verify composite electrode phase handling works correctly for
        plot_voltage_components with various electrode_phases options.
        """
        model = pybamm.lithium_ion.SPM({"particle phases": ("2", "1")})
        params = pybamm.ParameterValues("Chen2020_composite")
        sim = pybamm.Simulation(model, parameter_values=params)
        sol = sim.solve([0, 3600])

        # Test primary phase
        _fig, ax = sol.plot_voltage_components(
            show_plot=False,
            split_by_electrode=True,
            electrode_phases=("primary", "primary"),
        )
        assert ax is not None

        # Test secondary phase for anode
        _fig, ax = sol.plot_voltage_components(
            show_plot=False,
            split_by_electrode=True,
            electrode_phases=("secondary", "primary"),
        )
        assert ax is not None

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

        # Request time beyond solution range - should be clipped, not error
        quick_plot.plot(200)  # 200 seconds, but solution only goes to 100

        # Verify plot was created
        assert quick_plot.fig is not None

        # Request negative time - should be clipped to min
        quick_plot.plot(-50)
        assert quick_plot.fig is not None

        pybamm.close_plots()

    def test_spatial_variable_interpolation(self):
        """
        Guards against: 9ec472df4 - Fix plotting interpolation bug with spatial
        variables (#4841)

        Spatial variables should be correctly interpolated at the requested
        time points without interpolation errors.
        """
        model = pybamm.lithium_ion.SPMe()
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 100])

        # Test 1D spatial variable
        quick_plot = pybamm.QuickPlot(sol, ["Electrolyte concentration [mol.m-3]"])
        quick_plot.plot(50)

        # Get the y-data (concentration values)
        plot_data = quick_plot.plots[("Electrolyte concentration [mol.m-3]",)][0][0]
        ydata = plot_data.get_ydata()

        # Verify data is valid (no NaN or inf)
        assert not np.any(np.isnan(ydata))
        assert not np.any(np.isinf(ydata))

        pybamm.close_plots()

    def test_slider_update_time_bounds(self):
        """
        Verify slider_update respects time bounds properly.
        """
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 100])

        quick_plot = pybamm.QuickPlot(sol, ["Voltage [V]"])
        quick_plot.plot(0)

        # Update to various times
        quick_plot.slider_update(50)  # Within range
        quick_plot.slider_update(100)  # At max
        quick_plot.slider_update(0)  # At min

        # Should not raise errors
        assert quick_plot.fig is not None

        pybamm.close_plots()

    def test_solution_starting_nonzero_time(self):
        """
        Test QuickPlot handles solutions that don't start at t=0.
        """
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        # Solve from t=50 to t=150
        sol = sim.solve([50, 150])

        quick_plot = pybamm.QuickPlot(sol, ["Voltage [V]"])
        quick_plot.plot(100)  # Mid-point of solution

        # Verify plot was created
        assert quick_plot.fig is not None

        # Verify time scaling is correct
        assert quick_plot.min_t_unscaled == 50
        assert quick_plot.max_t_unscaled == 150

        pybamm.close_plots()


class TestQuickPlot2DVariablesFixes:
    """Guards for 2D variable plotting fixes."""

    def test_2d_variable_time_update(self):
        """
        Verify 2D variables update correctly with time slider.
        """
        model = pybamm.lithium_ion.SPMe()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5}
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        solver = model.default_solver
        t_eval = np.linspace(0, 100, 10)
        solution = solver.solve(model, t_eval)

        output_variables = ["Negative particle concentration [mol.m-3]"]
        quick_plot = pybamm.QuickPlot(solution, output_variables)
        quick_plot.plot(0)

        # Update to different time
        quick_plot.slider_update(50)

        # Get 2D data and verify it's valid
        var_key = ("Negative particle concentration [mol.m-3]",)
        plot_data = quick_plot.plots[var_key][0][1]
        assert not np.any(np.isnan(plot_data))

        pybamm.close_plots()
