import os
import pybamm
import unittest
import numpy as np


class TestQuickPlot(unittest.TestCase):
    def test_simple_ode_model(self):
        model = pybamm.lithium_ion.BaseModel(name="Simple ODE Model")

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        # Create variables: domain is explicitly empty since these variables are only
        # functions of time
        a = pybamm.Variable("a", domain=[])
        b = pybamm.Variable("b", domain=[])
        c = pybamm.Variable("c", domain=[])

        # Simple ODEs
        model.rhs = {a: pybamm.Scalar(0.2), b: pybamm.Scalar(0), c: -c}

        # Simple initial conditions
        model.initial_conditions = {
            a: pybamm.Scalar(0),
            b: pybamm.Scalar(1),
            c: pybamm.Scalar(1),
        }
        # no boundary conditions for an ODE model
        # Broadcast some of the variables
        model.variables = {
            "a": a,
            "b": b,
            "b broadcasted": pybamm.FullBroadcast(b, whole_cell, "current collector"),
            "c broadcasted": pybamm.FullBroadcast(
                c, ["negative electrode", "separator"], "current collector"
            ),
            "b broadcasted negative electrode": pybamm.PrimaryBroadcast(
                b, "negative particle"
            ),
            "c broadcasted positive electrode": pybamm.PrimaryBroadcast(
                c, "positive particle"
            ),
            "Variable with a very long name": a,
            "2D variable": pybamm.FullBroadcast(
                1, "negative particle", {"secondary": "negative electrode"}
            ),
            "NaN variable": pybamm.Scalar(np.nan),
        }

        # ODEs only (don't use Jacobian)
        model.use_jacobian = False

        # Process and solve
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        solver = model.default_solver
        t_eval = np.linspace(0, 2, 100)
        solution = solver.solve(model, t_eval)
        quick_plot = pybamm.QuickPlot(
            solution,
            [
                "a",
                "b broadcasted",
                "c broadcasted",
                "b broadcasted negative electrode",
                "c broadcasted positive electrode",
            ],
        )
        quick_plot.plot(0)

        # update the axis
        new_axis = [0, 0.5, 0, 1]
        quick_plot.axis_limits.update({("a",): new_axis})
        self.assertEqual(quick_plot.axis_limits[("a",)], new_axis)

        # and now reset them
        quick_plot.reset_axis()
        self.assertNotEqual(quick_plot.axis_limits[("a",)], new_axis)

        # check dynamic plot loads
        quick_plot.dynamic_plot(testing=True)

        quick_plot.slider_update(0.01)

        # Test with different output variables
        quick_plot = pybamm.QuickPlot(solution, ["b broadcasted"])
        self.assertEqual(len(quick_plot.axis_limits), 1)
        quick_plot.plot(0)

        quick_plot = pybamm.QuickPlot(
            solution,
            [
                ["a", "a"],
                ["b broadcasted", "b broadcasted"],
                "c broadcasted",
                "b broadcasted negative electrode",
                "c broadcasted positive electrode",
            ],
        )
        self.assertEqual(len(quick_plot.axis_limits), 5)
        quick_plot.plot(0)

        # update the axis
        new_axis = [0, 0.5, 0, 1]
        var_key = ("c broadcasted",)
        quick_plot.axis_limits.update({var_key: new_axis})
        self.assertEqual(quick_plot.axis_limits[var_key], new_axis)

        # and now reset them
        quick_plot.reset_axis()
        self.assertNotEqual(quick_plot.axis_limits[var_key], new_axis)

        # check dynamic plot loads
        quick_plot.dynamic_plot(testing=True)

        quick_plot.slider_update(0.01)

        # Test longer name
        quick_plot = pybamm.QuickPlot(solution, ["Variable with a very long name"])
        quick_plot.plot(0)

        # Test different inputs
        quick_plot = pybamm.QuickPlot(
            [solution, solution],
            ["a"],
            colors=["r", "g", "b"],
            linestyles=["-", "--"],
            figsize=(1, 2),
            labels=["sol 1", "sol 2"],
            n_rows=2,
        )
        self.assertEqual(quick_plot.colors, ["r", "g", "b"])
        self.assertEqual(quick_plot.linestyles, ["-", "--"])
        self.assertEqual(quick_plot.figsize, (1, 2))
        self.assertEqual(quick_plot.labels, ["sol 1", "sol 2"])
        self.assertEqual(quick_plot.n_rows, 2)
        self.assertEqual(quick_plot.n_cols, 1)

        # Test different time units
        quick_plot = pybamm.QuickPlot(solution, ["a"])
        self.assertEqual(quick_plot.time_scaling_factor, 1)
        quick_plot = pybamm.QuickPlot(solution, ["a"], time_unit="seconds")
        quick_plot.plot(0)
        self.assertEqual(quick_plot.time_scaling_factor, 1)
        np.testing.assert_array_almost_equal(
            quick_plot.plots[("a",)][0][0].get_xdata(), t_eval
        )
        np.testing.assert_array_almost_equal(
            quick_plot.plots[("a",)][0][0].get_ydata(), 0.2 * t_eval
        )
        quick_plot = pybamm.QuickPlot(solution, ["a"], time_unit="minutes")
        quick_plot.plot(0)
        self.assertEqual(quick_plot.time_scaling_factor, 60)
        np.testing.assert_array_almost_equal(
            quick_plot.plots[("a",)][0][0].get_xdata(), t_eval / 60
        )
        np.testing.assert_array_almost_equal(
            quick_plot.plots[("a",)][0][0].get_ydata(), 0.2 * t_eval
        )
        quick_plot = pybamm.QuickPlot(solution, ["a"], time_unit="hours")
        quick_plot.plot(0)
        self.assertEqual(quick_plot.time_scaling_factor, 3600)
        np.testing.assert_array_almost_equal(
            quick_plot.plots[("a",)][0][0].get_xdata(), t_eval / 3600
        )
        np.testing.assert_array_almost_equal(
            quick_plot.plots[("a",)][0][0].get_ydata(), 0.2 * t_eval
        )
        with self.assertRaisesRegex(ValueError, "time unit"):
            pybamm.QuickPlot(solution, ["a"], time_unit="bad unit")
        # long solution defaults to hours instead of seconds
        solution_long = solver.solve(model, np.linspace(0, 1e5))
        quick_plot = pybamm.QuickPlot(solution_long, ["a"])
        self.assertEqual(quick_plot.time_scaling_factor, 3600)

        # Test different spatial units
        quick_plot = pybamm.QuickPlot(solution, ["a"])
        self.assertEqual(quick_plot.spatial_unit, "$\mu$m")
        quick_plot = pybamm.QuickPlot(solution, ["a"], spatial_unit="m")
        self.assertEqual(quick_plot.spatial_unit, "m")
        quick_plot = pybamm.QuickPlot(solution, ["a"], spatial_unit="mm")
        self.assertEqual(quick_plot.spatial_unit, "mm")
        quick_plot = pybamm.QuickPlot(solution, ["a"], spatial_unit="um")
        self.assertEqual(quick_plot.spatial_unit, "$\mu$m")
        with self.assertRaisesRegex(ValueError, "spatial unit"):
            pybamm.QuickPlot(solution, ["a"], spatial_unit="bad unit")

        # Test 2D variables
        quick_plot = pybamm.QuickPlot(solution, ["2D variable"])
        quick_plot.plot(0)
        quick_plot.dynamic_plot(testing=True)
        quick_plot.slider_update(0.01)

        with self.assertRaisesRegex(NotImplementedError, "Cannot plot 2D variables"):
            pybamm.QuickPlot([solution, solution], ["2D variable"])

        # Test different variable limits
        quick_plot = pybamm.QuickPlot(
            solution, ["a", ["c broadcasted", "c broadcasted"]], variable_limits="tight"
        )
        self.assertEqual(quick_plot.axis_limits[("a",)][2:], [None, None])
        self.assertEqual(
            quick_plot.axis_limits[("c broadcasted", "c broadcasted")][2:], [None, None]
        )
        quick_plot.plot(0)
        quick_plot.slider_update(1)

        quick_plot = pybamm.QuickPlot(
            solution, ["2D variable"], variable_limits="tight"
        )
        self.assertEqual(quick_plot.variable_limits[("2D variable",)], (None, None))
        quick_plot.plot(0)
        quick_plot.slider_update(1)

        quick_plot = pybamm.QuickPlot(
            solution,
            ["a", ["c broadcasted", "c broadcasted"]],
            variable_limits={"a": [1, 2], ("c broadcasted", "c broadcasted"): [3, 4]},
        )
        self.assertEqual(quick_plot.axis_limits[("a",)][2:], [1, 2])
        self.assertEqual(
            quick_plot.axis_limits[("c broadcasted", "c broadcasted")][2:], [3, 4]
        )
        quick_plot.plot(0)
        quick_plot.slider_update(1)

        quick_plot = pybamm.QuickPlot(
            solution, ["a", "b broadcasted"], variable_limits={"a": "tight"}
        )
        self.assertEqual(quick_plot.axis_limits[("a",)][2:], [None, None])
        self.assertNotEqual(
            quick_plot.axis_limits[("b broadcasted",)][2:], [None, None]
        )
        quick_plot.plot(0)
        quick_plot.slider_update(1)

        with self.assertRaisesRegex(
            TypeError, "variable_limits must be 'fixed', 'tight', or a dict"
        ):
            pybamm.QuickPlot(
                solution, ["a", "b broadcasted"], variable_limits="bad variable limits"
            )

        # Test errors
        with self.assertRaisesRegex(ValueError, "Mismatching variable domains"):
            pybamm.QuickPlot(solution, [["a", "b broadcasted"]])
        with self.assertRaisesRegex(ValueError, "labels"):
            pybamm.QuickPlot(
                [solution, solution], ["a"], labels=["sol 1", "sol 2", "sol 3"]
            )

        # No variable can be NaN
        with self.assertRaisesRegex(
            ValueError, "All-NaN variable 'NaN variable' provided"
        ):
            pybamm.QuickPlot(solution, ["NaN variable"])

        pybamm.close_plots()

    def test_plot_with_different_models(self):
        model = pybamm.BaseModel()
        a = pybamm.Variable("a")
        model.rhs = {a: pybamm.Scalar(0)}
        model.initial_conditions = {a: pybamm.Scalar(0)}
        solution = pybamm.CasadiSolver("fast").solve(model, [0, 1])
        with self.assertRaisesRegex(ValueError, "No default output variables"):
            pybamm.QuickPlot(solution)

    def test_spm_simulation(self):
        # SPM
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        t_eval = np.linspace(0, 10, 2)
        sim.solve(t_eval)

        # pass only a simulation object
        # it should be converted to a list of corresponding solution
        quick_plot = pybamm.QuickPlot(sim)
        quick_plot.plot(0)

        # mixed simulation and solution input
        # solution should be extracted from the simulation
        quick_plot = pybamm.QuickPlot([sim, sim.solution])
        quick_plot.plot(0)

        # test creating a GIF
        quick_plot.create_gif(number_of_images=3, duration=3)
        assert not os.path.exists("plot*.png")
        assert os.path.exists("plot.gif")
        os.remove("plot.gif")

        pybamm.close_plots()

    def test_loqs_spme(self):
        t_eval = np.linspace(0, 10, 2)

        for model in [
            pybamm.lithium_ion.SPMe(),
            pybamm.lead_acid.LOQS(),
            pybamm.lithium_ion.SPMe({"working electrode": "positive"}),
        ]:
            geometry = model.default_geometry
            param = model.default_parameter_values
            param.process_model(model)
            param.process_geometry(geometry)
            var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5}
            mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            disc.process_model(model)
            solver = model.default_solver
            solution = solver.solve(model, t_eval)
            pybamm.QuickPlot(solution)

            # check 1D (space) variables update properly for different time units
            t = solution["Time [s]"].entries
            c_e_var = solution["Electrolyte concentration [mol.m-3]"]
            # 1D variables should be evaluated on edges
            c_e = c_e_var(t=t, x=mesh[c_e_var.domain].edges)

            for unit, scale in zip(["seconds", "minutes", "hours"], [1, 60, 3600]):
                quick_plot = pybamm.QuickPlot(
                    solution, ["Electrolyte concentration [mol.m-3]"], time_unit=unit
                )
                quick_plot.plot(0)

                qp_data = (
                    quick_plot.plots[("Electrolyte concentration [mol.m-3]",)][0][
                        0
                    ].get_ydata(),
                )[0]
                np.testing.assert_array_almost_equal(qp_data, c_e[:, 0])
                quick_plot.slider_update(t_eval[-1] / scale)

                qp_data = (
                    quick_plot.plots[("Electrolyte concentration [mol.m-3]",)][0][
                        0
                    ].get_ydata(),
                )[0][:, 0]
                np.testing.assert_array_almost_equal(qp_data, c_e[:, 1])

            # test quick plot of particle for spme
            if (
                model.name == "Single Particle Model with electrolyte"
                and model.options["working electrode"] != "positive"
            ):
                output_variables = [
                    "X-averaged negative particle concentration [mol.m-3]",
                    "X-averaged positive particle concentration [mol.m-3]",
                    "Negative particle concentration [mol.m-3]",
                    "Positive particle concentration [mol.m-3]",
                ]
                pybamm.QuickPlot(solution, output_variables)

                # check 2D (space) variables update properly for different time units
                c_n = solution["Negative particle concentration [mol.m-3]"]

                for unit, scale in zip(["seconds", "minutes", "hours"], [1, 60, 3600]):
                    quick_plot = pybamm.QuickPlot(
                        solution,
                        ["Negative particle concentration [mol.m-3]"],
                        time_unit=unit,
                    )
                    quick_plot.plot(0)
                    qp_data = quick_plot.plots[
                        ("Negative particle concentration [mol.m-3]",)
                    ][0][1]
                    c_n_eval = c_n(t_eval[0], r=c_n.first_dim_pts, x=c_n.second_dim_pts)
                    np.testing.assert_array_almost_equal(qp_data, c_n_eval)
                    quick_plot.slider_update(t_eval[-1] / scale)
                    qp_data = quick_plot.plots[
                        ("Negative particle concentration [mol.m-3]",)
                    ][0][1]
                    c_n_eval = c_n(
                        t_eval[-1], r=c_n.first_dim_pts, x=c_n.second_dim_pts
                    )
                    np.testing.assert_array_almost_equal(qp_data, c_n_eval)

        pybamm.close_plots()

    def test_plot_1plus1D_spme(self):
        spm = pybamm.lithium_ion.SPMe(
            {"current collector": "potential pair", "dimensionality": 1}
        )
        geometry = spm.default_geometry
        param = spm.default_parameter_values
        param.process_model(spm)
        param.process_geometry(geometry)
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5, "z": 5}
        mesh = pybamm.Mesh(geometry, spm.default_submesh_types, var_pts)
        disc_spm = pybamm.Discretisation(mesh, spm.default_spatial_methods)
        disc_spm.process_model(spm)
        t_eval = np.linspace(0, 100, 10)
        solution = spm.default_solver.solve(spm, t_eval)

        # check 2D (x,z space) variables update properly for different time units
        # Note: these should be the transpose of the entries in the processed variable
        c_e = solution["Electrolyte concentration [mol.m-3]"]

        for unit, scale in zip(["seconds", "minutes", "hours"], [1, 60, 3600]):
            quick_plot = pybamm.QuickPlot(
                solution, ["Electrolyte concentration [mol.m-3]"], time_unit=unit
            )
            quick_plot.plot(0)
            qp_data = quick_plot.plots[("Electrolyte concentration [mol.m-3]",)][0][1]
            c_e_eval = c_e(t_eval[0], x=c_e.first_dim_pts, z=c_e.second_dim_pts)
            np.testing.assert_array_almost_equal(qp_data.T, c_e_eval)
            quick_plot.slider_update(t_eval[-1] / scale)
            qp_data = quick_plot.plots[("Electrolyte concentration [mol.m-3]",)][0][1]
            c_e_eval = c_e(t_eval[-1], x=c_e.first_dim_pts, z=c_e.second_dim_pts)
            np.testing.assert_array_almost_equal(qp_data.T, c_e_eval)

        pybamm.close_plots()

    def test_plot_2plus1D_spm(self):
        spm = pybamm.lithium_ion.SPM(
            {"current collector": "potential pair", "dimensionality": 2}
        )
        geometry = spm.default_geometry
        param = spm.default_parameter_values
        param.process_model(spm)
        param.process_geometry(geometry)
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5, "y": 5, "z": 5}
        mesh = pybamm.Mesh(geometry, spm.default_submesh_types, var_pts)
        disc_spm = pybamm.Discretisation(mesh, spm.default_spatial_methods)
        disc_spm.process_model(spm)
        t_eval = np.linspace(0, 100, 10)
        solution = spm.default_solver.solve(spm, t_eval)

        quick_plot = pybamm.QuickPlot(
            solution,
            [
                "Negative current collector potential [V]",
                "Positive current collector potential [V]",
                "Voltage [V]",
            ],
        )
        quick_plot.dynamic_plot(testing=True)
        quick_plot.slider_update(1)

        # check 2D (y,z space) variables update properly for different time units
        # Note: these should be the transpose of the entries in the processed variable
        phi_n = solution["Negative current collector potential [V]"].entries

        for unit, scale in zip(["seconds", "minutes", "hours"], [1, 60, 3600]):
            quick_plot = pybamm.QuickPlot(
                solution, ["Negative current collector potential [V]"], time_unit=unit
            )
            quick_plot.plot(0)
            qp_data = quick_plot.plots[("Negative current collector potential [V]",)][
                0
            ][1]
            np.testing.assert_array_almost_equal(qp_data.T, phi_n[:, :, 0])
            quick_plot.slider_update(t_eval[-1] / scale)
            qp_data = quick_plot.plots[("Negative current collector potential [V]",)][
                0
            ][1]
            np.testing.assert_array_almost_equal(qp_data.T, phi_n[:, :, -1])

        with self.assertRaisesRegex(NotImplementedError, "Shape not recognized for"):
            pybamm.QuickPlot(solution, ["Negative particle concentration [mol.m-3]"])

        pybamm.close_plots()

    def test_failure(self):
        with self.assertRaisesRegex(TypeError, "solutions must be"):
            pybamm.QuickPlot(1)

    def test_model_with_inputs(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        model = pybamm.lithium_ion.SPMe()
        parameter_values.update({"Electrode height [m]": "[input]"})
        solver1 = pybamm.CasadiSolver(mode="safe")
        sim1 = pybamm.Simulation(
            model, parameter_values=parameter_values, solver=solver1
        )
        inputs1 = {"Electrode height [m]": 1.00}
        sol1 = sim1.solve(t_eval=np.linspace(0, 1000, 101), inputs=inputs1)
        solver2 = pybamm.CasadiSolver(mode="safe")
        sim2 = pybamm.Simulation(
            model, parameter_values=parameter_values, solver=solver2
        )
        inputs2 = {"Electrode height [m]": 2.00}
        sol2 = sim2.solve(t_eval=np.linspace(0, 1000, 101), inputs=inputs2)
        output_variables = [
            "Voltage [V]",
            "Current [A]",
            "Negative electrode potential [V]",
            "Positive electrode potential [V]",
            "Electrolyte potential [V]",
            "Electrolyte concentration [Molar]",
            "Negative particle surface concentration",
            "Positive particle surface concentration",
        ]
        quick_plot = pybamm.QuickPlot(
            solutions=[sol1, sol2], output_variables=output_variables
        )
        quick_plot.dynamic_plot(testing=True)
        quick_plot.slider_update(1)
        pybamm.close_plots()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
