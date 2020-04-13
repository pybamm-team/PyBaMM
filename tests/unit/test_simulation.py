import pybamm
import numpy as np
import pandas as pd
import os
import unittest


class TestSimulation(unittest.TestCase):
    def test_basic_ops(self):

        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        self.assertEqual(model.__class__, sim._model_class)
        self.assertEqual(model.options, sim._model_options)

        # check that the model is unprocessed
        self.assertEqual(sim._mesh, None)
        self.assertEqual(sim._disc, None)
        for val in list(sim.model.rhs.values()):
            self.assertTrue(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertFalse(val.has_symbol_of_classes(pybamm.Matrix))

        sim.set_parameters()
        self.assertEqual(sim._mesh, None)
        self.assertEqual(sim._disc, None)
        for val in list(sim.model_with_set_params.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertFalse(val.has_symbol_of_classes(pybamm.Matrix))

        sim.build()
        self.assertFalse(sim._mesh is None)
        self.assertFalse(sim._disc is None)
        for val in list(sim.built_model.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))
            # skip test for scalar variables (e.g. discharge capacity)
            if val.size > 1:
                self.assertTrue(val.has_symbol_of_classes(pybamm.Matrix))

        sim.reset()
        sim.set_parameters()
        self.assertEqual(sim._mesh, None)
        self.assertEqual(sim._disc, None)
        self.assertEqual(sim.built_model, None)

        for val in list(sim.model_with_set_params.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertFalse(val.has_symbol_of_classes(pybamm.Matrix))

        sim.build()
        sim.reset()
        self.assertEqual(sim._mesh, None)
        self.assertEqual(sim._disc, None)
        self.assertEqual(sim.model_with_set_params, None)
        self.assertEqual(sim.built_model, None)
        for val in list(sim.model.rhs.values()):
            self.assertTrue(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertFalse(val.has_symbol_of_classes(pybamm.Matrix))

    def test_solve(self):

        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.solve()
        self.assertFalse(sim._solution is None)
        for val in list(sim.built_model.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))
            # skip test for scalar variables (e.g. discharge capacity)
            if val.size > 1:
                self.assertTrue(val.has_symbol_of_classes(pybamm.Matrix))

        sim.reset()
        self.assertEqual(sim.model_with_set_params, None)
        self.assertEqual(sim.built_model, None)
        for val in list(sim.model.rhs.values()):
            self.assertTrue(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertFalse(val.has_symbol_of_classes(pybamm.Matrix))

        self.assertEqual(sim._solution, None)

        # test solve without check
        sim.reset()
        sim.solve(check_model=False)
        for val in list(sim.built_model.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))
            # skip test for scalar variables (e.g. discharge capacity)
            if val.size > 1:
                self.assertTrue(val.has_symbol_of_classes(pybamm.Matrix))

    def test_solve_non_battery_model(self):

        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -v}
        model.initial_conditions = {v: 1}
        model.variables = {"v": v}
        sim = pybamm.Simulation(
            model, solver=pybamm.ScipySolver(rtol=1e-10, atol=1e-10)
        )

        sim.solve()
        np.testing.assert_array_equal(sim.solution.t, np.linspace(0, 1, 100))
        np.testing.assert_array_almost_equal(
            sim.solution["v"].entries, np.exp(-np.linspace(0, 1, 100))
        )

    def test_reuse_commands(self):

        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())

        sim.set_parameters()
        sim.set_parameters()

        sim.build()
        sim.build()

        sim.solve()
        sim.solve()

        sim.build()
        sim.solve()
        sim.set_parameters()

    def test_specs(self):
        # test can rebuild after setting specs
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.build()

        model_options = {"thermal": "lumped"}
        sim.specs(model_options=model_options)
        sim.build()
        self.assertEqual(sim.model.options["thermal"], "lumped")

        params = sim.parameter_values
        # normally is 0.0001
        params.update({"Negative electrode thickness [m]": 0.0002})
        sim.specs(parameter_values=params)

        self.assertEqual(
            sim.parameter_values["Negative electrode thickness [m]"], 0.0002
        )
        sim.build()

        geometry = sim.unprocessed_geometry
        custom_geometry = {}
        x_n = pybamm.standard_spatial_vars.x_n
        custom_geometry["negative electrode"] = {
            "primary": {
                x_n: {"min": pybamm.Scalar(0), "max": pybamm.geometric_parameters.l_n}
            }
        }
        geometry.update(custom_geometry)
        sim.specs(geometry=geometry)
        sim.build()

        var_pts = sim.var_pts
        var_pts[pybamm.standard_spatial_vars.x_n] = 5
        sim.specs(var_pts=var_pts)
        sim.build()

        spatial_methods = sim.spatial_methods
        # nothing to change this to at the moment but just reload in
        sim.specs(spatial_methods=spatial_methods)
        sim.build()

    def test_set_crate(self):
        model = pybamm.lithium_ion.SPM()
        current_1C = model.default_parameter_values["Current function [A]"]
        sim = pybamm.Simulation(model, C_rate=2)
        self.assertEqual(sim.parameter_values["Current function [A]"], 2 * current_1C)
        self.assertEqual(sim.C_rate, 2)
        sim.specs(C_rate=3)
        self.assertEqual(sim.parameter_values["Current function [A]"], 3 * current_1C)
        self.assertEqual(sim.C_rate, 3)

    def test_set_defaults(self):
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())

        model_options = {"thermal": "x-full"}
        submesh_types = {
            "Negative particle": pybamm.MeshGenerator(pybamm.Exponential1DSubMesh)
        }
        solver = pybamm.BaseSolver()
        quick_plot_vars = ["Negative particle surface concentration"]
        sim.specs(
            model_options=model_options,
            submesh_types=submesh_types,
            solver=solver,
            quick_plot_vars=quick_plot_vars,
        )

        sim.set_defaults()

        self.assertEqual(sim.model_options["thermal"], "x-full")
        self.assertEqual(
            sim.submesh_types["negative particle"].submesh_type, pybamm.Uniform1DSubMesh
        )
        self.assertEqual(sim.quick_plot_vars, None)
        self.assertIsInstance(sim.solver, pybamm.ScipySolver)

    def test_get_variable_array(self):

        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.solve()

        phi_s_n = sim.get_variable_array("Negative electrode potential")

        self.assertIsInstance(phi_s_n, np.ndarray)

        c_s_n_surf, c_e = sim.get_variable_array(
            "Negative particle surface concentration", "Electrolyte concentration"
        )

        self.assertIsInstance(c_s_n_surf, np.ndarray)
        self.assertIsInstance(c_e, np.ndarray)

    def test_set_external_variable(self):
        model_options = {"thermal": "lumped", "external submodels": ["thermal"]}
        model = pybamm.lithium_ion.SPMe(model_options)
        sim = pybamm.Simulation(model)

        T_av = 0

        dt = 0.001

        external_variables = {"Volume-averaged cell temperature": T_av}
        sim.step(dt, external_variables=external_variables)

    def test_step(self):

        dt = 0.001
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sim.step(dt)  # 1 step stores first two points
        tau = sim.model.timescale.evaluate()
        self.assertEqual(sim.solution.t.size, 2)
        self.assertEqual(sim.solution.y[0, :].size, 2)
        self.assertEqual(sim.solution.t[0], 0)
        self.assertEqual(sim.solution.t[1], dt / tau)
        sim.step(dt)  # automatically append the next step
        self.assertEqual(sim.solution.t.size, 3)
        self.assertEqual(sim.solution.y[0, :].size, 3)
        self.assertEqual(sim.solution.t[0], 0)
        self.assertEqual(sim.solution.t[1], dt / tau)
        self.assertEqual(sim.solution.t[2], 2 * dt / tau)
        sim.step(dt, save=False)  # now only store the two end step points
        self.assertEqual(sim.solution.t.size, 2)
        self.assertEqual(sim.solution.y[0, :].size, 2)
        self.assertEqual(sim.solution.t[0], 2 * dt / tau)
        self.assertEqual(sim.solution.t[1], 3 * dt / tau)

    def test_solve_with_inputs(self):
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        param.update({"Current function [A]": "[input]"})
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.solve(inputs={"Current function [A]": 1})
        np.testing.assert_array_equal(sim.solution.inputs["Current function [A]"], 1)

    def test_step_with_inputs(self):
        dt = 0.001
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        param.update({"Current function [A]": "[input]"})
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.step(
            dt, inputs={"Current function [A]": 1}
        )  # 1 step stores first two points
        tau = sim.model.timescale.evaluate()
        self.assertEqual(sim.solution.t.size, 2)
        self.assertEqual(sim.solution.y[0, :].size, 2)
        self.assertEqual(sim.solution.t[0], 0)
        self.assertEqual(sim.solution.t[1], dt / tau)
        np.testing.assert_array_equal(sim.solution.inputs["Current function [A]"], 1)
        sim.step(
            dt, inputs={"Current function [A]": 2}
        )  # automatically append the next step
        self.assertEqual(sim.solution.t.size, 3)
        self.assertEqual(sim.solution.y[0, :].size, 3)
        self.assertEqual(sim.solution.t[0], 0)
        self.assertEqual(sim.solution.t[1], dt / tau)
        self.assertEqual(sim.solution.t[2], 2 * dt / tau)
        np.testing.assert_array_equal(
            sim.solution.inputs["Current function [A]"], np.array([1, 1, 2])
        )

    def test_save_load(self):
        model = pybamm.lead_acid.LOQS()
        model.use_jacobian = True
        sim = pybamm.Simulation(model)

        sim.save("test.pickle")
        sim_load = pybamm.load_sim("test.pickle")
        self.assertEqual(sim.model.name, sim_load.model.name)

        # save after solving
        sim.solve()
        sim.save("test.pickle")
        sim_load = pybamm.load_sim("test.pickle")
        self.assertEqual(sim.model.name, sim_load.model.name)

        # with python formats
        model.convert_to_format = None
        sim = pybamm.Simulation(model)
        sim.solve()
        sim.save("test.pickle")
        model.convert_to_format = "python"
        sim = pybamm.Simulation(model)
        sim.solve()
        with self.assertRaisesRegex(
            NotImplementedError, "Cannot save simulation if model format is python"
        ):
            sim.save("test.pickle")

    def test_save_load_dae(self):
        model = pybamm.lead_acid.LOQS({"surface form": "algebraic"})
        model.use_jacobian = True
        sim = pybamm.Simulation(model)

        # save after solving
        sim.solve()
        sim.save("test.pickle")
        sim_load = pybamm.load_sim("test.pickle")
        self.assertEqual(sim.model.name, sim_load.model.name)

        # with python format
        model.convert_to_format = None
        sim = pybamm.Simulation(model)
        sim.solve()
        sim.save("test.pickle")

        # with Casadi solver
        model.convert_to_format = "casadi"
        sim = pybamm.Simulation(model, solver=pybamm.CasadiSolver())
        sim.solve()
        sim.save("test.pickle")
        sim_load = pybamm.load_sim("test.pickle")
        self.assertEqual(sim.model.name, sim_load.model.name)

    def test_set_defaults2(self):
        model = pybamm.lithium_ion.SPM()

        # make simulation with silly options (should this be allowed?)
        sim = pybamm.Simulation(
            model,
            geometry={},
            parameter_values={},
            submesh_types={},
            var_pts={},
            spatial_methods={},
            solver={},
            quick_plot_vars=[],
        )

        # reset and check
        sim.set_defaults()
        # Not sure of best way to test nested dicts?
        # self.geometry = model.default_geometry
        self.assertEqual(
            sim._parameter_values._dict_items,
            model.default_parameter_values._dict_items,
        )
        for domain, submesh in model.default_submesh_types.items():
            self.assertEqual(
                sim._submesh_types[domain].submesh_type, submesh.submesh_type
            )
        self.assertEqual(sim._var_pts, model.default_var_pts)
        for domain, method in model.default_spatial_methods.items():
            self.assertIsInstance(sim._spatial_methods[domain], type(method))
        self.assertIsInstance(sim._solver, type(model.default_solver))
        self.assertEqual(sim._quick_plot_vars, None)

    def test_plot(self):
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())

        # test exception if not solved
        with self.assertRaises(ValueError):
            sim.plot()

        # now solve and plot
        t_eval = np.linspace(0, 100, 5)
        sim.solve(t_eval=t_eval)
        sim.plot(testing=True)

    def test_drive_cycle_data(self):
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        param["Current function [A]"] = "[current data]US06"

        drive_cycle = pd.read_csv(
            pybamm.get_parameters_filepath(
                os.path.join("input", "drive_cycles", "US06.csv")
            ),
            comment="#",
            skip_blank_lines=True,
            header=None,
        )
        time_data = drive_cycle.values[:, 0]

        sim = pybamm.Simulation(model, parameter_values=param)

        # check solution is returned at the times in the data
        sim.solve()
        tau = sim.model.timescale.evaluate()
        np.testing.assert_array_almost_equal(sim.solution.t, time_data / tau)

        # check warning raised if the largest gap in t_eval is bigger than the
        # smallest gap in the data
        sim.reset()
        with self.assertWarns(pybamm.SolverWarning):
            sim.solve(t_eval=np.linspace(0, 1, 100))

        # check warning raised if t_eval doesnt contain time_data , but has a finer
        # resolution (can still solve, but good for users to know they dont have
        # the solution returned at the data points)
        sim.reset()
        with self.assertWarns(pybamm.SolverWarning):
            sim.solve(t_eval=np.linspace(0, time_data[-1], 800))

    def test_discontinuous_current(self):
        def car_current(t):
            current = (
                1 * (t >= 0) * (t <= 1000)
                - 0.5 * (1000 < t) * (t <= 2000)
                + 0.5 * (2000 < t)
            )
            return current

        model = pybamm.lithium_ion.DFN()
        param = model.default_parameter_values
        param["Current function [A]"] = car_current

        sim = pybamm.Simulation(
            model, parameter_values=param, solver=pybamm.CasadiSolver(mode="fast")
        )
        sim.solve()
        current = sim.solution["Current [A]"]
        tau = sim.model.timescale.evaluate()
        self.assertEqual(current(0), 1)
        self.assertEqual(current(1500 / tau), -0.5)
        self.assertEqual(current(3000 / tau), 0.5)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
