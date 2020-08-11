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
        # Make sure model is unchanged
        self.assertNotEqual(sim.model, model)
        for val in list(model.rhs.values()):
            self.assertTrue(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertFalse(val.has_symbol_of_classes(pybamm.Matrix))

        sim.build()
        self.assertFalse(sim._mesh is None)
        self.assertFalse(sim._disc is None)
        for val in list(sim.built_model.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))
            # skip test for scalar variables (e.g. discharge capacity)
            if val.size > 1:
                self.assertTrue(val.has_symbol_of_classes(pybamm.Matrix))

    def test_specs_deprecated(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        with self.assertRaisesRegex(NotImplementedError, "specs"):
            sim.specs()

    def test_solve(self):

        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.solve([0, 600])
        self.assertFalse(sim._solution is None)
        for val in list(sim.built_model.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))
            # skip test for scalar variables (e.g. discharge capacity)
            if val.size > 1:
                self.assertTrue(val.has_symbol_of_classes(pybamm.Matrix))

        # test solve without check
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.solve(t_eval=[0, 600], check_model=False)
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

        sim.solve(np.linspace(0, 1, 100))
        np.testing.assert_array_equal(sim.solution.t, np.linspace(0, 1, 100))
        np.testing.assert_array_almost_equal(
            sim.solution["v"].entries, np.exp(-np.linspace(0, 1, 100))
        )

    def test_solve_already_partially_processed_model(self):

        model = pybamm.lithium_ion.SPM()

        # Process model manually
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        # Let simulation take over
        sim = pybamm.Simulation(model)
        sim.solve([0, 600])

        # Discretised manually
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        # Let simulation take over
        sim = pybamm.Simulation(model)
        sim.solve([0, 600])

    def test_reuse_commands(self):

        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())

        sim.set_parameters()
        sim.set_parameters()

        sim.build()
        sim.build()

        sim.solve([0, 600])
        sim.solve([0, 600])

        sim.build()
        sim.solve([0, 600])
        sim.set_parameters()

    def test_set_crate(self):
        model = pybamm.lithium_ion.SPM()
        current_1C = model.default_parameter_values["Current function [A]"]
        sim = pybamm.Simulation(model, C_rate=2)
        self.assertEqual(sim.parameter_values["Current function [A]"], 2 * current_1C)
        self.assertEqual(sim.C_rate, 2)

    def test_get_variable_array(self):

        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.solve([0, 600])

        phi_s_n = sim.get_variable_array("Negative electrode potential")

        self.assertIsInstance(phi_s_n, np.ndarray)

        c_s_n_surf, c_e = sim.get_variable_array(
            "Negative particle surface concentration", "Electrolyte concentration"
        )

        self.assertIsInstance(c_s_n_surf, np.ndarray)
        self.assertIsInstance(c_e, np.ndarray)

    def test_set_external_variable(self):
        model_options = {
            "thermal": "lumped",
            "external submodels": ["thermal", "negative particle"],
        }
        model = pybamm.lithium_ion.SPMe(model_options)
        sim = pybamm.Simulation(model)

        var = pybamm.standard_spatial_vars
        Nr = model.default_var_pts[var.r_n]

        T_av = 0
        c_s_n_av = np.ones((Nr, 1)) * 0.5
        external_variables = {
            "Volume-averaged cell temperature": T_av,
            "X-averaged negative particle concentration": c_s_n_av,
        }

        # Step
        dt = 0.1
        for _ in range(5):
            sim.step(dt, external_variables=external_variables)
        sim.plot(testing=True)

        # Solve
        t_eval = np.linspace(0, 3600)
        sim.solve(t_eval, external_variables=external_variables)
        sim.plot(testing=True)

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
        sim.solve(t_eval=[0, 600], inputs={"Current function [A]": 1})
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
            sim.solution.inputs["Current function [A]"], np.array([[1, 1, 2]])
        )

    def test_save_load(self):
        model = pybamm.lead_acid.LOQS()
        model.use_jacobian = True
        sim = pybamm.Simulation(model)

        sim.save("test.pickle")
        sim_load = pybamm.load_sim("test.pickle")
        self.assertEqual(sim.model.name, sim_load.model.name)

        # save after solving
        sim.solve([0, 600])
        sim.save("test.pickle")
        sim_load = pybamm.load_sim("test.pickle")
        self.assertEqual(sim.model.name, sim_load.model.name)

        # with python formats
        model.convert_to_format = None
        sim = pybamm.Simulation(model)
        sim.solve([0, 600])
        sim.save("test.pickle")
        model.convert_to_format = "python"
        sim = pybamm.Simulation(model)
        sim.solve([0, 600])
        with self.assertRaisesRegex(
            NotImplementedError, "Cannot save simulation if model format is python"
        ):
            sim.save("test.pickle")

    def test_save_load_dae(self):
        model = pybamm.lead_acid.LOQS({"surface form": "algebraic"})
        model.use_jacobian = True
        sim = pybamm.Simulation(model)

        # save after solving
        sim.solve([0, 600])
        sim.save("test.pickle")
        sim_load = pybamm.load_sim("test.pickle")
        self.assertEqual(sim.model.name, sim_load.model.name)

        # with python format
        model.convert_to_format = None
        sim = pybamm.Simulation(model)
        sim.solve([0, 600])
        sim.save("test.pickle")

        # with Casadi solver
        model.convert_to_format = "casadi"
        sim = pybamm.Simulation(model, solver=pybamm.CasadiSolver())
        sim.solve([0, 600])
        sim.save("test.pickle")
        sim_load = pybamm.load_sim("test.pickle")
        self.assertEqual(sim.model.name, sim_load.model.name)

    def test_plot(self):
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())

        # test exception if not solved
        with self.assertRaises(ValueError):
            sim.plot()

        # now solve and plot
        t_eval = np.linspace(0, 100, 5)
        sim.solve(t_eval=t_eval)
        sim.plot(testing=True)

        # test quick_plot_vars deprecation error
        with self.assertRaisesRegex(NotImplementedError, "'quick_plot_vars'"):
            sim.plot(quick_plot_vars=["var"])

    def test_drive_cycle_data(self):
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        param["Current function [A]"] = "[current data]US06"

        with self.assertRaisesRegex(NotImplementedError, "Drive cycle from data"):
            pybamm.Simulation(model, parameter_values=param)

    def test_drive_cycle_interpolant(self):
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        # Import drive cycle from file
        drive_cycle = pd.read_csv(
            pybamm.get_parameters_filepath(
                os.path.join("input", "drive_cycles", "US06.csv")
            ),
            comment="#",
            skip_blank_lines=True,
            header=None,
        )

        timescale = param.evaluate(model.timescale)
        current_interpolant = pybamm.Interpolant(
            drive_cycle.to_numpy(), timescale * pybamm.t
        )

        param["Current function [A]"] = current_interpolant

        time_data = drive_cycle.values[:, 0]

        sim = pybamm.Simulation(model, parameter_values=param)

        # check solution is returned at the times in the data
        sim.solve()
        tau = sim.model.timescale.evaluate()
        np.testing.assert_array_almost_equal(sim.solution.t, time_data / tau)

        # check warning raised if the largest gap in t_eval is bigger than the
        # smallest gap in the data
        with self.assertWarns(pybamm.SolverWarning):
            sim.solve(t_eval=np.linspace(0, 1, 100))

        # check warning raised if t_eval doesnt contain time_data , but has a finer
        # resolution (can still solve, but good for users to know they dont have
        # the solution returned at the data points)
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
        sim.solve([0, 3600])
        current = sim.solution["Current [A]"]
        self.assertEqual(current(0), 1)
        self.assertEqual(current(1500), -0.5)
        self.assertEqual(current(3000), 0.5)

    def test_t_eval(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        # test no t_eval
        with self.assertRaisesRegex(pybamm.SolverError, "'t_eval' must be provided"):
            sim.solve()

        # test t_eval list of length != 2
        with self.assertRaisesRegex(pybamm.SolverError, "'t_eval' can be provided"):
            sim.solve(t_eval=[0, 1, 2])

        # tets list gets turned into np.linspace(t0, tf, 100)
        sim.solve(t_eval=[0, 10])
        np.testing.assert_array_almost_equal(sim.t_eval, np.linspace(0, 10, 100))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
