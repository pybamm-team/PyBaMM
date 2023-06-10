import pybamm
import numpy as np
import pandas as pd
from tests import TestCase
import os
import sys
import unittest
import uuid


class TestSimulation(TestCase):
    def test_simple_model(self):
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -v}
        model.initial_conditions = {v: 1}
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 1])
        np.testing.assert_array_almost_equal(sol.y.full()[0], np.exp(-sol.t), decimal=5)

    def test_basic_ops(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        # check that the model is unprocessed
        self.assertEqual(sim._mesh, None)
        self.assertEqual(sim._disc, None)
        V = sim.model.variables["Voltage [V]"]
        self.assertTrue(V.has_symbol_of_classes(pybamm.Parameter))
        self.assertFalse(V.has_symbol_of_classes(pybamm.Matrix))

        sim.set_parameters()
        self.assertEqual(sim._mesh, None)
        self.assertEqual(sim._disc, None)
        V = sim.model_with_set_params.variables["Voltage [V]"]
        self.assertFalse(V.has_symbol_of_classes(pybamm.Parameter))
        self.assertFalse(V.has_symbol_of_classes(pybamm.Matrix))
        # Make sure model is unchanged
        self.assertNotEqual(sim.model, model)
        V = model.variables["Voltage [V]"]
        self.assertTrue(V.has_symbol_of_classes(pybamm.Parameter))
        self.assertFalse(V.has_symbol_of_classes(pybamm.Matrix))

        self.assertEqual(sim.submesh_types, model.default_submesh_types)
        self.assertEqual(sim.var_pts, model.default_var_pts)
        self.assertIsNone(sim.mesh)
        for key in sim.spatial_methods.keys():
            self.assertEqual(
                sim.spatial_methods[key].__class__,
                model.default_spatial_methods[key].__class__,
            )

        sim.build()
        self.assertFalse(sim._mesh is None)
        self.assertFalse(sim._disc is None)
        V = sim.built_model.variables["Voltage [V]"]
        self.assertFalse(V.has_symbol_of_classes(pybamm.Parameter))
        self.assertTrue(V.has_symbol_of_classes(pybamm.Matrix))

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
        sol = sim.solve(t_eval=[0, 600], check_model=False)
        for val in list(sim.built_model.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))
            # skip test for scalar variables (e.g. discharge capacity)
            if val.size > 1:
                self.assertTrue(val.has_symbol_of_classes(pybamm.Matrix))

        # Test options that are only available when simulating an experiment
        with self.assertRaisesRegex(ValueError, "save_at_cycles"):
            sim.solve(save_at_cycles=2)
        with self.assertRaisesRegex(ValueError, "starting_solution"):
            sim.solve(starting_solution=sol)

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

    def test_step(self):
        dt = 0.001
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        sim.step(dt)  # 1 step stores first two points
        self.assertEqual(sim.solution.y.full()[0, :].size, 2)
        np.testing.assert_array_almost_equal(sim.solution.t, np.array([0, dt]))
        saved_sol = sim.solution

        sim.step(dt)  # automatically append the next step
        self.assertEqual(sim.solution.y.full()[0, :].size, 4)
        np.testing.assert_array_almost_equal(
            sim.solution.t, np.array([0, dt, dt + 1e-9, 2 * dt])
        )

        sim.step(dt, save=False)  # now only store the two end step points
        self.assertEqual(sim.solution.y.full()[0, :].size, 2)
        np.testing.assert_array_almost_equal(
            sim.solution.t, np.array([2 * dt + 1e-9, 3 * dt])
        )
        # Start from saved solution
        sim.step(dt, starting_solution=saved_sol)
        self.assertEqual(sim.solution.y.full()[0, :].size, 4)
        np.testing.assert_array_almost_equal(
            sim.solution.t, np.array([0, dt, dt + 1e-9, 2 * dt])
        )

    def test_solve_with_initial_soc(self):
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.solve(t_eval=[0, 600], initial_soc=1)
        self.assertEqual(sim._built_initial_soc, 1)
        sim.solve(t_eval=[0, 600], initial_soc=0.5)
        self.assertEqual(sim._built_initial_soc, 0.5)
        exp = pybamm.Experiment(
            [pybamm.step.string("Discharge at 1C until 3.6V", period="1 minute")]
        )
        sim = pybamm.Simulation(model, parameter_values=param, experiment=exp)
        sim.solve(initial_soc=0.8)
        self.assertEqual(sim._built_initial_soc, 0.8)

        # test with drive cycle
        drive_cycle = pd.read_csv(
            os.path.join("pybamm", "input", "drive_cycles", "US06.csv"),
            comment="#",
            header=None,
        ).to_numpy()
        current_interpolant = pybamm.Interpolant(
            drive_cycle[:, 0], drive_cycle[:, 1], pybamm.t
        )
        param["Current function [A]"] = current_interpolant
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.solve(initial_soc=0.8)
        self.assertEqual(sim._built_initial_soc, 0.8)

        # Test that build works with initial_soc
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.build(initial_soc=0.5)
        self.assertEqual(sim._built_initial_soc, 0.5)

    def test_solve_with_inputs(self):
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        param.update({"Current function [A]": "[input]"})
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.solve(t_eval=[0, 600], inputs={"Current function [A]": 1})
        np.testing.assert_array_equal(
            sim.solution.all_inputs[0]["Current function [A]"], 1
        )

    def test_step_with_inputs(self):
        dt = 0.001
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        param.update({"Current function [A]": "[input]"})
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.step(
            dt, inputs={"Current function [A]": 1}
        )  # 1 step stores first two points
        self.assertEqual(sim.solution.t.size, 2)
        self.assertEqual(sim.solution.y.full()[0, :].size, 2)
        self.assertEqual(sim.solution.t[0], 0)
        self.assertEqual(sim.solution.t[1], dt)
        np.testing.assert_array_equal(
            sim.solution.all_inputs[0]["Current function [A]"], 1
        )
        sim.step(
            dt, inputs={"Current function [A]": 2}
        )  # automatically append the next step
        self.assertEqual(sim.solution.y.full()[0, :].size, 4)
        np.testing.assert_array_almost_equal(
            sim.solution.t, np.array([0, dt, dt + 1e-9, 2 * dt])
        )
        np.testing.assert_array_equal(
            sim.solution.all_inputs[1]["Current function [A]"], 2
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

    def test_load_param(self):
        # Test load_sim for parameters imports
        filename = f"{uuid.uuid4()}.p"
        model = pybamm.lithium_ion.SPM()
        params = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, parameter_values=params)
        sim.solve([0, 3600])
        sim.save(filename)

        try:
            pkl_obj = pybamm.load_sim(os.path.join(filename))
        except Exception as excep:
            os.remove(filename)
            raise excep

        self.assertEqual(
            "graphite_LGM50_electrolyte_exchange_current_density_Chen2020",
            pkl_obj.parameter_values[
                "Negative electrode exchange-current density [A.m-2]"
            ].__name__,
        )
        os.remove(filename)

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

        # with Casadi solver & experiment
        model.convert_to_format = "casadi"
        sim = pybamm.Simulation(
            model,
            experiment="Discharge at 1C for 20 minutes",
            solver=pybamm.CasadiSolver(),
        )
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

    def test_create_gif(self):
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.solve(t_eval=[0, 10])

        # create a GIF without calling the plot method
        sim.create_gif(number_of_images=3, duration=1)

        # call the plot method before creating the GIF
        sim.plot(testing=True)
        sim.create_gif(number_of_images=3, duration=1)

        os.remove("plot.gif")

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
        ).to_numpy()

        current_interpolant = pybamm.Interpolant(
            drive_cycle[:, 0], drive_cycle[:, 1], pybamm.t
        )

        param["Current function [A]"] = current_interpolant

        time_data = drive_cycle[:, 0]

        sim = pybamm.Simulation(model, parameter_values=param)

        # check solution is returned at the times in the data
        sim.solve()
        np.testing.assert_array_almost_equal(sim.solution.t, time_data)

        # check warning raised if the largest gap in t_eval is bigger than the
        # smallest gap in the data
        with self.assertWarns(pybamm.SolverWarning):
            sim.solve(t_eval=np.linspace(0, 10, 3))

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
        np.testing.assert_array_almost_equal(sim.solution.t, np.linspace(0, 10, 100))

    def test_battery_model_with_input_height(self):
        parameter_values = pybamm.ParameterValues("Marquis2019")
        model = pybamm.lithium_ion.SPM()
        parameter_values.update({"Electrode height [m]": "[input]"})
        # solve model for 1 minute
        t_eval = np.linspace(0, 60, 11)
        inputs = {"Electrode height [m]": 0.2}
        sim = pybamm.Simulation(model=model, parameter_values=parameter_values)
        sim.solve(t_eval=t_eval, inputs=inputs)


if __name__ == "__main__":
    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    unittest.main()
