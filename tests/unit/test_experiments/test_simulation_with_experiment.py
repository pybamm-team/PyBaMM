#
# Test setting up a simulation with an experiment
#
import pybamm
import numpy as np
import unittest


class TestSimulationExperiment(unittest.TestCase):
    def test_set_up(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/20 for 1 hour",
                "Charge at 1 A until 4.1 V",
                "Hold at 4.1 V until 50 mA",
                "Discharge at 2 W for 1 hour",
            ]
        )
        model = pybamm.lithium_ion.DFN()
        sim = pybamm.Simulation(model, experiment=experiment)

        self.assertEqual(sim.experiment, experiment)
        self.assertEqual(
            sim._experiment_inputs[0]["Current input [A]"],
            1 / 20 * model.default_parameter_values["Nominal cell capacity [A.h]"],
        )
        self.assertEqual(sim._experiment_inputs[0]["Current switch"], 1)
        self.assertEqual(sim._experiment_inputs[0]["Voltage switch"], 0)
        self.assertEqual(sim._experiment_inputs[0]["Power switch"], 0)
        self.assertEqual(sim._experiment_inputs[0]["Current cut-off [A]"], -1e10)
        self.assertEqual(sim._experiment_inputs[0]["Voltage cut-off [V]"], -1e10)
        self.assertEqual(sim._experiment_inputs[1]["Current input [A]"], -1)
        self.assertEqual(sim._experiment_inputs[1]["Current switch"], 1)
        self.assertEqual(sim._experiment_inputs[1]["Voltage switch"], 0)
        self.assertEqual(sim._experiment_inputs[1]["Power switch"], 0)
        self.assertEqual(sim._experiment_inputs[1]["Current cut-off [A]"], -1e10)
        self.assertEqual(sim._experiment_inputs[1]["Voltage cut-off [V]"], 4.1)
        self.assertEqual(sim._experiment_inputs[2]["Current switch"], 0)
        self.assertEqual(sim._experiment_inputs[2]["Voltage switch"], 1)
        self.assertEqual(sim._experiment_inputs[2]["Power switch"], 0)
        self.assertEqual(sim._experiment_inputs[2]["Voltage input [V]"], 4.1)
        self.assertEqual(sim._experiment_inputs[2]["Current cut-off [A]"], 0.05)
        self.assertEqual(sim._experiment_inputs[2]["Voltage cut-off [V]"], -1e10)
        self.assertEqual(sim._experiment_inputs[3]["Current switch"], 0)
        self.assertEqual(sim._experiment_inputs[3]["Voltage switch"], 0)
        self.assertEqual(sim._experiment_inputs[3]["Power switch"], 1)
        self.assertEqual(sim._experiment_inputs[3]["Power input [W]"], 2)
        self.assertEqual(sim._experiment_inputs[3]["Current cut-off [A]"], -1e10)
        self.assertEqual(sim._experiment_inputs[3]["Voltage cut-off [V]"], -1e10)

        self.assertEqual(
            sim._experiment_times, [3600, 7 * 24 * 3600, 7 * 24 * 3600, 3600]
        )

        model_I = sim.op_conds_to_model_and_param[(-1.0, "A")][0]
        model_V = sim.op_conds_to_model_and_param[(4.1, "V")][0]
        self.assertIn(
            "Current cut-off (positive) [A] [experiment]",
            [event.name for event in model_V.events],
        )
        self.assertIn(
            "Current cut-off (negative) [A] [experiment]",
            [event.name for event in model_V.events],
        )
        self.assertIn(
            "Voltage cut-off [V] [experiment]",
            [event.name for event in model_I.events],
        )

        # fails if trying to set up with something that isn't an experiment
        with self.assertRaisesRegex(TypeError, "experiment must be"):
            pybamm.Simulation(model, experiment=0)

    def test_run_experiment(self):
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at C/20 for 1 hour",
                    "Charge at 1 A until 4.1 V",
                    "Hold at 4.1 V until C/2",
                    "Discharge at 2 W for 1 hour",
                )
            ]
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve()
        self.assertEqual(sol.termination, "final time")
        self.assertEqual(len(sol.cycles), 1)

        # Solve again starting from solution
        sol2 = sim.solve(starting_solution=sol)
        self.assertEqual(sol2.termination, "final time")
        self.assertGreater(sol2.t[-1], sol.t[-1])
        self.assertEqual(sol2.cycles[0], sol.cycles[0])
        self.assertEqual(len(sol2.cycles), 2)
        self.assertEqual(len(sol.cycles), 1)

    def test_run_experiment_old_setup_type(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/20 for 1 hour",
                "Charge at 1 A until 4.1 V",
                "Hold at 4.1 V until C/2",
                "Discharge at 2 W for 1 hour",
            ],
            use_simulation_setup_type="old",
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.solve(solver=pybamm.CasadiSolver())
        self.assertEqual(sim._solution.termination, "final time")

    def test_run_experiment_breaks_early(self):
        experiment = pybamm.Experiment(["Discharge at 2 C for 1 hour"])
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        pybamm.set_logging_level("ERROR")
        # giving the time, should get ignored
        t_eval = [0, 1]
        sim.solve(t_eval, solver=pybamm.CasadiSolver())
        pybamm.set_logging_level("WARNING")
        self.assertEqual(sim._solution, None)

    def test_inputs(self):
        experiment = pybamm.Experiment(
            ["Discharge at C/2 for 1 hour", "Rest for 1 hour"]
        )
        model = pybamm.lithium_ion.SPM()

        # Change a parameter to an input
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)
        param["Negative electrode diffusivity [m2.s-1]"] = (
            pybamm.InputParameter("Dsn") * 3.9e-14
        )

        # Solve a first time
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sim.solve(inputs={"Dsn": 1})
        np.testing.assert_array_equal(sim.solution.all_inputs[0]["Dsn"], 1)

        # Solve again, input should change
        sim.solve(inputs={"Dsn": 2})
        np.testing.assert_array_equal(sim.solution.all_inputs[0]["Dsn"], 2)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
