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
            1 / 20 * model.default_parameter_values["Cell capacity [A.h]"],
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

        self.assertIn(
            "Current cut-off (positive) [A] [experiment]",
            [event.name for event in sim.model.events],
        )
        self.assertIn(
            "Current cut-off (negative) [A] [experiment]",
            [event.name for event in sim.model.events],
        )
        self.assertIn(
            "Voltage cut-off [V] [experiment]",
            [event.name for event in sim.model.events],
        )

        # fails if trying to set up with something that isn't an experiment
        with self.assertRaisesRegex(TypeError, "experiment must be"):
            pybamm.Simulation(model, experiment=0)

    def test_run_experiment(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/20 for 1 hour",
                "Charge at 1 A until 4.1 V",
                "Hold at 4.1 V until C/2",
                "Discharge at 2 W for 1 hour",
            ]
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
        self.assertIn("event", sim._solution.termination)

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
        np.testing.assert_array_equal(sim.solution.inputs["Dsn"], 1)

        # Solve again, input should change
        sim.solve(inputs={"Dsn": 2})
        np.testing.assert_array_equal(sim.solution.inputs["Dsn"], 2)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
