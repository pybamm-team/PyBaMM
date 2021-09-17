#
# Test setting up a simulation with an experiment
#
import casadi
import pybamm
import numpy as np
import os
import pandas as pd
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

        Crate = 1 / model.default_parameter_values["Nominal cell capacity [A.h]"]
        self.assertEqual(
            sim._experiment_times, [3600, 3 / Crate * 3600, 24 * 3600, 3600]
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
        model = pybamm.lithium_ion.DFN()
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve()
        self.assertEqual(sol.termination, "final time")
        self.assertEqual(len(sol.cycles), 1)

        for i, step in enumerate(sol.cycles[0].steps[:-1]):
            len_rhs = sol.all_models[0].concatenated_rhs.size
            y_left = step.all_ys[-1][:len_rhs, -1]
            if isinstance(y_left, casadi.DM):
                y_left = y_left.full()
            y_right = sol.cycles[0].steps[i + 1].all_ys[0][:len_rhs, 0]
            if isinstance(y_right, casadi.DM):
                y_right = y_right.full()
            np.testing.assert_array_equal(y_left.flatten(), y_right.flatten())

        # Solve again starting from solution
        sol2 = sim.solve(starting_solution=sol)
        self.assertEqual(sol2.termination, "final time")
        self.assertGreater(sol2.t[-1], sol.t[-1])
        self.assertEqual(sol2.cycles[0], sol.cycles[0])
        self.assertEqual(len(sol2.cycles), 2)
        # Check starting solution is unchanged
        self.assertEqual(len(sol.cycles), 1)

        # save
        sol2.save("test_experiment.sav")
        sol3 = pybamm.load("test_experiment.sav")
        self.assertEqual(len(sol3.cycles), 2)
        os.remove("test_experiment.sav")

    def test_run_experiment_cccv_ode(self):
        experiment_2step = pybamm.Experiment(
            [
                (
                    "Discharge at C/20 for 1 hour",
                    "Charge at 1 A until 4.1 V",
                    "Hold at 4.1 V until C/2",
                    "Discharge at 2 W for 1 hour",
                ),
            ],
        )
        experiment_ode = pybamm.Experiment(
            [
                (
                    "Discharge at C/20 for 1 hour",
                    "Charge at 1 A until 4.1 V",
                    "Hold at 4.1 V until C/2",
                    "Discharge at 2 W for 1 hour",
                ),
            ],
            cccv_handling="ode",
        )
        solutions = []
        for experiment in [experiment_2step, experiment_ode]:
            model = pybamm.lithium_ion.SPM()
            sim = pybamm.Simulation(model, experiment=experiment)
            solution = sim.solve(solver=pybamm.CasadiSolver("fast with events"))
            solutions.append(solution)

        np.testing.assert_array_almost_equal(
            solutions[0]["Terminal voltage [V]"].data,
            solutions[1]["Terminal voltage [V]"].data,
            decimal=1,
        )
        np.testing.assert_array_almost_equal(
            solutions[0]["Current [A]"].data,
            solutions[1]["Current [A]"].data,
            decimal=0,
        )
        self.assertEqual(solutions[1].termination, "final time")

    def test_run_experiment_old_setup_type(self):
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at C/20 for 1 hour",
                    "Charge at 1 A until 4.1 V",
                    "Hold at 4.1 V until C/2",
                    "Discharge at 2 W for 1 hour",
                ),
            ],
            use_simulation_setup_type="old",
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        solution1 = sim.solve(solver=pybamm.CasadiSolver())
        self.assertEqual(solution1.termination, "final time")

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

    def test_run_experiment_termination(self):
        # with percent
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 1C until 3V",
                    "Charge at 1C until 4.2 V",
                    "Hold at 4.2V until C/10",
                ),
            ]
            * 10,
            termination="99% capacity",
        )
        model = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited"})
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
        param["SEI kinetic rate constant [m.s-1]"] = 1e-14
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sol = sim.solve(solver=pybamm.CasadiSolver())
        C = sol.summary_variables["Capacity [A.h]"]
        np.testing.assert_array_less(np.diff(C), 0)
        # all but the last value should be above the termination condition
        np.testing.assert_array_less(0.99 * C[0], C[:-1])

        # with Ah value
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 1C until 3V",
                    "Charge at 1C until 4.2 V",
                    "Hold at 4.2V until C/10",
                ),
            ]
            * 10,
            termination="5.04Ah capacity",
        )
        model = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited"})
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
        param["SEI kinetic rate constant [m.s-1]"] = 1e-14
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sol = sim.solve(solver=pybamm.CasadiSolver())
        # all but the last value should be above the termination condition
        np.testing.assert_array_less(5.04, C[:-1])

    def test_save_at_cycles(self):
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 1C until 3.3V",
                    "Charge at 1C until 4.1 V",
                    "Hold at 4.1V until C/10",
                ),
            ]
            * 10,
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve(
            solver=pybamm.CasadiSolver("fast with events"), save_at_cycles=2
        )
        # Solution saves "None" for the cycles that are not saved
        for cycle_num in [2, 4, 6, 8]:
            self.assertIsNone(sol.cycles[cycle_num])
        for cycle_num in [0, 1, 3, 5, 7, 9]:
            self.assertIsNotNone(sol.cycles[cycle_num])
        # Summary variables are not None
        self.assertIsNotNone(sol.summary_variables["Capacity [A.h]"])

        sol = sim.solve(
            solver=pybamm.CasadiSolver("fast with events"), save_at_cycles=[3, 4, 5, 9]
        )
        # Note offset by 1 (0th cycle is cycle 1)
        for cycle_num in [1, 5, 6, 7, 9]:
            self.assertIsNone(sol.cycles[cycle_num])
        for cycle_num in [0, 2, 3, 4, 8]:
            self.assertIsNotNone(sol.cycles[cycle_num])
        # Summary variables are not None
        self.assertIsNotNone(sol.summary_variables["Capacity [A.h]"])

    def test_cycle_summary_variables(self):
        # Test cycle_summary_variables works for different combinations of data and
        # function OCPs
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 1C until 3.3V",
                    "Charge at C/3 until 4.0V",
                    "Hold at 4.0V until C/10",
                ),
            ]
            * 5,
        )
        model = pybamm.lithium_ion.SPM()

        # Chen 2020 plating: pos = function, neg = data
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020_plating)
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sim.solve(solver=pybamm.CasadiSolver("fast with events"), save_at_cycles=2)

        # Chen 2020: pos = function, neg = function
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sim.solve(solver=pybamm.CasadiSolver("fast with events"), save_at_cycles=2)

        # Chen 2020 with data: pos = data, neg = data
        # Load negative electrode OCP data
        filename = os.path.join(
            pybamm.root_dir(),
            "pybamm",
            "input",
            "parameters",
            "lithium_ion",
            "negative_electrodes",
            "graphite_Chen2020",
            "graphite_LGM50_ocp_Chen2020.csv",
        )
        function_name = "graphite_Chen2020"
        filename = pybamm.get_parameters_filepath(filename)
        data = pd.read_csv(
            filename, comment="#", skip_blank_lines=True, header=None
        ).to_numpy()
        param["Negative electrode OCP [V]"] = (function_name, data)

        # Load positive electrode OCP data
        filename = os.path.join(
            pybamm.root_dir(),
            "pybamm",
            "input",
            "parameters",
            "lithium_ion",
            "positive_electrodes",
            "nmc_Chen2020",
            "nmc_LGM50_ocp_Chen2020.csv",
        )
        function_name = "nmc_LGM50_ocp_Chen2020.csv"
        filename = pybamm.get_parameters_filepath(filename)
        data = pd.read_csv(
            filename, comment="#", skip_blank_lines=True, header=None
        ).to_numpy()
        param["Positive electrode OCP [V]"] = (function_name, data)

        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sim.solve(solver=pybamm.CasadiSolver("safe"), save_at_cycles=2)

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
