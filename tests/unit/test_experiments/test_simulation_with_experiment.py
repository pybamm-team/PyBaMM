#
# Test setting up a simulation with an experiment
#
import casadi
import pybamm
import numpy as np
import os
import unittest


class TestSimulationExperiment(unittest.TestCase):
    def test_set_up(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/20 for 1 hour",
                "Charge at 1 A until 4.1 V",
                "Hold at 4.1 V until 50 mA",
                "Discharge at 2 W until 3.5 V",
            ]
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.build_for_experiment()
        C = model.default_parameter_values["Nominal cell capacity [A.h]"]

        self.assertEqual(sim.experiment.args, experiment.args)
        op_conds = sim.experiment.operating_conditions
        self.assertEqual(op_conds[0]["Current input [A]"], C / 20)
        self.assertEqual(op_conds[1]["Current input [A]"], -1)
        self.assertEqual(op_conds[2]["Voltage input [V]"], 4.1)
        self.assertEqual(op_conds[3]["Power input [W]"], 2)

        Crate = 1 / C
        self.assertEqual(
            [op["time"] for op in op_conds],
            [3600, 3 / Crate * 3600, 24 * 3600, 24 * 3600],
        )

        model_I = sim.op_string_to_model["Charge at 1 A until 4.1 V"]
        model_V = sim.op_string_to_model["Hold at 4.1 V until 50 mA"]
        self.assertIn(
            "Current cut-off [A] [experiment]",
            [event.name for event in model_V.events],
        )
        self.assertIn(
            "Charge voltage cut-off [V] [experiment]",
            [event.name for event in model_I.events],
        )

        # fails if trying to set up with something that isn't an experiment
        with self.assertRaisesRegex(TypeError, "experiment must be"):
            pybamm.Simulation(model, experiment=0)

    def test_run_experiment(self):
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at C/20 for 1 hour at 30.5oC",
                    "Charge at 1 A until 4.1 V at 24oC",
                    "Hold at 4.1 V until C/2 at 24oC",
                    "Discharge at 2 W for 1 hour",
                )
            ],
            temperature=-14,
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        # test the callback here
        sol = sim.solve(callbacks=pybamm.callbacks.Callback())
        self.assertEqual(sol.termination, "final time")
        self.assertEqual(len(sol.cycles), 1)

        # Test outputs
        np.testing.assert_array_equal(sol.cycles[0].steps[0]["C-rate"].data, 1 / 20)
        np.testing.assert_array_equal(sol.cycles[0].steps[1]["Current [A]"].data, -1)
        np.testing.assert_array_almost_equal(
            sol.cycles[0].steps[2]["Voltage [V]"].data, 4.1, decimal=5
        )
        np.testing.assert_array_almost_equal(
            sol.cycles[0].steps[3]["Power [W]"].data, 2, decimal=5
        )

        np.testing.assert_array_equal(
            sol.cycles[0].steps[0]["Ambient temperature [C]"].data[0], 30.5
        )

        np.testing.assert_array_equal(
            sol.cycles[0].steps[1]["Ambient temperature [C]"].data[0], 24
        )

        np.testing.assert_array_equal(
            sol.cycles[0].steps[2]["Ambient temperature [C]"].data[0], 24
        )

        np.testing.assert_array_equal(
            sol.cycles[0].steps[3]["Ambient temperature [C]"].data[0], -14
        )

        for i, step in enumerate(sol.cycles[0].steps[:-1]):
            len_rhs = sol.all_models[0].concatenated_rhs.size
            y_left = step.all_ys[-1][:len_rhs, -1]
            if isinstance(y_left, casadi.DM):
                y_left = y_left.full()
            y_right = sol.cycles[0].steps[i + 1].all_ys[0][:len_rhs, 0]
            if isinstance(y_right, casadi.DM):
                y_right = y_right.full()
            np.testing.assert_array_almost_equal(y_left.flatten(), y_right.flatten())

        # Solve again starting from solution
        sol2 = sim.solve(starting_solution=sol)
        self.assertEqual(sol2.termination, "final time")
        self.assertGreater(sol2.t[-1], sol.t[-1])
        self.assertEqual(sol2.cycles[0], sol.cycles[0])
        self.assertEqual(len(sol2.cycles), 2)
        # Solve again starting from solution but only inputting the cycle
        sol2 = sim.solve(starting_solution=sol.cycles[-1])
        self.assertEqual(sol2.termination, "final time")
        self.assertGreater(sol2.t[-1], sol.t[-1])
        self.assertEqual(len(sol2.cycles), 2)

        # Check starting solution is unchanged
        self.assertEqual(len(sol.cycles), 1)

        # save
        sol2.save("test_experiment.sav")
        sol3 = pybamm.load("test_experiment.sav")
        self.assertEqual(len(sol3.cycles), 2)
        os.remove("test_experiment.sav")

    def test_run_experiment_multiple_times(self):
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at C/20 for 1 hour at 24oC",
                    "Charge at C/20 until 4.1 V at 26oC",
                )
            ]
            * 3
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)

        # Test that solving twice gives the same solution (see #2193)
        sol1 = sim.solve()
        sol2 = sim.solve()
        np.testing.assert_array_equal(
            sol1["Voltage [V]"].data, sol2["Voltage [V]"].data
        )

    def test_run_experiment_cccv_ode(self):
        experiment_2step = pybamm.Experiment(
            [
                (
                    "Discharge at C/20 for 1 hour at 20oC",
                    "Charge at 1 A until 4.1 V at 24oC",
                    "Hold at 4.1 V until C/2 at 24oC",
                    "Discharge at 2 W for 1 hour",
                ),
            ],
        )
        experiment_ode = pybamm.Experiment(
            [
                (
                    "Discharge at C/20 for 1 hour at 20oC",
                    "Charge at 1 A until 4.1 V at 24oC",
                    "Hold at 4.1 V until C/2 at 24oC",
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

        t = solutions[1]["Time [s]"].data
        np.testing.assert_array_almost_equal(
            solutions[0]["Voltage [V]"](t=t),
            solutions[1]["Voltage [V]"](t=t),
            decimal=1,
        )
        np.testing.assert_array_almost_equal(
            solutions[0]["Current [A]"](t=t),
            solutions[1]["Current [A]"](t=t),
            decimal=0,
        )

        self.assertEqual(solutions[1].termination, "final time")

    @unittest.skipIf(not pybamm.have_idaklu(), "idaklu solver is not installed")
    def test_run_experiment_cccv_solvers(self):
        experiment_2step = pybamm.Experiment(
            [
                (
                    "Discharge at C/20 for 1 hour",
                    "Charge at 1 A until 4.1 V",
                    "Hold at 4.1 V until C/2",
                    "Discharge at 2 W for 1 hour",
                ),
            ]
            * 2,
        )

        solutions = []
        for solver in [pybamm.CasadiSolver(), pybamm.IDAKLUSolver()]:
            model = pybamm.lithium_ion.SPM()
            sim = pybamm.Simulation(model, experiment=experiment_2step, solver=solver)
            solution = sim.solve()
            solutions.append(solution)

        np.testing.assert_array_almost_equal(
            solutions[0]["Voltage [V]"].data,
            solutions[1]["Voltage [V]"].data,
            decimal=1,
        )
        np.testing.assert_array_almost_equal(
            solutions[0]["Current [A]"].data,
            solutions[1]["Current [A]"].data,
            decimal=0,
        )
        self.assertEqual(solutions[1].termination, "final time")

    def test_run_experiment_drive_cycle(self):
        drive_cycle = np.array([np.arange(10), np.arange(10)]).T
        experiment = pybamm.Experiment(
            [
                (
                    "Run drive_cycle (A) at 35oC",
                    "Run drive_cycle (V)",
                    "Run drive_cycle (W)",
                )
            ],
            drive_cycles={"drive_cycle": drive_cycle},
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.build_for_experiment()
        self.assertIn(("Run drive_cycle (A) at 35oC"), sim.op_string_to_model)
        self.assertIn(("Run drive_cycle (V)"), sim.op_string_to_model)
        self.assertIn(("Run drive_cycle (W)"), sim.op_string_to_model)

    def test_run_experiment_breaks_early_infeasible(self):
        experiment = pybamm.Experiment(["Discharge at 2 C for 1 hour"])
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        pybamm.set_logging_level("ERROR")
        # giving the time, should get ignored
        t_eval = [0, 1]
        sim.solve(
            t_eval, solver=pybamm.CasadiSolver(), callbacks=pybamm.callbacks.Callback()
        )
        pybamm.set_logging_level("WARNING")
        self.assertEqual(sim._solution.termination, "event: Minimum voltage [V]")

    def test_run_experiment_breaks_early_error(self):
        experiment = pybamm.Experiment(
            [
                (
                    "Rest for 10 minutes",
                    "Discharge at 20 C for 10 minutes (10 minute period)",
                )
            ]
        )
        model = pybamm.lithium_ion.DFN()

        parameter_values = pybamm.ParameterValues("Chen2020")
        solver = pybamm.CasadiSolver(max_step_decrease_count=2)
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=parameter_values,
            solver=solver,
        )
        sol = sim.solve()
        self.assertEqual(len(sol.cycles), 1)
        self.assertEqual(len(sol.cycles[0].steps), 1)

        # Different experiment setup style
        experiment = pybamm.Experiment(
            [
                "Rest for 10 minutes",
                "Discharge at 20 C for 10 minutes (10 minute period)",
            ]
        )
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=parameter_values,
            solver=solver,
        )
        sol = sim.solve()
        self.assertEqual(len(sol.cycles), 1)
        self.assertEqual(len(sol.cycles[0].steps), 1)

        # Different callback - this is for coverage on the `Callback` class
        sol = sim.solve(callbacks=pybamm.callbacks.Callback())

    def test_run_experiment_termination_capacity(self):
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
        param = pybamm.ParameterValues("Chen2020")
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
        param = pybamm.ParameterValues("Chen2020")
        param["SEI kinetic rate constant [m.s-1]"] = 1e-14
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sol = sim.solve(solver=pybamm.CasadiSolver())
        # all but the last value should be above the termination condition
        np.testing.assert_array_less(5.04, C[:-1])

    def test_run_experiment_termination_voltage(self):
        # with percent
        experiment = pybamm.Experiment(
            [
                ("Discharge at 0.5C for 10 minutes", "Rest for 10 minutes"),
            ]
            * 5,
            termination="4V",
        )
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        # Test with calc_esoh=False here
        sol = sim.solve(calc_esoh=False)
        # Only two cycles should be completed, only 2nd cycle should go below 4V
        np.testing.assert_array_less(4, np.min(sol.cycles[0]["Voltage [V]"].data))
        np.testing.assert_array_less(np.min(sol.cycles[1]["Voltage [V]"].data), 4)
        self.assertEqual(len(sol.cycles), 2)

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

        # O'Kane 2022: pos = function, neg = data
        param = pybamm.ParameterValues("OKane2022")
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sim.solve(solver=pybamm.CasadiSolver("fast with events"), save_at_cycles=2)

        # Chen 2020: pos = function, neg = function
        param = pybamm.ParameterValues("Chen2020")
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
            "data",
            "graphite_LGM50_ocp_Chen2020.csv",
        )
        param["Negative electrode OCP [V]"] = pybamm.parameters.process_1D_data(
            filename
        )

        # Load positive electrode OCP data
        filename = os.path.join(
            pybamm.root_dir(),
            "pybamm",
            "input",
            "parameters",
            "lithium_ion",
            "data",
            "nmc_LGM50_ocp_Chen2020.csv",
        )
        param["Positive electrode OCP [V]"] = pybamm.parameters.process_1D_data(
            filename
        )

        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sim.solve(solver=pybamm.CasadiSolver("safe"), save_at_cycles=2)

    def test_inputs(self):
        experiment = pybamm.Experiment(
            ["Discharge at C/2 for 1 hour", "Rest for 1 hour"]
        )
        model = pybamm.lithium_ion.SPM()

        # Change a parameter to an input
        param = pybamm.ParameterValues("Marquis2019")
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

    def test_run_experiment_skip_steps(self):
        # Test experiment with steps being skipped due to initial conditions
        # already satisfying the events
        model = pybamm.lithium_ion.SPM()
        parameter_values = pybamm.ParameterValues("Chen2020")
        experiment = pybamm.Experiment(
            [
                (
                    "Charge at 1C until 4.2V",
                    "Hold at 4.2V until 10 mA",
                    "Discharge at 1C for 1 hour",
                    "Charge at 20C until 3V",
                    "Hold at 3V until 10 mA",
                ),
            ]
        )
        sim = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        )
        sol = sim.solve()
        self.assertIsInstance(sol.cycles[0].steps[0], pybamm.EmptySolution)
        self.assertIsInstance(sol.cycles[0].steps[3], pybamm.EmptySolution)

        # Should get the same result if we run without the charge steps
        # since they are skipped
        experiment2 = pybamm.Experiment(
            [
                (
                    "Hold at 4.2V until 10 mA",
                    "Discharge at 1C for 1 hour",
                    "Hold at 3V until 10 mA",
                ),
            ]
        )
        sim2 = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment2
        )
        sol2 = sim2.solve()
        np.testing.assert_array_almost_equal(
            sol["Voltage [V]"].data, sol2["Voltage [V]"].data, decimal=5
        )
        for idx1, idx2 in [(1, 0), (2, 1), (4, 2)]:
            np.testing.assert_array_almost_equal(
                sol.cycles[0].steps[idx1]["Voltage [V]"].data,
                sol2.cycles[0].steps[idx2]["Voltage [V]"].data,
                decimal=5,
            )

    def test_all_empty_solution_errors(self):
        model = pybamm.lithium_ion.SPM()
        parameter_values = pybamm.ParameterValues("Chen2020")

        # One step exceeded, suggests making a cycle
        experiment = pybamm.Experiment([("Charge at 1C until 4.2V")])
        sim = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        )
        with self.assertRaisesRegex(
            pybamm.SolverError,
            "Step 'Charge at 1C until 4.2V' is infeasible due to exceeded bounds",
        ):
            sim.solve()

        # Two steps exceeded, different error
        experiment = pybamm.Experiment(
            [("Charge at 1C until 4.2V", "Charge at 1C until 4.2V")]
        )
        sim = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        )
        with self.assertRaisesRegex(pybamm.SolverError, "All steps in the cycle"):
            sim.solve()

    def test_run_experiment_half_cell(self):
        experiment = pybamm.Experiment(
            [("Discharge at C/20 until 3.5V", "Charge at 1C until 3.8 V")]
        )
        model = pybamm.lithium_ion.SPM({"working electrode": "positive"})
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=pybamm.ParameterValues("Xu2019"),
        )
        sim.solve()

    def test_run_experiment_lead_acid(self):
        experiment = pybamm.Experiment(
            [("Discharge at C/20 until 10.5V", "Charge at C/20 until 12.5 V")]
        )
        model = pybamm.lead_acid.Full()
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.solve()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
