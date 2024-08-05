#
# Test setting up a simulation with an experiment
#
import casadi
import pybamm
import numpy as np
import os
import unittest
from datetime import datetime


class ShortDurationCRate(pybamm.step.CRate):
    def default_duration(self, value):
        # Set a short default duration for testing early stopping due to infeasible time
        return 1


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

        self.assertEqual(sim.experiment.args, experiment.args)
        steps = sim.experiment.steps

        model_I = sim.experiment_unique_steps_to_model[
            steps[1].basic_repr()
        ]  # CC charge
        model_V = sim.experiment_unique_steps_to_model[steps[2].basic_repr()]  # CV hold
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

    def test_setup_experiment_string_or_list(self):
        model = pybamm.lithium_ion.SPM()

        sim = pybamm.Simulation(model, experiment="Discharge at C/20 for 1 hour")
        sim.build_for_experiment()
        self.assertEqual(len(sim.experiment.steps), 1)
        self.assertEqual(
            sim.experiment.steps[0].description,
            "Discharge at C/20 for 1 hour",
        )
        sim = pybamm.Simulation(
            model,
            experiment=["Discharge at C/20 for 1 hour", pybamm.step.rest(60)],
        )
        sim.build_for_experiment()
        self.assertEqual(len(sim.experiment.steps), 2)

    def test_run_experiment(self):
        s = pybamm.step.string
        experiment = pybamm.Experiment(
            [
                (
                    s("Discharge at C/20 for 1 hour", temperature="30.5oC"),
                    s("Charge at 1 A until 4.1 V", temperature="24oC"),
                    s("Hold at 4.1 V until C/2", temperature="24oC"),
                    "Discharge at 2 W for 10 minutes",
                    "Discharge at 4 Ohm for 10 minutes",
                )
            ],
            temperature="-14oC",
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
        np.testing.assert_array_almost_equal(
            sol.cycles[0].steps[4]["Resistance [Ohm]"].data, 4, decimal=5
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
        s = pybamm.step.string
        experiment = pybamm.Experiment(
            [
                (
                    s("Discharge at C/20 for 1 hour", temperature="24oC"),
                    s("Charge at C/20 until 4.1 V", temperature="26oC"),
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
                    pybamm.step.current(drive_cycle, temperature="35oC"),
                    pybamm.step.voltage(drive_cycle),
                    pybamm.step.power(drive_cycle, termination="3 V"),
                )
            ],
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.build_for_experiment()
        self.assertEqual(
            sorted([step.basic_repr() for step in experiment.steps]),
            sorted(list(sim.experiment_unique_steps_to_model.keys())),
        )

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
        s = pybamm.step.string
        experiment = pybamm.Experiment(
            [
                (
                    "Rest for 10 minutes",
                    s("Discharge at 20 C for 10 minutes", period="10 minutes"),
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
                s("Discharge at 20 C for 10 minutes", period="10 minutes"),
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

    def test_run_experiment_infeasible_time(self):
        experiment = pybamm.Experiment(
            [ShortDurationCRate(1, termination="2.5V"), "Rest for 1 hour"]
        )
        model = pybamm.lithium_ion.SPM()
        parameter_values = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        )
        sol = sim.solve()
        self.assertEqual(len(sol.cycles), 1)
        self.assertEqual(len(sol.cycles[0].steps), 1)

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

    def test_run_experiment_with_pbar(self):
        # The only thing to test here is for errors.
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 1C for 1 sec",
                    "Charge at 1C for 1 sec",
                ),
            ]
            * 10,
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.solve(showprogress=True)

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

    def test_run_experiment_termination_time_min(self):
        experiment = pybamm.Experiment(
            [
                ("Discharge at 0.5C for 10 minutes", "Rest for 10 minutes"),
            ]
            * 5,
            termination="25 min",
        )
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        # Test with calc_esoh=False here
        sol = sim.solve(calc_esoh=False)
        # Only two cycles should be completed, only 2nd cycle should go below 4V
        np.testing.assert_array_less(np.max(sol.cycles[0]["Time [s]"].data), 1500)
        np.testing.assert_array_equal(np.max(sol.cycles[1]["Time [s]"].data), 1500)
        self.assertEqual(len(sol.cycles), 2)

    def test_run_experiment_termination_time_s(self):
        experiment = pybamm.Experiment(
            [
                ("Discharge at 0.5C for 10 minutes", "Rest for 10 minutes"),
            ]
            * 5,
            termination="1500 s",
        )
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        # Test with calc_esoh=False here
        sol = sim.solve(calc_esoh=False)
        # Only two cycles should be completed, only 2nd cycle should go below 4V
        np.testing.assert_array_less(np.max(sol.cycles[0]["Time [s]"].data), 1500)
        np.testing.assert_array_equal(np.max(sol.cycles[1]["Time [s]"].data), 1500)
        self.assertEqual(len(sol.cycles), 2)

    def test_run_experiment_termination_time_h(self):
        experiment = pybamm.Experiment(
            [
                ("Discharge at 0.5C for 10 minutes", "Rest for 10 minutes"),
            ]
            * 5,
            termination="0.5 h",
        )
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        # Test with calc_esoh=False here
        sol = sim.solve(calc_esoh=False)
        # Only two cycles should be completed, only 2nd cycle should go below 4V
        np.testing.assert_array_less(np.max(sol.cycles[0]["Time [s]"].data), 1800)
        np.testing.assert_array_equal(np.max(sol.cycles[1]["Time [s]"].data), 1800)
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
        for cycle_num in [1, 5, 6, 7]:
            self.assertIsNone(sol.cycles[cycle_num])
        for cycle_num in [0, 2, 3, 4, 8, 9]:  # first & last cycle always saved
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
        param["Negative particle diffusivity [m2.s-1]"] = (
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

    def test_skipped_step_continuous(self):
        model = pybamm.lithium_ion.SPM({"SEI": "solvent-diffusion limited"})
        experiment = pybamm.Experiment(
            [
                ("Rest for 24 hours (1 hour period)",),
                (
                    "Charge at C/3 until 4.1 V",
                    "Hold at 4.1V until C/20",
                    "Discharge at C/3 until 2.5 V",
                ),
            ]
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.solve(initial_soc=1)
        np.testing.assert_array_almost_equal(
            sim.solution.cycles[0].last_state.y.full(),
            sim.solution.cycles[1].steps[-1].first_state.y.full(),
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

    def test_solver_error(self):
        model = pybamm.lithium_ion.DFN()  # load model
        parameter_values = pybamm.ParameterValues("Chen2020")
        experiment = pybamm.Experiment(
            ["Discharge at 10C for 6 minutes or until 2.5 V"]
        )

        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            experiment=experiment,
            solver=pybamm.CasadiSolver(mode="fast"),
        )

        with self.assertRaisesRegex(pybamm.SolverError, "IDA_CONV_FAIL"):
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

    def test_padding_rest_model(self):
        model = pybamm.lithium_ion.SPM()

        # Test no padding rest model if there are no start_times
        experiment = pybamm.Experiment(["Rest for 1 hour"])
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.build_for_experiment()
        self.assertNotIn(
            "Rest for padding", sim.experiment_unique_steps_to_model.keys()
        )

        # Test padding rest model exists if there are start_times
        experiment = pybamm.step.string(
            "Rest for 1 hour", start_time=datetime(1, 1, 1, 8, 0, 0)
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.build_for_experiment()
        self.assertIn("Rest for padding", sim.experiment_unique_steps_to_model.keys())
        # Check at least there is an input parameter (temperature)
        self.assertGreater(
            len(sim.experiment_unique_steps_to_model["Rest for padding"].parameters), 0
        )
        # Check the model is the same
        self.assertIsInstance(
            sim.experiment_unique_steps_to_model["Rest for padding"],
            pybamm.lithium_ion.SPM,
        )

    def test_run_start_time_experiment(self):
        model = pybamm.lithium_ion.SPM()

        # Test experiment is cut short if next_start_time is early
        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at 0.5C for 1 hour",
                    start_time=datetime(2023, 1, 1, 8, 0, 0),
                ),
                pybamm.step.string(
                    "Rest for 1 hour", start_time=datetime(2023, 1, 1, 8, 30, 0)
                ),
            ]
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve(calc_esoh=False)
        self.assertEqual(sol["Time [s]"].entries[-1], 5400)

        # Test padding rest is added if time stamp is late
        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at 0.5C for 1 hour",
                    start_time=datetime(2023, 1, 1, 8, 0, 0),
                ),
                pybamm.step.string(
                    "Rest for 1 hour", start_time=datetime(2023, 1, 1, 10, 0, 0)
                ),
            ]
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve(calc_esoh=False)
        self.assertEqual(sol["Time [s]"].entries[-1], 10800)

    def test_starting_solution(self):
        model = pybamm.lithium_ion.SPM()

        experiment = pybamm.Experiment(
            [
                pybamm.step.string("Discharge at C/2 for 10 minutes"),
                pybamm.step.string("Rest for 5 minutes"),
                pybamm.step.string("Rest for 5 minutes"),
            ]
        )

        sim = pybamm.Simulation(model, experiment=experiment)
        solution = sim.solve(save_at_cycles=[1])

        # test that the last state is correct (i.e. final cycle is saved)
        self.assertEqual(solution.last_state.t[-1], 1200)

        experiment = pybamm.Experiment(
            [
                pybamm.step.string("Discharge at C/2 for 20 minutes"),
                pybamm.step.string("Rest for 20 minutes"),
            ]
        )

        sim = pybamm.Simulation(model, experiment=experiment)
        new_solution = sim.solve(calc_esoh=False, starting_solution=solution)

        # test that the final time is correct (i.e. starting solution correctly set)
        self.assertEqual(new_solution["Time [s]"].entries[-1], 3600)

    def test_experiment_start_time_starting_solution(self):
        model = pybamm.lithium_ion.SPM()

        # Test error raised if starting_solution does not have start_time
        experiment = pybamm.Experiment(
            [pybamm.step.string("Discharge at C/2 for 10 minutes")]
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        solution = sim.solve()

        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at C/2 for 10 minutes",
                    start_time=datetime(1, 1, 1, 9, 0, 0),
                )
            ]
        )

        sim = pybamm.Simulation(model, experiment=experiment)
        with self.assertRaisesRegex(ValueError, "experiments with `start_time`"):
            sim.solve(starting_solution=solution)

        # Test starting_solution works well with start_time
        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at C/2 for 10 minutes",
                    start_time=datetime(1, 1, 1, 8, 0, 0),
                ),
                pybamm.step.string(
                    "Discharge at C/2 for 10 minutes",
                    start_time=datetime(1, 1, 1, 8, 20, 0),
                ),
            ]
        )

        sim = pybamm.Simulation(model, experiment=experiment)
        solution = sim.solve()

        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at C/2 for 10 minutes",
                    start_time=datetime(1, 1, 1, 9, 0, 0),
                ),
                pybamm.step.string(
                    "Discharge at C/2 for 10 minutes",
                    start_time=datetime(1, 1, 1, 9, 20, 0),
                ),
            ]
        )

        sim = pybamm.Simulation(model, experiment=experiment)
        new_solution = sim.solve(starting_solution=solution)

        # test that the final time is correct (i.e. starting solution correctly set)
        self.assertEqual(new_solution["Time [s]"].entries[-1], 5400)

    def test_experiment_start_time_identical_steps(self):
        # Test that if we have the same step twice, with different start times,
        # they get processed only once
        model = pybamm.lithium_ion.SPM()

        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at C/2 for 10 minutes",
                    start_time=datetime(2023, 1, 1, 8, 0, 0),
                ),
                pybamm.step.string("Discharge at C/3 for 10 minutes"),
                pybamm.step.string(
                    "Discharge at C/2 for 10 minutes",
                    start_time=datetime(2023, 1, 1, 10, 0, 0),
                ),
                pybamm.step.string("Discharge at C/3 for 10 minutes"),
            ]
        )

        sim = pybamm.Simulation(model, experiment=experiment)
        sim.solve(calc_esoh=False)

        # Check that there are 4 steps
        self.assertEqual(len(experiment.steps), 4)

        # Check that there are only 2 unique steps
        self.assertEqual(len(sim.experiment.unique_steps), 2)

        # Check that there are only 3 built models (unique steps + padding rest)
        self.assertEqual(len(sim.steps_to_built_models), 3)

    def test_experiment_custom_steps(self):
        model = pybamm.lithium_ion.SPM()

        # Explicit control
        def custom_step_constant(variables):
            return 1

        custom_constant = pybamm.step.CustomStepExplicit(
            custom_step_constant, duration=1, period=0.1
        )

        experiment = pybamm.Experiment([custom_constant])
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve()
        np.testing.assert_array_equal(sol["Current [A]"].data, 1)

        # Implicit control (algebraic)
        def custom_step_voltage(variables):
            return 100 * (variables["Voltage [V]"] - 4.2)

        for control in ["differential"]:
            with self.subTest(control=control):
                custom_step_alg = pybamm.step.CustomStepImplicit(
                    custom_step_voltage, control=control, duration=100, period=10
                )

                experiment = pybamm.Experiment([custom_step_alg])
                sim = pybamm.Simulation(model, experiment=experiment)
                sol = sim.solve()
                # sol.plot()
                np.testing.assert_array_almost_equal(
                    sol["Voltage [V]"].data[2:], 4.2, decimal=3
                )

    def test_experiment_custom_termination(self):
        def neg_stoich_cutoff(variables):
            return variables["Negative electrode stoichiometry"] - 0.5

        neg_stoich_termination = pybamm.step.CustomTermination(
            name="Negative stoichiometry cut-off", event_function=neg_stoich_cutoff
        )

        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment(
            [pybamm.step.c_rate(1, termination=neg_stoich_termination)]
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve(calc_esoh=False)
        self.assertEqual(
            sol.cycles[0].steps[0].termination,
            "event: Negative stoichiometry cut-off [experiment]",
        )

        neg_stoich = sol["Negative electrode stoichiometry"].data
        self.assertAlmostEqual(neg_stoich[-1], 0.5, places=4)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
