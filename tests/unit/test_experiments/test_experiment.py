#
# Test the base experiment class
#
import pybamm
import numpy as np
import unittest
import pandas as pd
import os


class TestExperiment(unittest.TestCase):
    def test_read_strings(self):
        # Import drive cycle from file
        drive_cycle = pd.read_csv(
            pybamm.get_parameters_filepath(
                os.path.join("input", "drive_cycles", "US06.csv")
            ),
            comment="#",
            header=None,
        ).to_numpy()

        experiment = pybamm.Experiment(
            [
                "Discharge at 1C for 0.5 hours",
                "Discharge at C/20 for 0.5 hours",
                "Charge at 0.5 C for 45 minutes",
                "Discharge at 1 A for 0.5 hours",
                "Charge at 200 mA for 45 minutes (1 minute period)",
                "Discharge at 1W for 0.5 hours",
                "Charge at 200mW for 45 minutes",
                "Rest for 10 minutes (5 minute period)",
                "Hold at 1V for 20 seconds",
                "Charge at 1 C until 4.1V",
                "Hold at 4.1 V until 50mA",
                "Hold at 3V until C/50",
                "Discharge at C/3 for 2 hours or until 2.5 V",
                "Run US06 (A)",
                "Run US06 (V) for 5 minutes",
                "Run US06 (W) for 0.5 hours",
            ],
            {"test": "test"},
            drive_cycles={"US06": drive_cycle},
            period="20 seconds",
        )

        self.assertEqual(
            experiment.operating_conditions[:-3],
            [
                {"electric": (1, "C"), "time": 1800.0, "period": 20.0, "dc_data": None},
                {
                    "electric": (0.05, "C"),
                    "time": 1800.0,
                    "period": 20.0,
                    "dc_data": None,
                },
                {
                    "electric": (-0.5, "C"),
                    "time": 2700.0,
                    "period": 20.0,
                    "dc_data": None,
                },
                {"electric": (1, "A"), "time": 1800.0, "period": 20.0, "dc_data": None},
                {
                    "electric": (-0.2, "A"),
                    "time": 2700.0,
                    "period": 60.0,
                    "dc_data": None,
                },
                {"electric": (1, "W"), "time": 1800.0, "period": 20.0, "dc_data": None},
                {
                    "electric": (-0.2, "W"),
                    "time": 2700.0,
                    "period": 20.0,
                    "dc_data": None,
                },
                {"electric": (0, "A"), "time": 600.0, "period": 300.0, "dc_data": None},
                {"electric": (1, "V"), "time": 20.0, "period": 20.0, "dc_data": None},
                {"electric": (-1, "C"), "time": None, "period": 20.0, "dc_data": None},
                {"electric": (4.1, "V"), "time": None, "period": 20.0, "dc_data": None},
                {"electric": (3, "V"), "time": None, "period": 20.0, "dc_data": None},
                {
                    "electric": (1 / 3, "C"),
                    "time": 7200.0,
                    "period": 20.0,
                    "dc_data": None,
                },
            ],
        )
        # Calculation for operating conditions of drive cycle
        time_0 = drive_cycle[:, 0][-1]
        period_0 = np.min(np.diff(drive_cycle[:, 0]))
        drive_cycle_1 = experiment.extend_drive_cycle(drive_cycle, end_time=300)
        time_1 = drive_cycle_1[:, 0][-1]
        period_1 = np.min(np.diff(drive_cycle_1[:, 0]))
        drive_cycle_2 = experiment.extend_drive_cycle(drive_cycle, end_time=1800)
        time_2 = drive_cycle_2[:, 0][-1]
        period_2 = np.min(np.diff(drive_cycle_2[:, 0]))
        # Check drive cycle operating conditions
        np.testing.assert_array_equal(
            experiment.operating_conditions[-3]["dc_data"], drive_cycle
        )
        self.assertEqual(experiment.operating_conditions[-3]["electric"][1], "A")
        self.assertEqual(experiment.operating_conditions[-3]["time"], time_0)
        self.assertEqual(experiment.operating_conditions[-3]["period"], period_0)
        np.testing.assert_array_equal(
            experiment.operating_conditions[-2]["dc_data"], drive_cycle_1
        )
        self.assertEqual(experiment.operating_conditions[-2]["electric"][1], "V")
        self.assertEqual(experiment.operating_conditions[-2]["time"], time_1)
        self.assertEqual(experiment.operating_conditions[-2]["period"], period_1)
        np.testing.assert_array_equal(
            experiment.operating_conditions[-1]["dc_data"], drive_cycle_2
        )
        self.assertEqual(experiment.operating_conditions[-1]["electric"][1], "W")
        self.assertEqual(experiment.operating_conditions[-1]["time"], time_2)
        self.assertEqual(experiment.operating_conditions[-1]["period"], period_2)
        self.assertEqual(
            experiment.events,
            [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                (4.1, "V"),
                (0.05, "A"),
                (0.02, "C"),
                (2.5, "V"),
                None,
                None,
                None,
            ],
        )
        self.assertEqual(experiment.parameters, {"test": "test"})
        self.assertEqual(experiment.period, 20)

    def test_read_strings_cccv_combined(self):
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at C/20 for 0.5 hours",
                    "Charge at 0.5 C until 1V",
                    "Hold at 1V until C/50",
                    "Discharge at C/20 for 0.5 hours",
                ),
            ],
            cccv_handling="ode",
        )
        self.assertEqual(
            experiment.operating_conditions,
            [
                {
                    "electric": (0.05, "C"),
                    "time": 1800.0,
                    "period": 60.0,
                    "dc_data": None,
                },
                {
                    "electric": (-0.5, "C", 1, "V"),
                    "time": None,
                    "period": 60.0,
                    "dc_data": None,
                },
                {
                    "electric": (0.05, "C"),
                    "time": 1800.0,
                    "period": 60.0,
                    "dc_data": None,
                },
            ],
        )
        self.assertEqual(experiment.events, [None, (0.02, "C"), None])

        # Cases that don't quite match shouldn't do CCCV setup
        experiment = pybamm.Experiment(
            [
                "Charge at 0.5 C until 2V",
                "Hold at 1V until C/50",
            ],
            cccv_handling="ode",
        )
        self.assertEqual(
            experiment.operating_conditions,
            [
                {
                    "electric": (-0.5, "C"),
                    "time": None,
                    "period": 60.0,
                    "dc_data": None,
                },
                {"electric": (1, "V"), "time": None, "period": 60.0, "dc_data": None},
            ],
        )
        experiment = pybamm.Experiment(
            [
                "Charge at 0.5 C for 2 minutes",
                "Hold at 1V until C/50",
            ],
            cccv_handling="ode",
        )
        self.assertEqual(
            experiment.operating_conditions,
            [
                {
                    "electric": (-0.5, "C"),
                    "time": 120.0,
                    "period": 60.0,
                    "dc_data": None,
                },
                {"electric": (1, "V"), "time": None, "period": 60.0, "dc_data": None},
            ],
        )

    def test_read_strings_repeat(self):
        experiment = pybamm.Experiment(
            ["Discharge at 10 mA for 0.5 hours"]
            + ["Charge at 0.5 C for 45 minutes", "Hold at 1 V for 20 seconds"] * 2
        )
        self.assertEqual(
            experiment.operating_conditions,
            [
                {
                    "electric": (0.01, "A"),
                    "time": 1800.0,
                    "period": 60,
                    "dc_data": None,
                },
                {
                    "electric": (-0.5, "C"),
                    "time": 2700.0,
                    "period": 60,
                    "dc_data": None,
                },
                {"electric": (1, "V"), "time": 20.0, "period": 60, "dc_data": None},
                {
                    "electric": (-0.5, "C"),
                    "time": 2700.0,
                    "period": 60,
                    "dc_data": None,
                },
                {"electric": (1, "V"), "time": 20.0, "period": 60, "dc_data": None},
            ],
        )
        self.assertEqual(experiment.period, 60)

    def test_cycle_unpacking(self):
        experiment = pybamm.Experiment(
            [
                ("Discharge at C/20 for 0.5 hours", "Charge at C/5 for 45 minutes"),
                ("Discharge at C/20 for 0.5 hours"),
                "Charge at C/5 for 45 minutes",
            ]
        )
        self.assertEqual(
            experiment.operating_conditions,
            [
                {
                    "electric": (0.05, "C"),
                    "time": 1800.0,
                    "period": 60.0,
                    "dc_data": None,
                },
                {
                    "electric": (-0.2, "C"),
                    "time": 2700.0,
                    "period": 60.0,
                    "dc_data": None,
                },
                {
                    "electric": (0.05, "C"),
                    "time": 1800.0,
                    "period": 60.0,
                    "dc_data": None,
                },
                {
                    "electric": (-0.2, "C"),
                    "time": 2700.0,
                    "period": 60.0,
                    "dc_data": None,
                },
            ],
        )
        self.assertEqual(experiment.cycle_lengths, [2, 1, 1])

    def test_str_repr(self):
        conds = ["Discharge at 1 C for 20 seconds", "Charge at 0.5 W for 10 minutes"]
        experiment = pybamm.Experiment(conds)
        self.assertEqual(str(experiment), str(conds))
        self.assertEqual(
            repr(experiment),
            "pybamm.Experiment(['Discharge at 1 C for 20 seconds'"
            + ", 'Charge at 0.5 W for 10 minutes'])",
        )

    def test_bad_strings(self):
        with self.assertRaisesRegex(ValueError, "cccv_handling"):
            pybamm.Experiment(
                ["Discharge at 1 C for 20 seconds", "Charge at 0.5 W for 10 minutes"],
                cccv_handling="bad",
            )
        with self.assertRaisesRegex(
            TypeError, "Operating conditions should be strings or tuples of strings"
        ):
            pybamm.Experiment([1, 2, 3])
        with self.assertRaisesRegex(
            TypeError, "Operating conditions should be strings or tuples of strings"
        ):
            pybamm.Experiment([(1, 2, 3)])
        with self.assertRaisesRegex(ValueError, "Operating conditions must contain"):
            pybamm.Experiment(["Discharge at 1 A at 2 hours"])
        with self.assertRaisesRegex(ValueError, "Instruction must be"):
            pybamm.Experiment(["Run at 1 A for 2 hours"])
        with self.assertRaisesRegex(ValueError, "Type of drive cycle must be"):
            pybamm.Experiment(["Run US06 for 2 hours"])
        with self.assertRaisesRegex(ValueError, "Instruction must be"):
            pybamm.Experiment(["Run at at 1 A for 2 hours"])
        with self.assertRaisesRegex(ValueError, "Instruction must be"):
            pybamm.Experiment(["Play at 1 A for 2 hours"])
        with self.assertRaisesRegex(ValueError, "Instruction"):
            pybamm.Experiment(["Cell Charge at 1 A for 2 hours"])
        with self.assertRaisesRegex(ValueError, "units must be"):
            pybamm.Experiment(["Discharge at 1 B for 2 hours"])
        with self.assertRaisesRegex(ValueError, "time units must be"):
            pybamm.Experiment(["Discharge at 1 A for 2 years"])
        with self.assertRaisesRegex(
            TypeError, "experimental parameters should be a dictionary"
        ):
            pybamm.Experiment([], "not a dictionary")

    def test_termination(self):
        experiment = pybamm.Experiment(["Discharge at 1 C for 20 seconds"])
        self.assertEqual(experiment.termination, {})

        experiment = pybamm.Experiment(
            ["Discharge at 1 C for 20 seconds"], termination="80.7% capacity"
        )
        self.assertEqual(experiment.termination, {"capacity": (80.7, "%")})
        experiment = pybamm.Experiment(
            ["Discharge at 1 C for 20 seconds"], termination="80.7 % capacity"
        )
        self.assertEqual(experiment.termination, {"capacity": (80.7, "%")})

        experiment = pybamm.Experiment(
            ["Discharge at 1 C for 20 seconds"], termination="4.1Ah capacity"
        )
        self.assertEqual(experiment.termination, {"capacity": (4.1, "Ah")})
        experiment = pybamm.Experiment(
            ["Discharge at 1 C for 20 seconds"], termination="4.1 A.h capacity"
        )
        self.assertEqual(experiment.termination, {"capacity": (4.1, "Ah")})

        with self.assertRaisesRegex(ValueError, "Only capacity"):
            experiment = pybamm.Experiment(
                ["Discharge at 1 C for 20 seconds"], termination="bla bla capacity bla"
            )
        with self.assertRaisesRegex(ValueError, "Only capacity"):
            experiment = pybamm.Experiment(
                ["Discharge at 1 C for 20 seconds"], termination="4 A.h something else"
            )
        with self.assertRaisesRegex(ValueError, "Capacity termination"):
            experiment = pybamm.Experiment(
                ["Discharge at 1 C for 20 seconds"], termination="1 capacity"
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
