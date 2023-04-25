#
# Test the base experiment class
#
from tests import TestCase
import pybamm
import unittest


class TestExperiment(TestCase):
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
                    "C-rate input [-]": 0.05,
                    "type": "C-rate",
                    "time": 1800.0,
                    "period": 60.0,
                    "temperature": None,
                    "dc_data": None,
                    "string": "Discharge at C/20 for 0.5 hours",
                    "events": None,
                    "tags": None,
                },
                {
                    "C-rate input [-]": -0.2,
                    "type": "C-rate",
                    "time": 2700.0,
                    "period": 60.0,
                    "temperature": None,
                    "dc_data": None,
                    "string": "Charge at C/5 for 45 minutes",
                    "events": None,
                    "tags": None,
                },
                {
                    "C-rate input [-]": 0.05,
                    "type": "C-rate",
                    "time": 1800.0,
                    "period": 60.0,
                    "temperature": None,
                    "dc_data": None,
                    "string": "Discharge at C/20 for 0.5 hours",
                    "events": None,
                    "tags": None,
                },
                {
                    "C-rate input [-]": -0.2,
                    "type": "C-rate",
                    "time": 2700.0,
                    "period": 60.0,
                    "temperature": None,
                    "dc_data": None,
                    "string": "Charge at C/5 for 45 minutes",
                    "events": None,
                    "tags": None,
                },
            ],
        )
        self.assertEqual(experiment.cycle_lengths, [2, 1, 1])

    def test_str_repr(self):
        conds = ["Discharge at 1 C for 20 seconds", "Charge at 0.5 W for 10 minutes"]
        experiment = pybamm.Experiment(conds)
        self.assertEqual(
            str(experiment),
            "[('Discharge at 1 C for 20 seconds',)"
            + ", ('Charge at 0.5 W for 10 minutes',)]",
        )
        self.assertEqual(
            repr(experiment),
            "pybamm.Experiment([('Discharge at 1 C for 20 seconds',)"
            + ", ('Charge at 0.5 W for 10 minutes',)])",
        )

    def test_bad_strings(self):
        with self.assertRaisesRegex(
            TypeError, "Operating conditions should be strings or tuples of strings"
        ):
            pybamm.Experiment([1, 2, 3])
        with self.assertRaisesRegex(
            TypeError, "Operating conditions should be strings or tuples of strings"
        ):
            pybamm.Experiment([(1, 2, 3)])

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

        experiment = pybamm.Experiment(
            ["Discharge at 1 C for 20 seconds"], termination="3V"
        )
        self.assertEqual(experiment.termination, {"voltage": (3, "V")})

        experiment = pybamm.Experiment(
            ["Discharge at 1 C for 20 seconds"], termination=["3V", "4.1Ah capacity"]
        )
        self.assertEqual(
            experiment.termination, {"voltage": (3, "V"), "capacity": (4.1, "Ah")}
        )

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

    def test_search_tag(self):
        experiment = pybamm.Experiment(
            [
                ("Discharge at 1C for 0.5 hours [tag1]",),
                "Discharge at C/20 for 0.5 hours [tag2,tag3]",
                (
                    "Charge at 0.5 C for 45 minutes [tag2]",
                    "Discharge at 1 A for 0.5 hours [tag3]",
                ),
                "Charge at 200 mA for 45 minutes (1 minute period) [tag5]",
                (
                    "Discharge at 1W for 0.5 hours [tag4]",
                    "Charge at 200mW for 45 minutes [tag4]",
                ),
                "Rest for 10 minutes (5 minute period) [tag1,tag3,tag4]",
            ]
        )

        self.assertEqual(experiment.search_tag("tag1"), [0, 5])
        self.assertEqual(experiment.search_tag("tag2"), [1, 2])
        self.assertEqual(experiment.search_tag("tag3"), [1, 2, 5])
        self.assertEqual(experiment.search_tag("tag4"), [4, 5])
        self.assertEqual(experiment.search_tag("tag5"), [3])
        self.assertEqual(experiment.search_tag("no_tag"), [])


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
