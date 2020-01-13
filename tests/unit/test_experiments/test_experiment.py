#
# Test the base experiment class
#
import pybamm
import unittest


class TestExperiment(unittest.TestCase):
    def test_read_strings(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at 1 C for 0.5 hours",
                "Charge at 0.5 C for 45 minutes",
                "Hold at 1 V for 20 seconds",
            ]
        )
        self.assertEqual(
            experiment.operating_conditions,
            [(1, "C", 1800.0), (-0.5, "C", 2700.0), (1, "V", 20.0)],
        )

    def test_read_strings_repeat(self):
        experiment = pybamm.Experiment(
            ["Discharge at 10 mA for 0.5 hours"]
            + ["Charge at 0.5 C for 45 minutes", "Hold at 1 V for 20 seconds"] * 2
        )
        self.assertEqual(
            experiment.operating_conditions,
            [
                (1, "C", 1800.0),
                (-0.5, "C", 2700.0),
                (1, "V", 20.0),
                (-0.5, "C", 2700.0),
                (1, "V", 20.0),
            ],
        )

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
        with self.assertRaisesRegex(ValueError, "length"):
            pybamm.Experiment(["three word string"])
        with self.assertRaisesRegex(TypeError, "First entry"):
            pybamm.Experiment(["Discharge at r C for 2 hours"])
        with self.assertRaisesRegex(ValueError, "Second entry"):
            pybamm.Experiment(["Discharge at 1 B for 2 hours"])
        with self.assertRaisesRegex(ValueError, "Third entry"):
            pybamm.Experiment(["Discharge at 1 C at 2 hours"])
        with self.assertRaisesRegex(TypeError, "Fourth entry"):
            pybamm.Experiment(["Discharge at 1 C for x hours"])


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
