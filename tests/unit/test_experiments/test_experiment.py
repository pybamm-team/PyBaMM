#
# Test the base experiment class
#
import pybamm
import unittest


class TestExperiment(unittest.TestCase):
    def test_read_strings(self):
        experiment = pybamm.Experiment(["1 C for 0.5 hours", "0.5 C for 45 minutes"])
        self.assertEqual(
            experiment.operating_conditions, [(1, "C", 3600), (0.5, "C", 2700)]
        )

    def test_str_repr(self):
        conds = ["1 C for 20 seconds", "0.5 W for 10 minutes"]
        experiment = pybamm.Experiment(conds)
        self.assertEqual(str(experiment), conds)
        self.assertEqual(
            repr(experiment),
            "pybamm.Experiment(['1 C for 20 seconds', '0.5 W for 10 minutes'])",
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
