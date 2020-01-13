#
# Test setting up a simulation with an experiment
#
import pybamm
import unittest


class TestSimulationExperiment(unittest.TestCase):
    def test_set_up(self):
        experiment = pybamm.Experiment(
            ["1 C for 0.5 hours", "0.5 C for 45 minutes", "1 V for 20 seconds"]
        )
        self.assertEqual(
            experiment.operating_conditions,
            [(1, "C", 1800.0), (0.5, "C", 2700.0), (1, "V", 20.0)],
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
