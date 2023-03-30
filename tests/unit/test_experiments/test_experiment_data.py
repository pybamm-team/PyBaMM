#
# Test the experiment data class
#
import pybamm
import unittest

# import os
import pandas as pd


class TestExperimentData(unittest.TestCase):
    def setUp(self) -> None:
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at C/10 for 10 hours or until 3.3 V",
                    "Rest for 1 hour",
                    "Charge at 1 A until 4.1 V",
                    "Hold at 4.1 V until 50 mA",
                    "Rest for 1 hour",
                ),
            ]
            * 3
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve()
        self.filename = "test.csv"
        sol.save_data(
            self.filename,
            variables=["Time [s]", "Voltage [V]", "Current [A]"],
            to_format="csv",
        )
        return super().setUp()

    def test_set_up(self):
        data = pd.read_csv(self.filename)
        self.assertIn("Time [s]", data.columns)
        self.assertIn("Voltage [V]", data.columns)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
