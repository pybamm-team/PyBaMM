#
# Test the experiment data class
#
import pybamm
import unittest
import os
import pandas as pd


class TestExperimentData(unittest.TestCase):
    def setUp(self) -> None:
        # run an experiment to generate some data
        experiment = pybamm.Experiment(
            [
                ("Discharge at C/10 for 10 hours or until 3.3 V",
                "Rest for 1 hour",
                "Charge at 1 A until 4.1 V",
                "Hold at 4.1 V until 50 mA",
                "Rest for 1 hour"),
            ] * 3
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        self.sol = sim.solve()
        self.filename = 'test.csv'
        self.sol.save_data(self.filename, variables=['Time [s]', 'Voltage [V]', 'Current [A]'], to_format='csv')
        
        # add step and cycle information
        data = pd.read_csv(self.filename)
        steps = []
        cycles = []
        for (cycle_i, cycle) in enumerate(self.sol.cycles):
            for (step_i, step_ts) in enumerate(cycle.all_ts):
                for _ in step_ts:
                    steps.append(step_i)
                    cycles.append(cycle_i)
        data['Step'] = steps
        data['Cycle'] = cycles
        data.to_csv(self.filename)

        return super().setUp()

    def test_set_up(self):
        data = pd.read_csv(self.filename)
        self.assertIn('Time [s]', data.columns)
        self.assertIn('Voltage [V]', data.columns)
        self.assertIn('Step', data.columns)
    

    def test_quickplot(self):
        data = pd.read_csv(self.filename)
        pass

if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()


        