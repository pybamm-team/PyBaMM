#
# Test basic model classes
#
from tests import TestCase
import pybamm

import unittest


class BaseBasicModelTest:
    def test_with_experiment(self):
        model = self.model
        experiment = pybamm.Experiment(
            [
                "Discharge at C/3 until 3.5V",
                "Hold at 3.5V for 1 hour",
                "Rest for 10 min",
            ]
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.solve(calc_esoh=False)


class TestBasicSPM(BaseBasicModelTest, TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.SPM()


class TestBasicDFN(BaseBasicModelTest, TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.DFN()


class TestBasicDFNComposite(BaseBasicModelTest, TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.BasicDFNComposite()


class TestBasicDFNHalfCell(BaseBasicModelTest, TestCase):
    def setUp(self):
        options = {"working electrode": "positive"}
        self.model = pybamm.lithium_ion.BasicDFNHalfCell(options)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
