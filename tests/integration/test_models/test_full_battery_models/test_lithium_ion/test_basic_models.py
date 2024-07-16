#
# Test basic model classes
#
import pybamm
import pytest


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


class TestBasicSPM(BaseBasicModelTest):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.model = pybamm.lithium_ion.BasicSPM()


class TestBasicDFN(BaseBasicModelTest):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.model = pybamm.lithium_ion.BasicDFN()


class TestBasicDFNComposite(BaseBasicModelTest):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.model = pybamm.lithium_ion.BasicDFNComposite()


class TestBasicDFNHalfCell(BaseBasicModelTest):
    @pytest.fixture(autouse=True)
    def setUp(self):
        options = {"working electrode": "positive"}
        self.model = pybamm.lithium_ion.BasicDFNHalfCell(options)
