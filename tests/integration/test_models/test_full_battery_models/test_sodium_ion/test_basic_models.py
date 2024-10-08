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
        sol = sim.solve(calc_esoh=False)

        # Check the solve returned a solution
        assert sol is not None

        # Check that the solution contains the expected number of cycles
        assert len(sol.cycles) == 3

        # Check that the solution terminated because it reached final time
        assert sol.termination == "final time"


class TestBasicDFN(BaseBasicModelTest):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pybamm.sodium_ion.BasicDFN()
