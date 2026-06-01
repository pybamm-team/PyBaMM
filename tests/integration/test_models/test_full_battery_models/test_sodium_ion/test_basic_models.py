#
# Test basic model classes
#
import pytest

import pybamm


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


class TestStoichiometryConductivity:
    def test_basic_dfn(self):
        # surface stoichiometry feeds the electrode conductivity in the sodium-ion
        # basic DFN, so a stoichiometry-dependent conductivity changes the solution
        def solve(values):
            sim = pybamm.Simulation(
                pybamm.sodium_ion.BasicDFN(), parameter_values=values
            )
            sim.solve([0, 600])
            return sim.solution["Voltage [V]"].entries[-1]

        values = pybamm.sodium_ion.BasicDFN().default_parameter_values
        sigma_n = values["Negative electrode conductivity [S.m-1]"]
        sigma_p = values["Positive electrode conductivity [S.m-1]"]
        values_sto = values.copy()
        values_sto.update(
            {
                "Negative electrode conductivity [S.m-1]": lambda sto, T: (
                    0.1 * sigma_n * (0.1 + sto)
                ),
                "Positive electrode conductivity [S.m-1]": lambda sto, T: (
                    0.1 * sigma_p * (0.1 + sto)
                ),
            }
        )
        assert abs(solve(values) - solve(values_sto)) > 1e-4
