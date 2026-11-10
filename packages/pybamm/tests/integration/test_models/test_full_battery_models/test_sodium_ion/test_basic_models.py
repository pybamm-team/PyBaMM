#
# Test basic model classes
#
import numpy as np
import pytest

import pybamm


def sodium_per_area(model):
    """Sodium inventory per unit electrode area [mol.m-2]."""
    n = pybamm.Scalar(0)
    for domain, porosity, thickness in [
        (
            "Negative",
            "Negative electrode porosity",
            "Negative electrode thickness [m]",
        ),
        ("Separator", "Separator porosity", "Separator thickness [m]"),
        (
            "Positive",
            "Positive electrode porosity",
            "Positive electrode thickness [m]",
        ),
    ]:
        c_e = model.variables[f"{domain} electrolyte concentration [mol.m-3]"]
        n += pybamm.x_average(c_e * pybamm.Parameter(porosity)) * pybamm.Parameter(
            thickness
        )
    for c_s_name, eps_s, thickness in [
        (
            "Negative particle concentration [mol.m-3]",
            "Negative electrode active material volume fraction",
            "Negative electrode thickness [m]",
        ),
        (
            "Positive particle concentration [mol.m-3]",
            "Positive electrode active material volume fraction",
            "Positive electrode thickness [m]",
        ),
    ]:
        n += pybamm.x_average(
            pybamm.Parameter(eps_s) * pybamm.r_average(model.variables[c_s_name])
        ) * pybamm.Parameter(thickness)
    return n


class TestElectrolyteConservation:
    def test_basic_dfn_with_variable_transference_number(self):
        model = pybamm.sodium_ion.BasicDFN()
        model.variables["Sodium per area [mol.m-2]"] = sodium_per_area(model)
        parameter_values = model.default_parameter_values.copy()
        c_e_init = parameter_values["Initial concentration in electrolyte [mol.m-3]"]
        parameter_values["Cation transference number"] = lambda c_e, T: (
            0.45 + 1e-4 * (c_e - c_e_init)
        )
        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8),
        )
        sol = sim.solve([0, 1200], initial_soc=0.5)
        sodium = sol["Sodium per area [mol.m-2]"].entries
        np.testing.assert_allclose(sodium, sodium[0], rtol=1e-8)


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
