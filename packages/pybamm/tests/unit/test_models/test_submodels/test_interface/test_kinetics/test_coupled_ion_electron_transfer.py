import numpy as np
import pytest
from scipy.integrate import quad

import pybamm

C_MAX = 30000.0
C_E_INIT = 1000.0
T = 298.15
REORGANIZATION_ENERGY = 0.2
RATE_CONSTANT = 10.0


def mhc_rate(eta):
    """Marcus-Hush-Chidsey rate integral, with eta and lambda in units of kT/e."""
    F_RT = pybamm.constants.F.value / (pybamm.constants.R.value * T)
    lambda_T = F_RT * REORGANIZATION_ENERGY
    return quad(
        lambda x: np.exp(
            -((x - lambda_T + eta) ** 2) / (4 * lambda_T) - np.logaddexp(0, x)
        ),
        -np.inf,
        np.inf,
    )[0]


def expected_current(eta_r, stoichiometry, relative_c_e):
    F_RT = pybamm.constants.F.value / (pybamm.constants.R.value * T)
    eta_f = F_RT * eta_r + np.log(relative_c_e * (1 - stoichiometry) / stoichiometry)
    oxidation = stoichiometry * mhc_rate(eta_f)
    reduction = relative_c_e * (1 - stoichiometry) * mhc_rate(-eta_f)
    return RATE_CONSTANT * (oxidation - reduction)


class TestCoupledIonElectronTransfer:
    def get_current(self, eta_r, stoichiometry, relative_c_e):
        param = pybamm.LithiumIonParameters()
        options = pybamm.BatteryModelOptions(
            {"intercalation kinetics": "coupled ion-electron transfer"}
        )
        submodel = pybamm.kinetics.CoupledIonElectronTransfer(
            param, "negative", "lithium-ion main", options, "primary"
        )
        variables = {
            "Negative electrolyte concentration [mol.m-3]": pybamm.Scalar(
                relative_c_e * C_E_INIT
            ),
            "Negative electrode temperature [K]": pybamm.Scalar(T),
            "Negative particle surface concentration [mol.m-3]": pybamm.Scalar(
                stoichiometry * C_MAX
            ),
        }
        j0 = submodel._get_exchange_current_density(variables)
        j = submodel._get_kinetics(
            j0, pybamm.Scalar(1), pybamm.Scalar(eta_r), pybamm.Scalar(T), 1
        )
        parameter_values = pybamm.ParameterValues(
            {
                "Maximum concentration in negative electrode [mol.m-3]": C_MAX,
                "Initial concentration in electrolyte [mol.m-3]": C_E_INIT,
                "Negative electrode reorganization energy [eV]": (
                    REORGANIZATION_ENERGY
                ),
                "Negative electrode coupled ion-electron transfer rate constant "
                "[A.m-2]": RATE_CONSTANT,
            }
        )
        return (
            parameter_values.process_symbol(j0).evaluate().item(),
            parameter_values.process_symbol(j).evaluate().item(),
        )

    @pytest.mark.parametrize("stoichiometry", [0.05, 0.5, 0.95])
    @pytest.mark.parametrize("relative_c_e", [0.5, 1.0, 2.0])
    def test_no_current_at_equilibrium(self, stoichiometry, relative_c_e):
        _, j = self.get_current(0.0, stoichiometry, relative_c_e)
        assert j == pytest.approx(0, abs=1e-12)

    @pytest.mark.parametrize("eta_r", [-0.4, -0.2, -0.02, 0.02, 0.2, 0.4])
    @pytest.mark.parametrize("stoichiometry", [0.1, 0.6])
    def test_current_matches_marcus_integral(self, eta_r, stoichiometry):
        # The Zeng asymptotic form is within 5% of the integral at this lambda
        _, j = self.get_current(eta_r, stoichiometry, 1.2)
        np.testing.assert_allclose(
            j, expected_current(eta_r, stoichiometry, 1.2), rtol=0.05
        )
        assert np.sign(j) == np.sign(eta_r)

    def test_current_saturates(self):
        F_RT = pybamm.constants.F.value / (pybamm.constants.R.value * T)
        plateau = 2 * RATE_CONSTANT * np.sqrt(np.pi * F_RT * REORGANIZATION_ENERGY)
        stoichiometry, relative_c_e = 0.3, 0.8
        _, j_oxidation = self.get_current(5.0, stoichiometry, relative_c_e)
        _, j_reduction = self.get_current(-5.0, stoichiometry, relative_c_e)
        np.testing.assert_allclose(j_oxidation, plateau * stoichiometry, rtol=1e-6)
        np.testing.assert_allclose(
            j_reduction, -plateau * relative_c_e * (1 - stoichiometry), rtol=1e-6
        )

    @pytest.mark.parametrize("stoichiometry", [0.2, 0.8])
    def test_exchange_current_is_linear_response(self, stoichiometry):
        eta_r = 1e-6
        j0, j = self.get_current(eta_r, stoichiometry, 1.0)
        F_RT = pybamm.constants.F.value / (pybamm.constants.R.value * T)
        np.testing.assert_allclose(j, j0 * F_RT * eta_r, rtol=1e-4)

    def test_lead_acid_not_supported(self):
        with pytest.raises(pybamm.OptionError, match=r"lithium-ion intercalation"):
            pybamm.kinetics.CoupledIonElectronTransfer(
                pybamm.LeadAcidParameters(),
                "negative",
                "lead-acid main",
                pybamm.BatteryModelOptions({}),
            )

    def test_current_sigmoid_not_supported(self):
        options = pybamm.BatteryModelOptions(
            {
                "intercalation kinetics": "coupled ion-electron transfer",
                "exchange-current density": "current sigmoid",
            }
        )
        with pytest.raises(pybamm.OptionError, match=r"exchange-current density"):
            pybamm.kinetics.CoupledIonElectronTransfer(
                pybamm.LithiumIonParameters(), "negative", "lithium-ion main", options
            )

    def test_citations(self):
        citations = pybamm.citations
        citations._reset()
        pybamm.kinetics.CoupledIonElectronTransfer(
            pybamm.LithiumIonParameters(),
            "negative",
            "lithium-ion main",
            pybamm.BatteryModelOptions({}),
        )
        assert "Fraggedakis2021" in citations._papers_to_cite
        assert "Zeng2014" in citations._papers_to_cite
        assert "Sripad2020" not in citations._papers_to_cite
