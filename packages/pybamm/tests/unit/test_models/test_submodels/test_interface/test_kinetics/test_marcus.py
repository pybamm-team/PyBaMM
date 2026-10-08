import numpy as np
import pytest

import pybamm


class TestMarcus:
    @pytest.mark.parametrize("eta_r", [-0.05, -0.01, 0.01, 0.05])
    def test_current_follows_marcus_rate_law(self, eta_r):
        param = pybamm.LithiumIonParameters()
        options = pybamm.BatteryModelOptions({"intercalation kinetics": "Marcus"})
        submodel = pybamm.kinetics.Marcus(
            param, "negative", "lithium-ion main", options, "primary"
        )
        j0, T, reorganization_energy = 2.0, 298.15, 0.2
        j = submodel._get_kinetics(
            pybamm.Scalar(j0),
            pybamm.Scalar(1),
            pybamm.Scalar(eta_r),
            pybamm.Scalar(T),
            pybamm.Scalar(1),
        )
        parameter_values = pybamm.ParameterValues(
            {"Negative electrode reorganization energy [eV]": reorganization_energy}
        )
        j = parameter_values.process_symbol(j).evaluate()

        # Oxidation goes with (lambda - e*eta)^2 and reduction with (lambda + e*eta)^2
        F_RT = pybamm.constants.F.value / (pybamm.constants.R.value * T)
        lambda_T = F_RT * reorganization_energy
        eta_T = F_RT * eta_r
        expected = j0 * (
            np.exp(-((lambda_T - eta_T) ** 2) / (4 * lambda_T))
            - np.exp(-((lambda_T + eta_T) ** 2) / (4 * lambda_T))
        )
        np.testing.assert_allclose(j, expected, rtol=1e-12)
        assert np.sign(j) == np.sign(eta_r)
