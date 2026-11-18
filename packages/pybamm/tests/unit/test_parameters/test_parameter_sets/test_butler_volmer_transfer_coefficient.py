#
# Tests that parameter sets provide the Butler-Volmer transfer coefficient read by
# asymmetric Butler-Volmer kinetics
#
import numpy as np
import pytest

import pybamm

PARAMETER_SETS = [
    name
    for name in sorted(pybamm.parameter_sets)
    if pybamm.parameter_sets[name]["chemistry"] in ("lithium_ion", "sodium_ion")
    and not name.startswith("MSMR")
]


class TestButlerVolmerTransferCoefficient:
    @pytest.mark.parametrize("parameter_set", PARAMETER_SETS)
    def test_parameter_set_supports_asymmetric_butler_volmer(self, parameter_set):
        parameter_values = pybamm.ParameterValues(parameter_set)
        for domain in ["Negative", "Positive"]:
            assert f"{domain} electrode charge transfer coefficient" not in (
                parameter_values
            )

        options = {"intercalation kinetics": "asymmetric Butler-Volmer"}
        # Half-cell sets have no porous negative electrode
        if "Negative electrode porosity" not in parameter_values:
            options["working electrode"] = "positive"
        phases = tuple(
            "2"
            if f"Primary: {domain} electrode Butler-Volmer transfer coefficient"
            in parameter_values
            else "1"
            for domain in ["Negative", "Positive"]
        )
        if phases != ("1", "1"):
            options["particle phases"] = phases
        parameter_values.process_model(pybamm.lithium_ion.SPM(options))

    def test_half_transfer_coefficient_matches_symmetric_kinetics(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        t = np.linspace(0, 600, 31)
        voltages = []
        for kinetics in ["symmetric Butler-Volmer", "asymmetric Butler-Volmer"]:
            model = pybamm.lithium_ion.SPM({"intercalation kinetics": kinetics})
            sim = pybamm.Simulation(model, parameter_values=parameter_values)
            solution = sim.solve([0, 600])
            voltages.append(solution["Voltage [V]"](t))
        np.testing.assert_allclose(voltages[1], voltages[0], rtol=1e-10)
