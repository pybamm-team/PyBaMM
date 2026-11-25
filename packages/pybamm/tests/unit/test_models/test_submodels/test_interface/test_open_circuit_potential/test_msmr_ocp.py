import pytest

import pybamm


class TestMSMROpenCircuitPotential:
    def test_only_main_reaction(self):
        options = pybamm.BatteryModelOptions({})
        submodel = pybamm.open_circuit_potential.MSMROpenCircuitPotential(
            pybamm.LithiumIonParameters(options),
            "negative",
            "lithium metal plating",
            options,
            "primary",
        )
        with pytest.raises(pybamm.OptionError, match=r"not defined for 'lithium"):
            submodel.build()
