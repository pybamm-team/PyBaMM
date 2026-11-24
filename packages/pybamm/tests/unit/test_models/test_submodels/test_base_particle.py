#
# Tests for the base particle submodel
#
import pytest

import pybamm


def _has_stress_factor(model, variable_name):
    rhs = next(eq for var, eq in model.rhs.items() if var.name == variable_name)
    return any(
        "partial molar volume" in symbol.name
        for symbol in rhs.pre_order()
        if isinstance(symbol, (pybamm.Parameter, pybamm.FunctionParameter))
    )


class TestBaseParticle:
    @pytest.mark.parametrize(
        "options, expected",
        [
            # stress-induced diffusion defaults per phase from particle mechanics
            (
                {
                    "particle phases": ("2", "1"),
                    "particle mechanics": (("swelling and cracking", "none"), "none"),
                },
                (True, False),
            ),
            (
                {
                    "particle phases": ("2", "1"),
                    "particle mechanics": (("swelling only", "swelling only"), "none"),
                    "stress-induced diffusion": (("false", "true"), "false"),
                },
                (False, True),
            ),
        ],
    )
    def test_stress_induced_diffusion_per_phase(self, options, expected):
        model = pybamm.lithium_ion.DFN(options)
        assert (
            _has_stress_factor(
                model, "Negative primary particle concentration [mol.m-3]"
            ),
            _has_stress_factor(
                model, "Negative secondary particle concentration [mol.m-3]"
            ),
        ) == expected
        assert not _has_stress_factor(
            model, "Positive particle concentration [mol.m-3]"
        )
