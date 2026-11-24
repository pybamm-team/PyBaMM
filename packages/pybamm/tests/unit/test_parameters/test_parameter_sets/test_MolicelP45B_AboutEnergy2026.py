#
# Tests for Molicel P45B About:Energy (2026) parameter set
#

import pytest

import pybamm
from pybamm.input.parameters.lithium_ion.MolicelP45B_AboutEnergy2026 import (
    _build_electrolyte_functions,
)


class TestMolicelP45B_AboutEnergy2026:
    def test_functions(self):
        param = pybamm.ParameterValues("MolicelP45B_AboutEnergy2026")
        T = pybamm.Scalar(318.15)

        fun_test = {
            "Electrolyte conductivity [S.m-1]": ([1000, T], 1.21043),
            "Electrolyte diffusivity [m2.s-1]": ([1000, T], 3.95956e-10),
            "Cation transference number": ([1000, T], 0.38300),
            "Thermodynamic factor": ([1000, T], 2.55297),
            "Primary: Negative electrode exchange-current density [A.m-2]": (
                [1000, 15263, 30526, T],
                5.32167,
            ),
            "Secondary: Negative electrode exchange-current density [A.m-2]": (
                [1000, 62665, 125330, T],
                39.4168,
            ),
        }

        for name, value in fun_test.items():
            assert param.evaluate(param[name](*value[0])) == pytest.approx(
                value[1], rel=1e-5
            )

    def test_user_defined_parameters_converted(self):
        param = pybamm.ParameterValues("MolicelP45B_AboutEnergy2026")
        assert not [
            key
            for key in param
            if "func_type" in key
            or "poly." in key
            or "pre-multiplier" in key
            or ("Negative electrode reaction rate constant" in key)
            or key.startswith(("Graphite: ", "Silicon Additive: "))
        ]
        assert param["Contact resistance [Ohm]"] == pytest.approx(0.00241)

    def test_unknown_func_type(self):
        values = {"Thermodynamic factor func_type unknown": 0.0}
        with pytest.raises(pybamm.ModelError, match=r"Unknown func_type 'unknown'"):
            _build_electrolyte_functions(values)
