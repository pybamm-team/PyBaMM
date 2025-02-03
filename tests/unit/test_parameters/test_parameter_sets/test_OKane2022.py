#
# Tests for O'Kane (2022) parameter set
#

import pytest
import pybamm


class TestOKane2022:
    def test_functions(self):
        param = pybamm.ParameterValues("OKane2022")
        sto = pybamm.Scalar(0.9)
        T = pybamm.Scalar(298.15)

        fun_test = {
            # Lithium plating
            "Exchange-current density for plating [A.m-2]": ([1e3, 1e4, T], 9.6485e-2),
            "Exchange-current density for stripping [A.m-2]": (
                [1e3, 1e4, T],
                9.6485e-1,
            ),
            "Dead lithium decay rate [s-1]": ([1e-8], 5e-7),
            # Negative electrode
            "Negative particle diffusivity [m2.s-1]": ([sto, T], 3.3e-14),
            "Negative electrode exchange-current density [A.m-2]": (
                [1000, 16566.5, 33133, T],
                0.33947,
            ),
            "Negative electrode cracking rate": ([T], 3.9e-20),
            "Negative electrode volume change": ([sto], 0.0897),
            # Positive electrode
            "Positive particle diffusivity [m2.s-1]": ([sto, T], 4e-15),
            "Positive electrode exchange-current density [A.m-2]": (
                [1000, 31552, 63104, T],
                3.4123,
            ),
            "Positive electrode OCP [V]": ([sto], 3.5682),
            "Positive electrode cracking rate": ([T], 3.9e-20),
            "Positive electrode volume change": ([sto], 0.70992),
        }

        for name, value in fun_test.items():
            assert param.evaluate(param[name](*value[0])) == pytest.approx(
                value[1], abs=0.0001
            )
