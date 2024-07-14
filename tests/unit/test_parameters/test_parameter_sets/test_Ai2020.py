#
# Tests for Ai (2020) Enertech parameter set loads
#
import pytest
import pybamm


class TestAi2020:
    def test_functions(self):
        param = pybamm.ParameterValues("Ai2020")
        sto = pybamm.Scalar(0.5)
        T = pybamm.Scalar(298.15)

        c_p_max = param["Maximum concentration in positive electrode [mol.m-3]"]
        c_n_max = param["Maximum concentration in negative electrode [mol.m-3]"]
        fun_test = {
            # Positive electrode
            "Positive electrode cracking rate": ([T], 3.9e-20),
            "Positive particle diffusivity [m2.s-1]": ([sto, T], 5.387e-15),
            "Positive electrode exchange-current density [A.m-2]": (
                [1e3, 1e4, c_p_max, T],
                0.6098,
            ),
            "Positive electrode OCP entropic change [V.K-1]": (
                [sto, c_p_max],
                -2.1373e-4,
            ),
            "Positive electrode volume change": ([sto, c_p_max], -1.8179e-2),
            # Negative electrode
            "Negative electrode cracking rate": ([T], 3.9e-20),
            "Negative particle diffusivity [m2.s-1]": ([sto, T], 3.9e-14),
            "Negative electrode exchange-current density [A.m-2]": (
                [1e3, 1e4, c_n_max, T],
                0.4172,
            ),
            "Negative electrode OCP entropic change [V.K-1]": (
                [sto, c_n_max],
                -1.1033e-4,
            ),
            "Negative electrode volume change": ([sto, c_n_max], 5.1921e-2),
        }

        for name, value in fun_test.items():
            assert param.evaluate(param[name](*value[0])) == pytest.approx(
                value[1], abs=0.0001
            )
