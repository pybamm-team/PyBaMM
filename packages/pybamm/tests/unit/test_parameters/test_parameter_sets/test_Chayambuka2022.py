#
# Tests for Chayambuka et al (2022) parameter set
#

import pytest

import pybamm


class TestChayambuka2022:
    def test_functions(self):
        param = pybamm.ParameterValues("Chayambuka2022")
        sto = pybamm.Scalar(0.5)
        T = pybamm.Scalar(298.15)
        c_e = 1000
        c_n_max = 14540
        c_p_max = 15320

        fun_test = {
            # Negative electrode
            "Negative particle diffusivity [m2.s-1]": ([sto, T], 1.8761e-15),
            "Negative electrode OCP [V]": ([sto], 0.0859),
            "Negative electrode exchange-current density [A.m-2]": (
                [c_e, sto * c_n_max, c_n_max, T],
                0.0202,
            ),
            # Positive electrode
            "Positive particle diffusivity [m2.s-1]": ([sto, T], 8.3317e-17),
            "Positive electrode OCP [V]": ([sto], 4.1482),
            "Positive electrode exchange-current density [A.m-2]": (
                [c_e, sto * c_p_max, c_p_max, T],
                0.0036,
            ),
            # Electrolyte
            "Electrolyte diffusivity [m2.s-1]": ([c_e, T], 2.5061e-10),
            "Electrolyte conductivity [S.m-1]": ([c_e, T], 0.8830),
        }

        for name, value in fun_test.items():
            assert param.evaluate(param[name](*value[0])) == pytest.approx(
                value[1], abs=0.0001
            )

    def test_positive_particle_diffusivity(self):
        # The NVPF data are tabulated against the solid concentration, so the
        # lookup must scale the stoichiometry by the positive maximum concentration
        param = pybamm.ParameterValues("Chayambuka2022")
        T = pybamm.Scalar(298.15)
        for sto, expected in [(0.3, 5.6673e-17), (0.5, 8.3317e-17), (0.9, 3.5627e-17)]:
            D_p = param["Positive particle diffusivity [m2.s-1]"](pybamm.Scalar(sto), T)
            assert param.evaluate(D_p) == pytest.approx(expected, rel=1e-4, abs=0)
