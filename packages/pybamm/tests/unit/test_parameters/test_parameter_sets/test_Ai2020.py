#
# Tests for Ai (2020) Enertech parameter set loads
#
import numpy as np
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
                [sto],
                -2.1373e-4,
            ),
            "Positive electrode volume change": ([sto], -1.8179e-2),
            # Negative electrode
            "Negative electrode cracking rate": ([T], 3.9e-20),
            "Negative particle diffusivity [m2.s-1]": ([sto, T], 3.9e-14),
            "Negative electrode exchange-current density [A.m-2]": (
                [1e3, 1e4, c_n_max, T],
                0.4172,
            ),
            "Negative electrode OCP entropic change [V.K-1]": (
                [sto],
                -1.1033e-4,
            ),
            "Negative electrode volume change": ([sto], 5.1921e-2),
        }

        for name, value in fun_test.items():
            assert param.evaluate(param[name](*value[0])) == pytest.approx(
                value[1], abs=0.0001
            )

    def test_electrolyte_diffusivity(self):
        # Same Valoen and Reimers (2005) fit as Xu2019, which already converts to m2/s
        ai2020 = pybamm.ParameterValues("Ai2020")
        xu2019 = pybamm.ParameterValues("Xu2019")
        name = "Electrolyte diffusivity [m2.s-1]"
        for c_e, T in [(1000, 298.15), (500, 318.15), (1500, 278.15)]:
            inputs = [pybamm.Scalar(c_e), pybamm.Scalar(T)]
            np.testing.assert_allclose(
                ai2020.evaluate(ai2020[name](*inputs)),
                xu2019.evaluate(xu2019[name](*inputs)),
                rtol=1e-12,
            )
        D_e = ai2020.evaluate(ai2020[name](pybamm.Scalar(1000), pybamm.Scalar(298.15)))
        assert D_e == pytest.approx(3.2227e-10, rel=1e-4, abs=0)
