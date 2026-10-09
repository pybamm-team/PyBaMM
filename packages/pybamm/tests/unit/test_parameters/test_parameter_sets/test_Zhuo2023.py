#
# Tests for the Zhuo (2023) parameter set
#

import numpy as np
import pytest

import pybamm


class TestZhuo2023:
    def test_functions(self):
        param = pybamm.ParameterValues("Zhuo2023")
        T = pybamm.Scalar(298.15)
        F = pybamm.constants.F.value

        fun_test = {
            "Negative particle diffusivity [m2.s-1]": ([0.5, T], 1e-14),
            "Positive particle diffusivity [m2.s-1]": ([0.5, T], 1e-14),
            "Positive core diffusivity [m2.s-1]": ([0.5, T], 1e-14),
            "Positive shell diffusivity [m2.s-1]": ([0.5, T], 1e-15),
            "Negative electrode exchange-current density [A.m-2]": (
                [1000, 17128.5, 34257, T],
                1e-11 * F * np.sqrt(1000) * 17128.5,
            ),
            "Positive electrode exchange-current density [A.m-2]": (
                [1000, 24670, 49340, T],
                3.2e-11 * F * np.sqrt(1000) * 24670,
            ),
            "Electrolyte conductivity [S.m-1]": ([1000, T], 1.1046),
            "Electrolyte diffusivity [m2.s-1]": ([1000, T], 5.34e-10 * np.exp(-0.65)),
            "Initial oxygen concentration in positive shell [mol.m-3]": ([0.3], 0),
            "Negative electrode OCP [V]": ([0.5], 0.078784),
            "Negative electrode OCP entropic change [V.K-1]": (
                [0.500427],
                -0.029047e-3,
            ),
            "Positive electrode OCP entropic change [V.K-1]": ([0.4997], -0.007855e-3),
        }

        for name, (args, expected) in fun_test.items():
            value = np.asarray(param.evaluate(param[name](*args))).item()
            assert value == pytest.approx(expected, rel=1e-10, abs=1e-20)

    def test_positive_ocp_includes_asymptote(self):
        param = pybamm.ParameterValues("Zhuo2023")
        sto = 0.499685
        value = param.evaluate(param["Positive electrode OCP [V]"](pybamm.Scalar(sto)))
        expected = 3.944915 + 1e-6 * (1 / sto + 1 / (sto - 1))
        assert np.asarray(value).item() == pytest.approx(expected, rel=1e-12)
