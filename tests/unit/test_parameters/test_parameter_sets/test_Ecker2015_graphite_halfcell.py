#
# Tests for O'Kane (2022) parameter set
#
import pytest
import pybamm


class TestEcker2015_graphite_halfcell:
    def test_functions(self):
        param = pybamm.ParameterValues("Ecker2015_graphite_halfcell")
        sto = pybamm.Scalar(0.5)
        T = pybamm.Scalar(298.15)

        fun_test = {
            # Lithium plating
            "Exchange-current density for plating [A.m-2]": ([1e3, 1e4, T], 9.6485e-3),
            "Exchange-current density for stripping [A.m-2]": (
                [1e3, 1e4, T],
                9.6485e-2,
            ),
            "Dead lithium decay rate [s-1]": ([1e-8], 5e-7),
            # Positive electrode
            "Positive particle diffusivity [m2.s-1]": ([sto, T], 1.219e-14),
            "Positive electrode exchange-current density [A.m-2]": (
                [1000, 15960, 31920, T],
                6.2517,
            ),
            "Positive electrode OCP [V]": ([sto], 0.124),
            # Electrolyte
            "Electrolyte diffusivity [m2.s-1]": ([1000, T], 2.593e-10),
            "Electrolyte conductivity [S.m-1]": ([1000, T], 0.9738),
        }

        for name, value in fun_test.items():
            assert param.evaluate(param[name](*value[0])) == pytest.approx(
                value[1], abs=0.0001
            )
