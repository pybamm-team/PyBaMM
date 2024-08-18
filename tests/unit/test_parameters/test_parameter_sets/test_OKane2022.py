import pytest
import pybamm


class TestOKane2022:
    @pytest.mark.parametrize(
        "name, inputs, expected_output",
        [
            # Lithium plating
            (
                "Exchange-current density for plating [A.m-2]",
                [1e3, 1e4, pybamm.Scalar(298.15)],
                9.6485e-2,
            ),
            (
                "Exchange-current density for stripping [A.m-2]",
                [1e3, 1e4, pybamm.Scalar(298.15)],
                9.6485e-1,
            ),
            ("Dead lithium decay rate [s-1]", [1e-8], 5e-7),
            # Negative electrode
            (
                "Negative particle diffusivity [m2.s-1]",
                [pybamm.Scalar(0.9), pybamm.Scalar(298.15)],
                3.3e-14,
            ),
            (
                "Negative electrode exchange-current density [A.m-2]",
                [1000, 16566.5, 33133, pybamm.Scalar(298.15)],
                0.33947,
            ),
            ("Negative electrode cracking rate", [pybamm.Scalar(298.15)], 3.9e-20),
            ("Negative electrode volume change", [pybamm.Scalar(0.9), 33133], 0.0897),
            # Positive electrode
            (
                "Positive particle diffusivity [m2.s-1]",
                [pybamm.Scalar(0.9), pybamm.Scalar(298.15)],
                4e-15,
            ),
            (
                "Positive electrode exchange-current density [A.m-2]",
                [1000, 31552, 63104, pybamm.Scalar(298.15)],
                3.4123,
            ),
            ("Positive electrode OCP [V]", [pybamm.Scalar(0.9)], 3.5682),
            ("Positive electrode cracking rate", [pybamm.Scalar(298.15)], 3.9e-20),
            ("Positive electrode volume change", [pybamm.Scalar(0.9), 63104], 0.70992),
        ],
    )
    def test_functions(self, name, inputs, expected_output):
        param = pybamm.ParameterValues("OKane2022")
        assert param.evaluate(param[name](*inputs)) == pytest.approx(
            expected_output, abs=0.0001
        )
