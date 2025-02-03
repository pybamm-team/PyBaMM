#
# Test that the model works with an example parameter set
#
import pybamm
import numpy as np
import tests


class TestSplitOCVR:
    def test_basic_processing(self):
        # example parameters
        qp0 = 8.73231852
        qn0 = 5.82761507
        theta0_n = 0.9013973983641687 * 0.9
        theta0_p = 0.5142305254580026 * 0.83

        # OCV functions
        def Un(theta_n):
            Un = (
                0.1493
                + 0.8493 * np.exp(-61.79 * theta_n)
                + 0.3824 * np.exp(-665.8 * theta_n)
                - np.exp(39.42 * theta_n - 41.92)
                - 0.03131 * np.arctan(25.59 * theta_n - 4.099)
                - 0.009434 * np.arctan(32.49 * theta_n - 15.74)
            )
            return Un

        def Up(theta_p):
            Up = (
                -10.72 * theta_p**4
                + 23.88 * theta_p**3
                - 16.77 * theta_p**2
                + 2.595 * theta_p
                + 4.563
            )
            return Up

        pars = pybamm.ParameterValues(
            {
                "Positive electrode capacity [A.h]": qp0,
                "Ohmic resistance [Ohm]": 0.001,
                "Negative electrode initial stoichiometry": theta0_n,
                "Lower voltage cut-off [V]": 2.8,
                "Positive electrode initial stoichiometry": theta0_p,
                "Upper voltage cut-off [V]": 4.2,
                "Negative electrode capacity [A.h]": qn0,
                "Current function [A]": 5,
                "Positive electrode OCP [V]": Up,
                "Negative electrode OCP [V]": Un,
                "Nominal cell capacity [A.h]": 5,
            }
        )
        model = pybamm.lithium_ion.SplitOCVR()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(param=pars)
