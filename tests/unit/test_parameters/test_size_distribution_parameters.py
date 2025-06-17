#
# Tests particle size distribution parameters are loaded into a parameter set
# and give expected values
#
import numpy as np
import pytest

import pybamm


class TestSizeDistributionParameters:
    def test_parameter_values(self):
        values = pybamm.lithium_ion.BaseModel().default_parameter_values
        param = pybamm.LithiumIonParameters()

        # add distribution parameter values for positive electrode
        values = pybamm.get_size_distribution_parameters(
            values,
            working_electrode="positive",
        )

        # check negative parameters aren't there yet
        with pytest.raises(KeyError):
            values["Negative maximum particle radius [m]"]

        # now add distribution parameter values for negative electrode
        values = pybamm.get_size_distribution_parameters(
            values,
            working_electrode="both",
        )

        # check parameters

        # min and max radii
        np.testing.assert_almost_equal(values.evaluate(param.n.prim.R_min), 0.0, 3)
        np.testing.assert_almost_equal(values.evaluate(param.p.prim.R_min), 0.0, 3)
        np.testing.assert_almost_equal(values.evaluate(param.n.prim.R_max), 2.5e-5, 3)
        np.testing.assert_almost_equal(values.evaluate(param.p.prim.R_max), 2.5e-5, 3)

        # check function parameters (size distributions) evaluate
        R_test = pybamm.Scalar(1.0)
        values.evaluate(param.n.prim.f_a_dist(R_test))
        values.evaluate(param.p.prim.f_a_dist(R_test))

        # check that the composite parameters works as well
        params_composite = pybamm.ParameterValues("Chen2020_composite")
        params_composite.update(
            {
                "Primary: Positive particle radius [m]": 1e-6,
                "Secondary: Positive particle radius [m]": 1e-6,
                "Negative particle radius [m]": 1e-6,
            },
            check_already_exists=False,
        )
        values_composite = pybamm.get_size_distribution_parameters(
            params_composite,
            composite="negative",
        )
        assert "Primary: Negative maximum particle radius [m]" in values_composite
        param = pybamm.LithiumIonParameters({"particle phases": ("2", "1")})
        R_test_n_prim = pybamm.Scalar(1e-6)
        R_test_n_prim.domains = {"primary": ["negative primary particle size"]}
        R_test_n_sec = pybamm.Scalar(1e-6)
        R_test_n_sec.domains = {"primary": ["negative secondary particle size"]}
        values_composite.evaluate(param.n.prim.f_a_dist(R_test_n_prim))
        values_composite.evaluate(param.n.sec.f_a_dist(R_test_n_sec))
        params_composite = pybamm.ParameterValues("Chen2020_composite")
        params_composite.update(
            {
                "Primary: Positive particle radius [m]": 1e-6,
                "Secondary: Positive particle radius [m]": 1e-6,
                "Negative particle radius [m]": 1e-6,
            },
            check_already_exists=False,
        )
        values_composite = pybamm.get_size_distribution_parameters(
            params_composite,
            composite="positive",
        )
        assert "Primary: Positive maximum particle radius [m]" in values_composite
        param = pybamm.LithiumIonParameters({"particle phases": ("1", "2")})
        R_test_p_prim = pybamm.Scalar(1e-6)
        R_test_p_prim.domains = {"primary": ["positive primary particle size"]}
        R_test_p_sec = pybamm.Scalar(1e-6)
        R_test_p_sec.domains = {"primary": ["positive secondary particle size"]}
        values_composite.evaluate(param.p.prim.f_a_dist(R_test_p_prim))
        values_composite.evaluate(param.p.sec.f_a_dist(R_test_p_sec))
        params_composite = pybamm.ParameterValues("Chen2020_composite")
        params_composite.update(
            {
                "Primary: Positive particle radius [m]": 1e-6,
                "Secondary: Positive particle radius [m]": 1e-6,
                "Negative particle radius [m]": 1e-6,
            },
            check_already_exists=False,
        )
        values_composite = pybamm.get_size_distribution_parameters(
            params_composite,
            composite="both",
        )
        assert "Primary: Negative maximum particle radius [m]" in values_composite
        assert "Primary: Positive maximum particle radius [m]" in values_composite
        params_composite = pybamm.ParameterValues("Chen2020_composite")
        params_composite.update(
            {
                "Primary: Positive particle radius [m]": 1e-6,
                "Secondary: Positive particle radius [m]": 1e-6,
                "Negative particle radius [m]": 1e-6,
            },
            check_already_exists=False,
        )
        values_composite = pybamm.get_size_distribution_parameters(
            params_composite,
        )
        assert "Primary: Negative maximum particle radius [m]" not in values_composite
        assert "Primary: Positive maximum particle radius [m]" not in values_composite
