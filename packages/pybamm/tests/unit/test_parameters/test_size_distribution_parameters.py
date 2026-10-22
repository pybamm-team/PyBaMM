#
# Tests particle size distribution parameters are loaded into a parameter set
# and give expected values
#
import re

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
            }
        )
        values_composite = pybamm.get_size_distribution_parameters(
            params_composite,
            composite="negative",
        )
        assert "Primary: Negative maximum particle radius [m]" in values_composite
        param = pybamm.LithiumIonParameters({"particle phases": ("2", "1")})
        R_test_n_prim = pybamm.Scalar(1e-6).with_domains(
            {"primary": ["negative primary particle size"]}
        )
        R_test_n_sec = pybamm.Scalar(1e-6).with_domains(
            {"primary": ["negative secondary particle size"]}
        )
        values_composite.evaluate(param.n.prim.f_a_dist(R_test_n_prim))
        values_composite.evaluate(param.n.sec.f_a_dist(R_test_n_sec))
        params_composite = pybamm.ParameterValues("Chen2020_composite")
        params_composite.update(
            {
                "Primary: Positive particle radius [m]": 1e-6,
                "Secondary: Positive particle radius [m]": 1e-6,
                "Negative particle radius [m]": 1e-6,
            }
        )
        values_composite = pybamm.get_size_distribution_parameters(
            params_composite,
            composite="positive",
        )
        assert "Primary: Positive maximum particle radius [m]" in values_composite
        param = pybamm.LithiumIonParameters({"particle phases": ("1", "2")})
        R_test_p_prim = pybamm.Scalar(1e-6).with_domains(
            {"primary": ["positive primary particle size"]}
        )
        R_test_p_sec = pybamm.Scalar(1e-6).with_domains(
            {"primary": ["positive secondary particle size"]}
        )
        values_composite.evaluate(param.p.prim.f_a_dist(R_test_p_prim))
        values_composite.evaluate(param.p.sec.f_a_dist(R_test_p_sec))
        params_composite = pybamm.ParameterValues("Chen2020_composite")
        params_composite.update(
            {
                "Primary: Positive particle radius [m]": 1e-6,
                "Secondary: Positive particle radius [m]": 1e-6,
                "Negative particle radius [m]": 1e-6,
            }
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
            }
        )
        values_composite = pybamm.get_size_distribution_parameters(
            params_composite,
        )
        assert "Primary: Negative maximum particle radius [m]" not in values_composite
        assert "Primary: Positive maximum particle radius [m]" not in values_composite

    @pytest.mark.parametrize("composite", ["neither", "negative"])
    def test_distribution_follows_mean_radius_parameter(self, composite):
        if composite == "neither":
            values = pybamm.lithium_ion.BaseModel().default_parameter_values
            prefix = ""
        else:
            values = pybamm.ParameterValues("Chen2020_composite")
            values.update(
                {
                    "Primary: Positive particle radius [m]": 1e-6,
                    "Secondary: Positive particle radius [m]": 1e-6,
                    "Negative particle radius [m]": 1e-6,
                }
            )
            prefix = "Primary: "
        values = pybamm.get_size_distribution_parameters(values, composite=composite)
        radius = f"{prefix}Negative particle radius [m]"
        r_max = pybamm.Parameter(f"{prefix}Negative maximum particle radius [m]")

        # The support is truncated at 2.5 means, so the mean is ~0.1% low
        def distribution_mean(values):
            radii = np.linspace(1e-12, values.evaluate(r_max), 20001)
            f_a_dist = values.evaluate(
                pybamm.FunctionParameter(
                    f"{prefix}Negative area-weighted particle-size distribution [m-1]",
                    {"R": pybamm.Vector(radii)},
                )
            ).ravel()
            return np.trapezoid(radii * f_a_dist, radii) / np.trapezoid(f_a_dist, radii)

        r_mean = values.evaluate(pybamm.Parameter(radius))
        assert values.evaluate(r_max) == pytest.approx(2.5 * r_mean)
        assert distribution_mean(values) == pytest.approx(r_mean, rel=2e-3)

        # Changing the radius after building the distribution moves it too
        values.update({radius: 3 * r_mean})
        assert values.evaluate(r_max) == pytest.approx(7.5 * r_mean)
        assert distribution_mean(values) == pytest.approx(3 * r_mean, rel=2e-3)

        # An input radius stays symbolic in the processed parameters
        values.update({radius: "[input]"})
        processed = values.process_symbol(r_max)
        assert processed.evaluate(inputs={radius: 2e-6}) == pytest.approx(5e-6)

    def test_missing_mean_radius_raises(self):
        values = pybamm.ParameterValues("Chen2020")
        with pytest.raises(
            KeyError,
            match=re.escape(
                "get_size_distribution_parameters needs 'Primary: Negative particle "
                "radius [m]' in the parameter values."
            ),
        ):
            pybamm.get_size_distribution_parameters(values, composite="negative")
