#
# Tests particle size distribution parameters are loaded into a parameter set
# and give expected values
#
import pytest
import pybamm
import numpy as np


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
