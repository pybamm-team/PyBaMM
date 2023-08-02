#
# Tests particle size distribution parameters are loaded into a parameter set
# and give expected values
#
import pybamm
import unittest
import numpy as np
from tests import TestCase


class TestSizeDistributionParameters(TestCase):
    def test_parameter_values(self):
        values = pybamm.lithium_ion.BaseModel().default_parameter_values
        param = pybamm.LithiumIonParameters()

        # add distribution parameter values for negative electrode
        values = pybamm.get_size_distribution_parameters(values, electrode="negative")

        # check positive parameters aren't there yet
        with self.assertRaises(KeyError):
            values["Positive maximum particle radius [m]"]

        # now add distribution parameter values for positive electrode
        values = pybamm.get_size_distribution_parameters(values, electrode="positive")

        # check parameters

        # min and max radii
        np.testing.assert_almost_equal(values.evaluate(param.n.prim.R_min), 0.0, 3)
        np.testing.assert_almost_equal(values.evaluate(param.p.prim.R_min), 0.0, 3)
        np.testing.assert_almost_equal(values.evaluate(param.n.prim.R_max), 2.5e-5, 3)
        np.testing.assert_almost_equal(values.evaluate(param.p.prim.R_max), 2.5e-5, 3)

        # standard deviations
        np.testing.assert_almost_equal(values.evaluate(param.n.prim.sd_a), 3e-6, 3)
        np.testing.assert_almost_equal(values.evaluate(param.p.prim.sd_a), 3e-6, 3)

        # check function parameters (size distributions) evaluate
        R_test = pybamm.Scalar(1.0)
        values.evaluate(param.n.prim.f_a_dist(R_test))
        values.evaluate(param.p.prim.f_a_dist(R_test))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
