#
# Tests for external submodels
#
import pybamm
import unittest
import numpy as np


class TestExternalCC(unittest.TestCase):

    model_options = {
        "current collector": "potential pair",
        "dimensionality": 1,
        "external submodels": ["current collector"],
    }
    model = pybamm.lithium_ion.SPM(model_options)
    var_pts = model.default_var_pts
    var_pts.update({pybamm.standard_spatial_vars.y: 5})
    sim = pybamm.Simulation(model)

    # provide phi_s_n and i_cc

    # return phi_s_p


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
