#
# Tests for the electrolyte submodels
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import unittest


class TestOhm(unittest.TestCase):
    def test_basic_processing(self):
        # Parameters
        param = pybamm.standard_parameters
        param.__dict__.update(pybamm.standard_parameters_lithium_ion.__dict__)

        # Variables
        phi_s_n = pybamm.Variable(
            "Negative electrode solid potential", domain=["negative electrode"]
        )
        phi_s_s = pybamm.Variable("Separator solid potential", domain=["separator"])
        phi_s_p = pybamm.Variable(
            "Positive electrode solid potential", domain=["positive electrode"]
        )
        phi_s = pybamm.Concatenation(phi_s_n, phi_s_s, phi_s_p)

        # Porosity and current
        eps = param.epsilon
        j = pybamm.interface.homogeneous_reaction(
            ["negative electrode", "separator", "positive electrode"]
        )

        # Set up model and test
        model = pybamm.electrode.Ohm(phi_s, eps, j, param)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
