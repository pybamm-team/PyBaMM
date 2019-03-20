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
        phi_s_s = pybamm.Broadcast(pybamm.Scalar(0), ["separator"])
        phi_s_p = pybamm.Variable(
            "Positive electrode solid potential", domain=["positive electrode"]
        )
        phi_s = pybamm.Concatenation(phi_s_n, phi_s_s, phi_s_p)

        # Porosity and current
        eps = param.epsilon
        j = pybamm.interface.homogeneous_reaction(
            ["negative electrode", "separator", "positive electrode"]
        )
        eps_n, eps_s, eps_p = eps.orphans
        j_n, j_s, j_p = j.orphans

        # Set up model and test
        model_n = pybamm.electrode.Ohm(phi_s_n, eps_n, j_n, param)
        model_n_test = tests.StandardModelTest(model_n)
        model_n_test.test_all()

        model_p = pybamm.electrode.Ohm(phi_s_p, eps_p, j_p, param)
        model_p_test = tests.StandardModelTest(model_p)
        model_p_test.test_all()

        model_whole = pybamm.electrode.Ohm(phi_s, eps, j, param)
        model_whole_test = tests.StandardModelTest(model_whole)
        model_whole_test.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
