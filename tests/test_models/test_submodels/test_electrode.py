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

        # Current density
        j = pybamm.interface.homogeneous_reaction(
            ["negative electrode", "separator", "positive electrode"]
        )
        j_n, j_s, j_p = j.orphans

        # Set up model and test
        model_n = pybamm.electrode.Ohm(phi_s_n, j_n, param)
        model_n_test = tests.StandardModelTest(model_n)
        model_n_test.test_all()

        model_p = pybamm.electrode.Ohm(phi_s_p, j_p, param)
        # overwrite boundary conditions for purposes of the test
        i_s_p = model_p.variables["Positive electrode solid current"]
        model_p.boundary_conditions = {phi_s_p: {"right": 0}, i_s_p: {"left": 0}}
        model_p_test = tests.StandardModelTest(model_p)
        model_p_test.test_all()

        model_whole = pybamm.electrode.Ohm(phi_s, j, param)
        # overwrite boundary conditions for purposes of the test
        i_s_n = model_whole.variables["Negative electrode solid current"]
        i_s_p = model_whole.variables["Positive electrode solid current"]
        model_whole.boundary_conditions = {
            phi_s_n: {"left": 0},
            i_s_n: {"right": 0},
            phi_s_p: {"right": 0},
            i_s_p: {"left": 0},
        }
        model_whole_test = tests.StandardModelTest(model_whole)
        model_whole_test.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
