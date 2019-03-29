#
# Tests for the electrolyte submodels
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import unittest


class TestStefanMaxwellDiffusion(unittest.TestCase):
    def test_make_tree(self):
        # Parameter values
        param = pybamm.standard_parameters
        param.__dict__.update(pybamm.standard_parameters_lithium_ion.__dict__)

        # Variables and parameters
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        j = pybamm.Scalar(1)
        c_e = pybamm.Variable("c_e", domain=whole_cell)
        epsilon = pybamm.Scalar(1)
        pybamm.electrolyte_diffusion.StefanMaxwell(c_e, epsilon, j, param)

    def test_basic_processing(self):
        # Parameter values
        param = pybamm.standard_parameters
        param.__dict__.update(pybamm.standard_parameters_lithium_ion.__dict__)

        # Variables and parameters
        j = pybamm.Scalar(0.001)
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c_e = pybamm.Variable("c_e", domain=whole_cell)
        epsilon = pybamm.Scalar(1)
        model = pybamm.electrolyte_diffusion.StefanMaxwell(c_e, epsilon, j, param)

        modeltest = tests.StandardModelTest(model)
        # Either
        # 1. run the tests individually (can pass options to individual tests)
        # modeltest.test_processing_parameters()
        # modeltest.test_processing_disc()
        # modeltest.test_solving()

        # Or
        # 2. run all the tests in one go
        modeltest.test_all()

    def test_basic_processing_different_bcs(self):
        "Change the boundary conditions and test"
        # Parameter values
        param = pybamm.standard_parameters
        param.__dict__.update(pybamm.standard_parameters_lithium_ion.__dict__)

        # Variables and parameters
        j = pybamm.Scalar(0.001)
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c_e = pybamm.Variable("c_e", domain=whole_cell)
        epsilon = pybamm.Scalar(1)
        model = pybamm.electrolyte_diffusion.StefanMaxwell(c_e, epsilon, j, param)

        # Dirichlet conditions
        model.boundary_conditions = {c_e: {"left": 0, "right": 0}}
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

        # Dirichlet and Neumann conditions
        model2 = pybamm.electrolyte_diffusion.StefanMaxwell(c_e, epsilon, j, param)
        N_e = model2.variables["Cation flux"]
        model2.boundary_conditions = {c_e: {"left": 0}, N_e: {"right": 0}}
        modeltest2 = tests.StandardModelTest(model2)
        modeltest2.test_all()

        model3 = pybamm.electrolyte_diffusion.StefanMaxwell(c_e, epsilon, j, param)
        N_e = model3.variables["Cation flux"]
        model3.boundary_conditions = {N_e: {"left": 0}, c_e: {"right": 0}}
        modeltest3 = tests.StandardModelTest(model3)
        modeltest3.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
