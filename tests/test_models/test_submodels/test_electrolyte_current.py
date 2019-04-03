#
# Tests for the electrolyte submodels
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import unittest


class TestMacInnesStefanMaxwell(unittest.TestCase):
    def test_basic_processing(self):
        # Parameters
        param = pybamm.standard_parameters
        param.__dict__.update(pybamm.standard_parameters_lead_acid.__dict__)

        # Variables
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        phi_e = pybamm.Variable("potential", whole_cell)

        # Other
        c_e = pybamm.Variable("concentration", whole_cell)
        eps = pybamm.Broadcast(pybamm.Scalar(1), whole_cell)
        j = pybamm.interface.homogeneous_reaction(whole_cell)

        # Set up model
        model = pybamm.electrolyte_current.MacInnesStefanMaxwell(
            c_e, phi_e, j, param, eps=eps
        )
        # some small changes so that tests pass
        i_e = model.variables["Electrolyte current"]
        model.algebraic.update({c_e: c_e - pybamm.Scalar(1)})
        model.initial_conditions.update({c_e: pybamm.Scalar(1)})
        model.boundary_conditions = {
            c_e: {"left": 1},
            phi_e: {"left": 0},
            i_e: {"right": 0},
        }

        # Test
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()


class TestFirstOrderPotential(unittest.TestCase):
    def test_basic_processing(self):
        loqs_model = pybamm.lead_acid.LOQS()
        c_e_n = pybamm.Broadcast(pybamm.Scalar(1), ["negative electrode"])
        c_e_s = pybamm.Broadcast(pybamm.Scalar(1), ["separator"])
        c_e_p = pybamm.Broadcast(pybamm.Scalar(1), ["positive electrode"])
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        param = pybamm.standard_parameters
        param.__dict__.update(pybamm.standard_parameters_lead_acid.__dict__)

        model = pybamm.electrolyte_current.StefanMaxwellFirstOrderPotential(
            loqs_model, c_e, param
        )

        parameter_values = loqs_model.default_parameter_values
        parameter_values.process_model(model)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
