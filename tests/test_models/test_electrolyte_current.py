#
# Tests for the electrolyte submodels
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


@unittest.skip("not implemented")
class TestStefanMaxwellCurrent(unittest.TestCase):
    def test_make_tree(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c_e = pybamm.Variable("c_e", domain=whole_cell)
        phi_e = pybamm.Variable("phi_e", domain=whole_cell)
        G = pybamm.Scalar(1)
        pybamm.electrolyte_current.StefanMaxwell(c_e, phi_e, G)

    def test_basic_processing(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c_e = pybamm.Variable("c_e", domain=whole_cell)
        phi_e = pybamm.Variable("phi_e", domain=whole_cell)
        G = pybamm.Scalar(0.001)
        model = pybamm.electrolyte_current.StefanMaxwell(c_e, phi_e, G)

        param = model.default_parameter_values
        param.process_model(model)


class TestFirstOrderPotential(unittest.TestCase):
    def test_basic_processing(self):
        loqs_model = pybamm.lead_acid.LOQS()
        c_e_n = pybamm.Variable(
            "Negative electrode concentration", domain=["negative electrode"]
        )
        c_e_s = pybamm.Variable("Separator concentration", domain=["separator"])
        c_e_p = pybamm.Variable(
            "Positive electrode concentration", domain=["positive electrode"]
        )
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        param = pybamm.standard_parameters
        param.__dict__.update(pybamm.standard_parameters_lead_acid.__dict__)

        model = pybamm.electrolyte_current.FirstOrderPotential(loqs_model, c_e, param)

        parameter_values = loqs_model.default_parameter_values
        parameter_values.process_model(model)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
