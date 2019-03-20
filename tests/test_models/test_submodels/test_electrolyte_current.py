#
# Tests for the electrolyte submodels
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestFirstOrderPotential(unittest.TestCase):
    def test_basic_processing(self):
        loqs_model = pybamm.lead_acid.LOQS()
        c_e = pybamm.Broadcast(
            pybamm.Scalar(1),
            domain=["negative electrode", "separator", "positive electrode"],
        )

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
