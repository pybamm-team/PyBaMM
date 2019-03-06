#
# Tests for the electrolyte submodels
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestStandardElectrode(unittest.TestCase):
    def test_make_tree(self):
        phi_n = pybamm.Variable("phi_n", domain=["negative electrode"])
        G = pybamm.Scalar(1)
        pybamm.electrode.Standard(phi_n, G)

    def test_basic_processing(self):
        phi_n = pybamm.Variable("phi_n", domain=["negative electrode"])
        G = pybamm.Scalar(0.001)
        model = pybamm.electrode.Standard(phi_n, G)

        param = model.default_parameter_values
        param.process_model(model)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
