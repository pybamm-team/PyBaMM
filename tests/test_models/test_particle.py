#
# Tests for the particle submodel
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestStandardParticle(unittest.TestCase):
    def test_make_tree(self):
        c_n = pybamm.Variable("c_n", domain=["negative particle"])
        c_p = pybamm.Variable("c_p", domain=["positive particle"])

        G_n = pybamm.Scalar(1, domain=["negative particle"])
        G_p = pybamm.Scalar(1, domain=["positive particle"])

        pybamm.models.submodels.particle.Standard(c_n, G_n)
        pybamm.models.submodels.particle.Standard(c_p, G_p)

    def test_basic_processing(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c_e = pybamm.Variable("c_e", domain=whole_cell)
        phi_e = pybamm.Variable("phi_e", domain=whole_cell)
        G = pybamm.Scalar(0.001)
        model = pybamm.electrolyte_current.StefanMaxwell(c_e, phi_e, G)

        param = model.default_parameter_values
        param.process_model(model)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
