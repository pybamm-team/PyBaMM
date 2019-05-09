#
# Tests for the particle submodel
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestStandardParticle(unittest.TestCase):
    def test_make_tree(self):
        param = pybamm.standard_parameters_lithium_ion

        c_n = pybamm.Variable("c_n", domain=["negative particle"])
        c_p = pybamm.Variable("c_p", domain=["positive particle"])

        j_n = pybamm.Scalar(1)
        j_p = pybamm.Scalar(1)

        model_n = pybamm.particle.Standard(param)
        model_n.set_differential_system(c_n, j_n)
        model_p = pybamm.particle.Standard(param)
        model_p.set_differential_system(c_p, j_p)

    def test_basic_processing(self):
        param = pybamm.standard_parameters_lithium_ion

        c_n = pybamm.Variable("c_n", domain=["negative particle"])
        c_p = pybamm.Variable("c_p", domain=["positive particle"])

        j_n = pybamm.Scalar(1)
        j_p = pybamm.Scalar(1)

        model_n = pybamm.particle.Standard(param)
        model_n.set_differential_system(c_n, j_n)
        model_p = pybamm.particle.Standard(param)
        model_p.set_differential_system(c_p, j_p)

        parameter_values = model_n.default_parameter_values
        parameter_values.process_model(model_n)
        parameter_values.process_model(model_p)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
