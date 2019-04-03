#
# Tests for the particle submodel
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestStandardParticle(unittest.TestCase):
    def test_make_tree(self):
        param = pybamm.standard_parameters
        param.__dict__.update(pybamm.standard_parameters_lithium_ion.__dict__)

        c_n = pybamm.Variable("c_n", domain=["negative particle"])
        c_p = pybamm.Variable("c_p", domain=["positive particle"])

        j_n = pybamm.Scalar(1)
        j_p = pybamm.Scalar(1)

        pybamm.models.submodels.particle.Standard(c_n, j_n, param)
        pybamm.models.submodels.particle.Standard(c_p, j_p, param)

    def test_basic_processing(self):
        param = pybamm.standard_parameters
        param.__dict__.update(pybamm.standard_parameters_lithium_ion.__dict__)

        c_n = pybamm.Variable("c_n", domain=["negative particle"])
        c_p = pybamm.Variable("c_p", domain=["positive particle"])

        j_n = pybamm.Scalar(1)
        j_p = pybamm.Scalar(1)

        model_n = pybamm.models.submodels.particle.Standard(c_n, j_n, param)
        model_p = pybamm.models.submodels.particle.Standard(c_p, j_p, param)

        param = model_n.default_parameter_values
        param = model_p.default_parameter_values
        param.process_model(model_n)
        param.process_model(model_p)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
