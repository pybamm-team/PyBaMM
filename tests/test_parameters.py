from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestParameters(unittest.TestCase):
    def test_read_parameters_csv(self):
        data = pybamm.read_parameters_csv("input/parameters/default.csv")
        self.assertEqual(data["R"], 8.314)

    def test_parameters_defaults(self):
        # basic tests on how the parameters interact
        param = pybamm.Parameters()
        self.assertAlmostEqual(param.ln + param.ls + param.lp, 1, places=10)

    def test_parameters_options(self):
        param = pybamm.Parameters(
            optional_parameters={"Ln": 1 / 3, "Ls": 1 / 3, "Lp": 1 / 3}
        )
        self.assertAlmostEqual(param.ln + param.ls + param.lp, 1, places=10)
        param = pybamm.Parameters(
            optional_parameters="input/parameters/optional_test.csv"
        )
        self.assertAlmostEqual(param.ln + param.ls + param.lp, 1, places=10)

    def test_mesh_dependent_parameters(self):
        param = pybamm.Parameters()
        mesh = pybamm.Mesh(param, 10)
        param.set_mesh_dependent_parameters(mesh)
        self.assertEqual(param.s.shape, mesh.xc.shape)


if __name__ == "__main__":
    unittest.main()
