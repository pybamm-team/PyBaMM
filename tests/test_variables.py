from pybamm.parameters import Parameters
from pybamm.mesh import Mesh, UniformMesh
from pybamm.variables import Variables

import numpy as np

import unittest

class TestVariables(unittest.TestCase):

    def test_variables_shapes(self):
        param = Parameters()
        mesh = Mesh(param, 50)
        y = np.ones_like(mesh.xc)
        vars = Variables(y, param, mesh)

        self.assertEqual(vars.c.shape, mesh.xc.shape)
        self.assertEqual(vars.cn.shape, mesh.xcn.shape)
        self.assertEqual(vars.cs.shape, mesh.xcs.shape)
        self.assertEqual(vars.cp.shape, mesh.xcp.shape)


if __name__ == "__main__":
    unittest.main()
