import pybamm

import numpy as np
from numpy.linalg import norm

import unittest


class TestVariables(unittest.TestCase):
    def setUp(self):
        self.param = pybamm.Parameters()
        self.mesh = pybamm.Mesh(self.param, 50)
        self.y = np.ones_like(self.mesh.xc)
        self.model = pybamm.Model("Electrolyte diffusion")
        self.vars = pybamm.Variables(
            self.mesh.time, self.y, self.model, self.mesh
        )

    def tearDown(self):
        del self.param
        del self.mesh
        del self.y
        del self.vars

    def test_variables_shapes(self):
        self.assertEqual(self.vars.c.shape, self.mesh.xc.shape)
        self.assertEqual(self.vars.cn.shape, self.mesh.xcn.shape)
        self.assertEqual(self.vars.cs.shape, self.mesh.xcs.shape)
        self.assertEqual(self.vars.cp.shape, self.mesh.xcp.shape)

    def test_variables_average_basic(self):
        self.vars.average(self.param, self.mesh)
        self.assertEqual(self.vars.c_avg, 1.0)
        self.assertEqual(self.vars.cn_avg, 1.0)
        self.assertEqual(self.vars.cp_avg, 1.0)

    def test_variables_average_convergence(self):
        ns = [50, 100, 200]
        errs = [0] * len(ns)
        for i, n in enumerate(ns):
            mesh = pybamm.Mesh(self.param, n)
            y = mesh.xc ** 2
            vars = pybamm.Variables(mesh.time, y, self.model, mesh)
            vars.average(self.param, mesh)
            c_avg_exact = 1 / 3
            errs[i] = norm(vars.c_avg - c_avg_exact) / norm(c_avg_exact)
        [
            self.assertLess(errs[i + 1] / errs[i], 0.26)
            for i in range(len(errs) - 1)
        ]


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
