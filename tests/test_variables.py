#
# Tests the variables class
#
import pybamm

import numpy as np
from numpy.linalg import norm

import unittest


class TestVariables(unittest.TestCase):
    """Test the variables class."""

    def setUp(self):
        self.param = pybamm.Parameters()
        self.mesh = pybamm.Mesh(self.param, 50)
        self.param.set_mesh(self.mesh)
        self.model = pybamm.ReactionDiffusionModel()
        self.model.set_simulation(self.param, None, None)
        self.vars = pybamm.Variables(self.model, self.mesh)

    def tearDown(self):
        del self.param
        del self.mesh
        del self.vars

    def test_variables_shapes(self):
        y = np.ones_like(self.mesh.xc)
        self.vars.update(self.mesh.time, y)
        self.assertEqual(self.vars.c.shape, self.mesh.xc.shape)
        self.assertEqual(self.vars.cn.shape, self.mesh.xcn.shape)
        self.assertEqual(self.vars.cs.shape, self.mesh.xcs.shape)
        self.assertEqual(self.vars.cp.shape, self.mesh.xcp.shape)

    def test_variables_average_basic(self):
        y = np.ones_like(self.mesh.xc)
        self.vars.update(self.mesh.time, y)
        self.vars.average()
        self.assertAlmostEqual(self.vars.c_avg, 1.0, places=10)
        self.assertAlmostEqual(self.vars.cn_avg, 1.0, places=10)
        self.assertAlmostEqual(self.vars.cs_avg, 1.0, places=10)
        self.assertAlmostEqual(self.vars.cp_avg, 1.0, places=10)

    def test_variables_average_convergence(self):
        ns = [50, 100, 200]
        err = [0] * len(ns)
        errn = [0] * len(ns)
        errs = [0] * len(ns)
        errp = [0] * len(ns)
        for i, n in enumerate(ns):
            mesh = pybamm.Mesh(self.param, n)
            y = mesh.xc ** 2
            vars = pybamm.Variables(self.model, mesh)
            vars.update(mesh.time, y)
            vars.average()
            c_avg_exact = 1 / 3
            cn_avg_exact = self.param.ln ** 2 / 3
            cs_avg_exact = ((1 - self.param.lp) ** 3 - self.param.ln ** 3) / (
                3 * self.param.ls
            )
            cp_avg_exact = (1 - (1 - self.param.lp) ** 3) / (3 * self.param.lp)
            err[i] = norm(vars.c_avg - c_avg_exact) / norm(c_avg_exact)
            errn[i] = norm(vars.cn_avg - cn_avg_exact) / norm(cn_avg_exact)
            errs[i] = norm(vars.cs_avg - cs_avg_exact) / norm(cs_avg_exact)
            errp[i] = norm(vars.cp_avg - cp_avg_exact) / norm(cp_avg_exact)

        for i in range(len(err) - 1):
            self.assertLess(err[i + 1] / err[i], 0.26)
            self.assertLess(errn[i + 1] / errn[i], 0.26)
            self.assertLess(errs[i + 1] / errs[i], 0.26)
            self.assertLess(errp[i + 1] / errp[i], 0.26)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
