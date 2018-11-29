#
# Tests the variables class
#
import pybamm

import numpy as np
from numpy.linalg import norm

import unittest


class ModelForTesting(object):
    def __init__(self, param, mesh):
        self.param = param
        self.mesh = mesh
        self.pde_variables = [("c", mesh.x)]


class TestVariables(unittest.TestCase):
    """Test the variables class."""

    def setUp(self):
        # Set up with a model and default simulation
        param = pybamm.Parameters()
        self.mesh = pybamm.Mesh(param)
        model = ModelForTesting(param, self.mesh)
        self.vars = pybamm.Variables(model)
        self.mesh = self.mesh
        y = np.ones_like(self.mesh.x.centres)
        self.vars.update(self.mesh.time, y)

    def tearDown(self):
        del self.vars
        del self.mesh

    def test_variables_shapes(self):
        # Test
        self.assertEqual(self.vars.c.shape, self.mesh.x.centres.shape)
        self.assertEqual(self.vars.cn.shape, self.mesh.xn.centres.shape)
        self.assertEqual(self.vars.cs.shape, self.mesh.xs.centres.shape)
        self.assertEqual(self.vars.cp.shape, self.mesh.xp.centres.shape)

    def test_variables_neg_pos(self):
        np.testing.assert_array_equal(self.vars.cn, self.vars.neg["c"])
        np.testing.assert_array_equal(self.vars.cp, self.vars.pos["c"])

    def test_variables_average_basic(self):
        self.vars.average()
        self.assertAlmostEqual(self.vars.c_avg, 1.0, places=10)
        self.assertAlmostEqual(self.vars.cn_avg, 1.0, places=10)
        self.assertAlmostEqual(self.vars.cs_avg, 1.0, places=10)
        self.assertAlmostEqual(self.vars.cp_avg, 1.0, places=10)

    def test_variables_average_convergence(self):
        # Set up
        param = pybamm.Parameters()

        ns = [50, 100, 200]
        err = [0] * len(ns)
        errn = [0] * len(ns)
        errs = [0] * len(ns)
        errp = [0] * len(ns)
        for i, n in enumerate(ns):
            # Set up
            mesh = pybamm.Mesh(param, n)
            model = ModelForTesting(param, mesh)
            y = mesh.x.centres ** 2
            vars = pybamm.Variables(model)
            vars.update(mesh.time, y)
            vars.average()

            # Exact solution
            ln, ls, lp = [param.geometric.__dict__[l] for l in ["ln", "ls", "lp"]]
            c_avg_exact = 1 / 3
            cn_avg_exact = ln ** 2 / 3
            cs_avg_exact = ((1 - lp) ** 3 - ln ** 3) / (3 * ls)
            cp_avg_exact = (1 - (1 - lp) ** 3) / (3 * lp)

            # Compare
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
