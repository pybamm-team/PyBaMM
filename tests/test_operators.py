import pybamm

import numpy as np
import unittest


class TestOperators(unittest.TestCase):
    def setUp(self):
        # Generate parameters and grids
        self.param = pybamm.Parameters()
        self.mesh = pybamm.Mesh(self.param, 50)

    def tearDown(self):
        del self.param
        del self.mesh

    def test_operators_basic(self):
        with self.assertRaises(NotImplementedError):
            pybamm.Operators("Finite Volumes", "not a domain", self.mesh)

    def test_grad_div_1D_FV_basic(self):
        y = np.ones_like(self.mesh.xc)
        N = np.ones_like(self.mesh.x)
        yn = np.ones_like(self.mesh.xcn)
        Nn = np.ones_like(self.mesh.xn)
        yp = np.ones_like(self.mesh.xcp)
        Np = np.ones_like(self.mesh.xp)

        # Get operators
        operators = {
            domain: pybamm.Operators("Finite Volumes", domain, self.mesh)
            for domain in ["xc", "xcn", "xcp"]
        }

        # Check output shape
        self.assertEqual(operators["xc"].grad(y).shape[0], y.shape[0] - 1)
        self.assertEqual(operators["xc"].div(N).shape[0], N.shape[0] - 1)
        self.assertEqual(operators["xcn"].grad(yn).shape[0], yn.shape[0] - 1)
        self.assertEqual(operators["xcn"].div(Nn).shape[0], Nn.shape[0] - 1)
        self.assertEqual(operators["xcp"].grad(yp).shape[0], yp.shape[0] - 1)
        self.assertEqual(operators["xcp"].div(Np).shape[0], Np.shape[0] - 1)

        # Check grad and div are both zero
        self.assertEqual(np.linalg.norm(operators["xc"].grad(y)), 0)
        self.assertEqual(np.linalg.norm(operators["xc"].div(N)), 0)
        self.assertEqual(np.linalg.norm(operators["xcn"].grad(yn)), 0)
        self.assertEqual(np.linalg.norm(operators["xcn"].div(Nn)), 0)
        self.assertEqual(np.linalg.norm(operators["xcp"].grad(yp)), 0)
        self.assertEqual(np.linalg.norm(operators["xcp"].div(Np)), 0)

    def test_grad_div_1D_FV_convergence(self):
        # Convergence
        ns = [50, 100, 200]
        grad_errs = [0] * len(ns)
        div_errs = [0] * len(ns)
        for i, n in enumerate(ns):
            # Define problem and exact solutions
            mesh = pybamm.Mesh(self.param, n)
            y = np.sin(mesh.xc)
            grad_y_exact = np.cos(mesh.x[1:-1])
            div_exact = -np.sin(mesh.xc)

            # Get operators and flux
            operators = pybamm.Operators("Finite Volumes", "xc", mesh)
            grad_y_approx = operators.grad(y)

            # Calculate divergence of exact flux to avoid double errors
            # (test for those separately)
            N_exact = np.cos(mesh.x)
            div_approx = operators.div(N_exact)

            # Calculate errors
            grad_errs[i] = np.linalg.norm(
                grad_y_approx - grad_y_exact
            ) / np.linalg.norm(grad_y_exact)
            div_errs[i] = np.linalg.norm(
                div_approx - div_exact
            ) / np.linalg.norm(div_exact)

        # Expect h**2 convergence
        [
            self.assertLess(grad_errs[i + 1] / grad_errs[i], 0.26)
            for i in range(len(grad_errs) - 1)
        ]
        [
            self.assertLess(div_errs[i + 1] / div_errs[i], 0.26)
            for i in range(len(div_errs) - 1)
        ]


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
