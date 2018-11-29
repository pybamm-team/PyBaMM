#
# Test for the operator class
#
import pybamm

import numpy as np
import unittest


class TestOperators(unittest.TestCase):
    def test_grad_div_1D_FV_basic(self):
        param = pybamm.Parameters()
        mesh = pybamm.Mesh(param, target_npts=50)

        y = np.ones_like(mesh.x.centres)
        N = np.ones_like(mesh.x.edges)
        yn = np.ones_like(mesh.xn.centres)
        Nn = np.ones_like(mesh.xn.edges)
        yp = np.ones_like(mesh.xp.centres)
        Np = np.ones_like(mesh.xp.edges)

        # Get all operators
        all_operators = pybamm.AllOperators("Finite Volumes", mesh)

        # Check output shape
        self.assertEqual(all_operators.x.grad(y).shape[0], y.shape[0] - 1)
        self.assertEqual(all_operators.x.div(N).shape[0], N.shape[0] - 1)
        self.assertEqual(all_operators.xn.grad(yn).shape[0], yn.shape[0] - 1)
        self.assertEqual(all_operators.xn.div(Nn).shape[0], Nn.shape[0] - 1)
        self.assertEqual(all_operators.xp.grad(yp).shape[0], yp.shape[0] - 1)
        self.assertEqual(all_operators.xp.div(Np).shape[0], Np.shape[0] - 1)

        # Check grad and div are both zero
        self.assertEqual(np.linalg.norm(all_operators.x.grad(y)), 0)
        self.assertEqual(np.linalg.norm(all_operators.x.div(N)), 0)
        self.assertEqual(np.linalg.norm(all_operators.xn.grad(yn)), 0)
        self.assertEqual(np.linalg.norm(all_operators.xn.div(Nn)), 0)
        self.assertEqual(np.linalg.norm(all_operators.xp.grad(yp)), 0)
        self.assertEqual(np.linalg.norm(all_operators.xp.div(Np)), 0)

    def test_grad_div_1D_FV_convergence(self):
        # Convergence
        param = pybamm.Parameters()
        ns = [50, 100, 200]
        grad_errs = [0] * len(ns)
        div_errs = [0] * len(ns)
        for i, n in enumerate(ns):
            # Define problem and exact solutions
            mesh = pybamm.Mesh(param, target_npts=n)
            y = np.sin(mesh.x.centres)
            grad_y_exact = np.cos(mesh.x.edges[1:-1])
            div_exact = -np.sin(mesh.x.centres)

            # Get operators and flux
            operators = pybamm.operators.CartesianFiniteVolumes(mesh.x)
            grad_y_approx = operators.grad(y)

            # Calculate divergence of exact flux to avoid double errors
            # (test for those separately)
            N_exact = np.cos(mesh.x.edges)
            div_approx = operators.div(N_exact)

            # Calculate errors
            grad_errs[i] = np.linalg.norm(
                grad_y_approx - grad_y_exact
            ) / np.linalg.norm(grad_y_exact)
            div_errs[i] = np.linalg.norm(div_approx - div_exact) / np.linalg.norm(
                div_exact
            )

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
