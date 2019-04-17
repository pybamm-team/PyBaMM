#
# Test for the operator class
#
import pybamm
from tests import get_mesh_for_testing, get_p2d_mesh_for_testing

import numpy as np
import unittest


class TestFiniteVolumeConvergence(unittest.TestCase):
    def test_grad_convergence(self):
        # Convergence
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {
                "left": pybamm.Scalar(np.exp(0)),
                "right": pybamm.Scalar(np.exp(1)),
            }
        }

        # Function for convergence testing
        def get_error(n):
            # create discretisation
            mesh = get_mesh_for_testing(n)
            spatial_methods = {"macroscale": pybamm.FiniteVolume}
            disc = pybamm.Discretisation(mesh, spatial_methods)
            disc._bcs = boundary_conditions

            # Set up discretisation
            whole_cell = ["negative electrode", "separator", "positive electrode"]
            combined_submesh = mesh.combine_submeshes(*whole_cell)

            # Define exact solutions
            y = np.exp(combined_submesh[0].nodes)
            grad_exact = np.exp(combined_submesh[0].edges)

            # Discretise and evaluate
            disc.set_variable_slices([var])
            grad_eqn_disc = disc.process_symbol(grad_eqn)
            grad_approx = grad_eqn_disc.evaluate(None, y)

            # Return difference
            return grad_approx - grad_exact

        # Get errors
        ns = 2 ** np.arange(1, 8)
        errs = {n: get_error(int(n)) for n in ns}
        # expect quadratic convergence at internal points
        for idx_factor in [1, 3, 5]:
            err_at_idx = np.array([errs[n][idx_factor * n // 2] for n in ns])
            rates = np.log2(err_at_idx[:-1] / err_at_idx[1:])
            np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)
        # expect linear convergence at the boundaries
        for idx in [0, -1]:
            err_at_idx = np.array([errs[n][idx] for n in ns])
            rates = np.log2(err_at_idx[:-1] / err_at_idx[1:])
            np.testing.assert_array_less(0.98 * np.ones_like(rates), rates)

    def test_div_convergence(self):
        # Convergence
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        N = pybamm.Function(np.arcsin, var) ** 2 * pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(np.cos(1))}
        }

        # Function for convergence testing
        def get_error(n):
            # create discretisation
            mesh = get_mesh_for_testing(n)
            spatial_methods = {"macroscale": pybamm.FiniteVolume}
            disc = pybamm.Discretisation(mesh, spatial_methods)
            disc._bcs = boundary_conditions

            whole_cell = ["negative electrode", "separator", "positive electrode"]
            combined_submesh = mesh.combine_submeshes(*whole_cell)

            # Define exact solutions
            y = np.sin(combined_submesh[0].nodes)
            x = combined_submesh[0].nodes
            # N = x**2 * cos(x) --> dNdx = x*(2cos(x) - xsin(x))
            div_exact = x * (2 * np.cos(x) - x * np.sin(x))

            # Discretise and evaluate
            disc.set_variable_slices([var])
            div_eqn_disc = disc.process_symbol(div_eqn)
            div_approx = div_eqn_disc.evaluate(None, y)

            # Return difference between approx and exact
            return div_approx - div_exact

        # Get errors
        ns = 3 ** np.arange(1, 8)
        errs = {n: get_error(int(n)) for n in ns}
        # expect quadratic convergence at internal points
        for idx_factor in [1, 2, 3]:
            err_at_idx = np.array(
                [errs[n][((2 * idx_factor + 1) * n - 3) // 6] for n in ns]
            )
            rates = np.log(err_at_idx[:-1] / err_at_idx[1:]) / np.log(3)
            np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)
        # expect linear convergence at the boundaries
        for idx in [0, -1]:
            err_at_idx = np.array([errs[n][idx] for n in ns])
            rates = np.log(err_at_idx[:-1] / err_at_idx[1:]) / np.log(3)
            np.testing.assert_array_less(0.98 * np.ones_like(rates), rates)

    def test_spherical_convergence(self):
        # test div( grad( r**3 )) == 12*r

        domain = ["negative particle"]
        c = pybamm.Variable("c", domain=domain)
        N = pybamm.grad(c)
        eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(3)}
        }

        def get_error(n):
            mesh = get_mesh_for_testing(n)
            spatial_methods = {"negative particle": pybamm.FiniteVolume}
            disc = pybamm.Discretisation(mesh, spatial_methods)
            disc._bcs = boundary_conditions
            mesh = disc.mesh["negative particle"]
            r = mesh[0].nodes

            # exact solution
            y = r ** 3
            div_grad_exact = 12 * r

            # discretise and evaluate
            variables = [c]
            disc.set_variable_slices(variables)
            eqn_disc = disc.process_symbol(eqn)
            div_grad_approx = eqn_disc.evaluate(None, y)

            # Return difference between approx and exact
            return div_grad_approx - div_grad_exact

        # Get errors
        ns = 100 * 3 ** np.arange(2, 5)
        errs = {n: get_error(int(n)) for n in ns}
        # expect quadratic convergence at internal points
        for idx_factor in [0, 1, 2]:
            err_at_idx = np.array(
                [errs[n][((2 * idx_factor + 1) * n - 3) // 6] for n in ns]
            )
            rates = np.log(err_at_idx[:-1] / err_at_idx[1:]) / np.log(3)
            np.testing.assert_array_less(1.9 * np.ones_like(rates), rates)
        # expect linear convergence at the boundaries
        for idx in [0, -1]:
            err_at_idx = np.array([errs[n][idx] for n in ns])
            rates = np.log(err_at_idx[:-1] / err_at_idx[1:]) / np.log(3)
            np.testing.assert_array_less(0.98 * np.ones_like(rates), rates)

    def test_p2d_spherical_convergence(self):
        # test div( grad( sin(r) )) == (2/r)*cos(r) - sin(r)

        domain = ["negative particle"]
        c = pybamm.Variable("c", domain=domain)
        N = pybamm.grad(c)
        eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(np.cos(0)), "right": pybamm.Scalar(np.cos(1))}
        }

        def get_error(m):
            mesh = get_p2d_mesh_for_testing(3, m)
            spatial_methods = {"negative particle": pybamm.FiniteVolume}
            disc = pybamm.Discretisation(mesh, spatial_methods)
            disc._bcs = boundary_conditions
            mesh = disc.mesh["negative particle"]
            r = mesh[0].nodes

            prim_pts = mesh[0].npts
            sec_pts = len(mesh)

            # exact solution
            y = np.kron(np.ones(sec_pts), np.sin(r))
            exact = (2 / r) * np.cos(r) - np.sin(r)
            exact = np.kron(np.ones(sec_pts), exact)

            # discretise and evaluate
            variables = [c]
            disc.set_variable_slices(variables)
            eqn_disc = disc.process_symbol(eqn)
            approx_eval = eqn_disc.evaluate(None, y)
            approx_eval = np.reshape(approx_eval, [sec_pts, prim_pts])
            approx = approx_eval
            approx = np.reshape(approx, [sec_pts * prim_pts])

            # Return difference between approx and exact
            return approx - exact

        # Get errors
        ns = 100 * 3 ** np.arange(2, 5)
        errs = {n: get_error(int(n)) for n in ns}
        # expect quadratic convergence at internal points
        for idx_factor in [0, 1, 2]:
            err_at_idx = np.array(
                [errs[n][((2 * idx_factor + 1) * n - 3) // 6] for n in ns]
            )
            rates = np.log(err_at_idx[:-1] / err_at_idx[1:]) / np.log(3)
            np.testing.assert_array_less(1.9 * np.ones_like(rates), rates)
        # expect linear convergence at the boundaries
        for idx in [0, -1]:
            err_at_idx = np.array([errs[n][idx] for n in ns])
            rates = np.log(err_at_idx[:-1] / err_at_idx[1:]) / np.log(3)
            np.testing.assert_array_less(0.98 * np.ones_like(rates), rates)

    def test_p2d_with_x_dep_bcs_spherical_convergence(self):
        # test div( grad( sin(r) )) == (2/r)*cos(r) - *sin(r)

        xn = pybamm.SpatialVariable("x", ["negative electrode"])
        c = pybamm.Variable("c", domain=["negative particle"])
        N = pybamm.grad(c)
        eqn = pybamm.div(N)
        boundary_conditions = {
            N: {
                "left": pybamm.Scalar(np.cos(0)) * xn,
                "right": pybamm.Scalar(np.cos(1)) * xn,
            }
        }

        def get_error(m):
            mesh = get_p2d_mesh_for_testing(6, m)
            spatial_methods = {
                "negative particle": pybamm.FiniteVolume,
                "negative electrode": pybamm.FiniteVolume,
            }
            disc = pybamm.Discretisation(mesh, spatial_methods)
            mesh = disc.mesh["negative particle"]
            disc._bcs = {
                key.id: disc.process_dict(value)
                for key, value in boundary_conditions.items()
            }
            r = mesh[0].nodes
            xn_disc = disc.process_symbol(xn)

            prim_pts = mesh[0].npts
            sec_pts = len(mesh)

            # exact solution
            y = np.kron(xn_disc.entries, np.sin(r))
            div_grad_exact = (2 / r) * np.cos(r) - np.sin(r)
            div_grad_exact = np.kron(xn_disc.entries, div_grad_exact)

            # discretise and evaluate
            variables = [c]
            disc.set_variable_slices(variables)
            eqn_disc = disc.process_symbol(eqn)
            approx_eval = eqn_disc.evaluate(None, y)
            approx_eval = np.reshape(approx_eval, [sec_pts, prim_pts])
            approx = approx_eval
            approx = np.reshape(approx, [sec_pts * prim_pts])

            # Return difference between approx and exact
            return approx - div_grad_exact

        # Get errors
        ns = 100 * 3 ** np.arange(2, 5)
        errs = {n: get_error(int(n)) for n in ns}
        # expect quadratic convergence at internal points
        for idx_factor in [0, 1, 2]:
            err_at_idx = np.array(
                [errs[n][((2 * idx_factor + 1) * n - 3) // 6] for n in ns]
            )
            rates = np.log(err_at_idx[:-1] / err_at_idx[1:]) / np.log(3)
            np.testing.assert_array_less(1.9 * np.ones_like(rates), rates)
        # expect linear convergence at the boundaries
        for idx in [0, -1]:
            err_at_idx = np.array([errs[n][idx] for n in ns])
            rates = np.log(err_at_idx[:-1] / err_at_idx[1:]) / np.log(3)
            np.testing.assert_array_less(0.98 * np.ones_like(rates), rates)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
