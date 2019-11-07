#
# Test for the operator class
#
import pybamm

# from tests import (
#     get_mesh_for_testing,
#     get_p2d_mesh_for_testing,
#     get_1p1d_mesh_for_testing,
# )

import numpy as np

# from scipy.sparse import kron, eye
import unittest


def errors(pts, function, method_options, bcs=None):

    domain = "test"
    x = pybamm.SpatialVariable("x", domain=domain)
    geometry = {
        domain: {"primary": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}
    }
    submesh_types = {domain: pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)}
    var_pts = {x: pts}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

    spatial_methods = {"test": pybamm.FiniteVolume(method_options)}
    disc = pybamm.Discretisation(mesh, spatial_methods)

    var = pybamm.Variable("var", domain="test")
    left_extrap = pybamm.BoundaryValue(var, "left")
    right_extrap = pybamm.BoundaryValue(var, "right")

    if bcs:
        model = pybamm.BaseBatteryModel()
        bc_dict = {var: bcs}
        model.boundary_conditions = bc_dict
        disc.bcs = disc.process_boundary_conditions(model)

    submesh = mesh["test"]
    y, l_true, r_true = function(submesh[0].nodes)

    disc.set_variable_slices([var])
    left_extrap_processed = disc.process_symbol(left_extrap)
    right_extrap_processed = disc.process_symbol(right_extrap)

    l_error = np.abs(l_true - left_extrap_processed.evaluate(None, y))
    r_error = np.abs(r_true - right_extrap_processed.evaluate(None, y))

    return l_error, r_error


def get_errors(function, method_options, pts, bcs=None):

    l_errors = np.zeros(pts.shape)
    r_errors = np.zeros(pts.shape)

    for i, pt in enumerate(pts):
        l_errors[i], r_errors[i] = errors(pt, function, method_options, bcs)

    return l_errors, r_errors


class TestExtrapolation(unittest.TestCase):
    def test_convergence_without_bcs(self):

        # all tests are performed on x in [0, 1]
        linear = {"extrapolation": {"order": "linear"}}
        quad = {"extrapolation": {"order": "quadratic"}}

        def x_squared(x):
            y = x ** 2
            l_true = 0
            r_true = 1
            return y, l_true, r_true

        pts = 10 ** np.arange(1, 6, 1)
        dx = 1 / pts

        l_errors_lin, r_errors_lin = get_errors(x_squared, linear, pts)
        l_errors_quad, r_errors_quad = get_errors(x_squared, quad, pts)

        l_lin_rates = np.log(l_errors_lin[:-1] / l_errors_lin[1:]) / np.log(
            dx[:-1] / dx[1:]
        )

        r_lin_rates = np.log(r_errors_lin[:-1] / r_errors_lin[1:]) / np.log(
            dx[:-1] / dx[1:]
        )

        np.testing.assert_array_almost_equal(l_lin_rates, 2)
        np.testing.assert_array_almost_equal(r_lin_rates, 2)

        # check quadratic is equal up to machine precision
        np.testing.assert_array_almost_equal(l_errors_quad, 0, decimal=14)
        np.testing.assert_array_almost_equal(r_errors_quad, 0, decimal=14)

        def x_cubed(x):
            y = x ** 3
            l_true = 0
            r_true = 1
            return y, l_true, r_true

        l_errors_lin, r_errors_lin = get_errors(x_squared, linear, pts)

        l_lin_rates = np.log(l_errors_lin[:-1] / l_errors_lin[1:]) / np.log(
            dx[:-1] / dx[1:]
        )

        r_lin_rates = np.log(r_errors_lin[:-1] / r_errors_lin[1:]) / np.log(
            dx[:-1] / dx[1:]
        )

        np.testing.assert_array_almost_equal(l_lin_rates, 2)
        np.testing.assert_array_almost_equal(r_lin_rates, 2)

        # quadratic case
        pts = 5 ** np.arange(1, 7, 1)
        dx = 1 / pts
        l_errors_quad, r_errors_quad = get_errors(x_cubed, quad, pts)

        l_quad_rates = np.log(l_errors_quad[:-1] / l_errors_quad[1:]) / np.log(
            dx[:-1] / dx[1:]
        )

        r_quad_rates = np.log(r_errors_quad[:-1] / r_errors_quad[1:]) / np.log(
            dx[:-1] / dx[1:]
        )

        np.testing.assert_array_almost_equal(l_quad_rates, 3)
        np.testing.assert_array_almost_equal(r_quad_rates, 3, decimal=3)

    def test_extrapolation_with_bcs_right_neumann(self):
        # simple particle with a flux bc

        pts = 10 ** np.arange(1, 6, 1)
        dx = 1 / pts

        left_val = 1
        right_flux = 2

        def x_cubed(x):
            n = 3
            f_x = x ** n
            f_l = 0
            fp_r = n
            y = f_x + (right_flux - fp_r) * x + (left_val - f_l)

            true_left = left_val
            true_right = 1 + (right_flux - fp_r) + (left_val - f_l)

            return y, true_left, true_right

        bcs = {"left": (left_val, "Dirichlet"), "right": (right_flux, "Neumann")}

        linear = {"extrapolation": {"order": "linear", "use bcs": True}}
        quad = {"extrapolation": {"order": "quadratic", "use bcs": True}}
        l_errors_lin_no_bc, r_errors_lin_no_bc = get_errors(x_cubed, linear, pts,)
        l_errors_quad_no_bc, r_errors_quad_no_bc = get_errors(x_cubed, quad, pts,)

        l_errors_lin_with_bc, r_errors_lin_with_bc = get_errors(
            x_cubed, linear, pts, bcs
        )
        l_errors_quad_with_bc, r_errors_quad_with_bc = get_errors(
            x_cubed, quad, pts, bcs
        )

        # test that with bc is better than without

        np.testing.assert_array_less(l_errors_lin_with_bc, l_errors_lin_no_bc)
        np.testing.assert_array_less(r_errors_lin_with_bc, r_errors_lin_no_bc)
        np.testing.assert_array_less(l_errors_quad_with_bc, l_errors_quad_no_bc)
        np.testing.assert_array_less(r_errors_quad_with_bc, r_errors_quad_no_bc)

        # note that with bcs we now obtain the left Dirichlet condition exactly

        r_lin_rates_bc = np.log(
            r_errors_lin_with_bc[:-1] / r_errors_lin_with_bc[1:]
        ) / np.log(dx[:-1] / dx[1:])
        r_quad_rates_bc = np.log(
            r_errors_quad_with_bc[:-1] / r_errors_quad_with_bc[1:]
        ) / np.log(dx[:-1] / dx[1:])

        # check convergence is about the correct order
        np.testing.assert_array_almost_equal(r_lin_rates_bc, 2, decimal=2)
        np.testing.assert_array_almost_equal(r_quad_rates_bc, 3, decimal=1)

    def test_extrapolation_with_bcs_left_neumann(self):
        # simple particle with a flux bc

        pts = 10 ** np.arange(1, 5, 1)
        dx = 1 / pts

        left_flux = 2
        right_val = 1

        def x_cubed(x):
            n = 3
            f_x = x ** n
            fp_l = 0
            f_r = 1
            y = f_x + (left_flux - fp_l) * x + (right_val - f_r - left_flux + fp_l)

            true_left = right_val - f_r - left_flux + fp_l
            true_right = right_val

            return y, true_left, true_right

        bcs = {"left": (left_flux, "Neumann"), "right": (right_val, "Dirichlet")}

        linear = {"extrapolation": {"order": "linear", "use bcs": True}}
        quad = {"extrapolation": {"order": "quadratic", "use bcs": True}}
        l_errors_lin_no_bc, r_errors_lin_no_bc = get_errors(x_cubed, linear, pts,)
        l_errors_quad_no_bc, r_errors_quad_no_bc = get_errors(x_cubed, quad, pts,)

        l_errors_lin_with_bc, r_errors_lin_with_bc = get_errors(
            x_cubed, linear, pts, bcs
        )
        l_errors_quad_with_bc, r_errors_quad_with_bc = get_errors(
            x_cubed, quad, pts, bcs
        )

        # test that with bc is better than without

        np.testing.assert_array_less(l_errors_lin_with_bc, l_errors_lin_no_bc)
        np.testing.assert_array_less(r_errors_lin_with_bc, r_errors_lin_no_bc)
        np.testing.assert_array_less(l_errors_quad_with_bc, l_errors_quad_no_bc)
        np.testing.assert_array_less(r_errors_quad_with_bc, r_errors_quad_no_bc)

        # note that with bcs we now obtain the right Dirichlet condition exactly

        l_lin_rates_bc = np.log(
            l_errors_lin_with_bc[:-1] / l_errors_lin_with_bc[1:]
        ) / np.log(dx[:-1] / dx[1:])
        l_quad_rates_bc = np.log(
            l_errors_quad_with_bc[:-1] / l_errors_quad_with_bc[1:]
        ) / np.log(dx[:-1] / dx[1:])

        # check convergence is about the correct order
        np.testing.assert_array_less(2, l_lin_rates_bc)
        np.testing.assert_array_almost_equal(l_quad_rates_bc, 3, decimal=1)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()

