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


def errors(pts, function, extrap):

    domain = "test"
    x = pybamm.SpatialVariable("x", domain=domain)
    geometry = {
        domain: {"primary": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}
    }
    submesh_types = {domain: pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)}
    var_pts = {x: pts}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

    spatial_methods = {"test": pybamm.FiniteVolume}
    disc = pybamm.Discretisation(mesh, spatial_methods)

    disc.spatial_methods["test"].extrapolation = extrap
    var = pybamm.Variable("var", domain="test")
    left_extrap = pybamm.BoundaryValue(var, "left")
    right_extrap = pybamm.BoundaryValue(var, "right")

    submesh = mesh["test"]
    y, l_true, r_true = function(submesh[0].nodes)

    disc.set_variable_slices([var])
    left_extrap_processed = disc.process_symbol(left_extrap)
    right_extrap_processed = disc.process_symbol(right_extrap)

    l_error = np.abs(l_true - left_extrap_processed.evaluate(None, y))
    r_error = np.abs(r_true - right_extrap_processed.evaluate(None, y))

    return l_error, r_error


def get_errors(function, extrap, pts):

    l_errors = np.zeros(pts.shape)
    r_errors = np.zeros(pts.shape)

    for i, pt in enumerate(pts):
        l_errors[i], r_errors[i] = errors(pt, function, extrap)

    return l_errors, r_errors


class TestExtrapolation(unittest.TestCase):
    def test_quadratic_convergence(self):

        # all tests are performed on x in [0, 1]

        def x_squared(x):
            y = x ** 2
            l_true = 0
            r_true = 1
            return y, l_true, r_true

        pts = 10 ** np.arange(1, 6, 1)
        dx = 1 / pts
        l_errors_lin, r_errors_lin = get_errors(x_squared, "linear", pts)
        l_errors_quad, r_errors_quad = get_errors(x_squared, "quadratic", pts)

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

        l_errors_lin, r_errors_lin = get_errors(x_squared, "linear", pts)

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
        l_errors_quad, r_errors_quad = get_errors(x_cubed, "quadratic", pts)

        l_quad_rates = np.log(l_errors_quad[:-1] / l_errors_quad[1:]) / np.log(
            dx[:-1] / dx[1:]
        )

        r_quad_rates = np.log(r_errors_quad[:-1] / r_errors_quad[1:]) / np.log(
            dx[:-1] / dx[1:]
        )

        np.testing.assert_array_almost_equal(l_quad_rates, 3)
        np.testing.assert_array_almost_equal(r_quad_rates, 3, decimal=3)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()

