#
# Test for the extrapolations in the finite volume class
#

import numpy as np
import pytest

import pybamm
from tests import (
    get_1p1d_mesh_for_testing,
    get_mesh_for_testing,
    get_mesh_for_testing_symbolic,
    get_p2d_mesh_for_testing,
)


def errors(
    pts, function, method_options, bcs=None, submesh_type=pybamm.Uniform1DSubMesh
):
    domain = "test"
    x = pybamm.SpatialVariable("x", domain=domain)
    geometry = {domain: {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}
    submesh_types = {domain: submesh_type}
    var_pts = {x: pts}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

    spatial_methods = {"test": pybamm.FiniteVolume(method_options)}
    disc = pybamm.Discretisation(mesh, spatial_methods)

    var = pybamm.Variable("var", domain="test")
    left_extrap = pybamm.BoundaryValue(var, "left")
    right_extrap = pybamm.BoundaryValue(var, "right")
    left_grad = pybamm.BoundaryGradient(var, "left")
    right_grad = pybamm.BoundaryGradient(var, "right")

    if bcs:
        model = pybamm.BaseBatteryModel()
        bc_dict = {var: bcs}
        model.boundary_conditions = bc_dict
        disc.bcs = disc.process_boundary_conditions(model)
        # Note that we will have to be careful to make sure to only use these when necessary.
        if bcs["left"][1] == "Neumann":
            l_true_grad = bcs["left"][0].evaluate(None, None)
        else:
            l_true_grad = 0
        if bcs["right"][1] == "Neumann":
            r_true_grad = bcs["right"][0].evaluate(None, None)
        else:
            r_true_grad = 0
    else:
        l_true_grad = 0
        r_true_grad = 0

    submesh = mesh["test"]
    y, l_true, r_true = function(submesh.nodes)

    disc.set_variable_slices([var])
    left_extrap_processed = disc.process_symbol(left_extrap)
    right_extrap_processed = disc.process_symbol(right_extrap)
    left_grad_processed = disc.process_symbol(left_grad)
    right_grad_processed = disc.process_symbol(right_grad)

    # address numpy 1.25 deprecation warning: array should have ndim=0 before conversion
    l_error = np.abs(l_true - left_extrap_processed.evaluate(None, y)).item()
    r_error = np.abs(r_true - right_extrap_processed.evaluate(None, y)).item()
    l_grad_error = np.abs(l_true_grad - left_grad_processed.evaluate(None, y)).item()
    r_grad_error = np.abs(r_true_grad - right_grad_processed.evaluate(None, y)).item()

    return l_error, r_error, l_grad_error, r_grad_error


def get_errors(
    function, method_options, pts, bcs=None, submesh_type=pybamm.Uniform1DSubMesh
):
    l_errors = np.zeros(pts.shape)
    r_errors = np.zeros(pts.shape)
    l_grad_errors = np.zeros(pts.shape)
    r_grad_errors = np.zeros(pts.shape)

    for i, pt in enumerate(pts):
        l_errors[i], r_errors[i], l_grad_errors[i], r_grad_errors[i] = errors(
            pt, function, method_options, bcs, submesh_type
        )

    return l_errors, r_errors, l_grad_errors, r_grad_errors


class TestExtrapolation:
    def test_constant_extrapolation(self):
        linear = {
            "extrapolation": {"order": {"gradient": "linear", "value": "constant"}}
        }

        geometry = {"domain": {"x": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}
        submesh_types = {"domain": pybamm.Uniform1DSubMesh}
        var_pts = {"x": 10}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        spatial_methods_linear = {"domain": pybamm.FiniteVolume(linear)}

        disc_linear = pybamm.Discretisation(mesh, spatial_methods_linear)

        var = pybamm.Variable("var", domain="domain")
        left_extrap_linear = pybamm.BoundaryValue(var, "left")
        right_extrap_linear = pybamm.BoundaryValue(var, "right")

        disc_linear.set_variable_slices([var])

        submesh = mesh["domain"]
        y = submesh.nodes

        left_extrap_linear_disc = disc_linear.process_symbol(left_extrap_linear)
        right_extrap_linear_disc = disc_linear.process_symbol(right_extrap_linear)

        np.testing.assert_allclose(
            left_extrap_linear_disc.evaluate(None, y), y[0], rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            right_extrap_linear_disc.evaluate(None, y), y[-1], rtol=1e-7, atol=1e-6
        )

    def test_order_override(self):
        linear = {"extrapolation": {"order": {"gradient": "linear", "value": "linear"}}}

        geometry = {"domain": {"x": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}
        submesh_types = {"domain": pybamm.Uniform1DSubMesh}
        var_pts = {"x": 10}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        spatial_methods_linear = {"domain": pybamm.FiniteVolume(linear)}

        disc_linear = pybamm.Discretisation(mesh, spatial_methods_linear)

        var = pybamm.Variable("var", domain="domain")
        left_extrap_cubic = pybamm.BoundaryValue(var, "left", order="cubic")
        left_grad_cubic = pybamm.BoundaryGradient(var, "left", order="cubic")

        disc_linear.set_variable_slices([var])

        with pytest.raises(NotImplementedError):
            disc_linear.process_symbol(left_extrap_cubic)
        with pytest.raises(NotImplementedError):
            disc_linear.process_symbol(left_grad_cubic)

    def test_raises_error_for_high_order_extrapolation(self):
        value_cubic = {
            "extrapolation": {"order": {"gradient": "linear", "value": "cubic"}}
        }
        gradient_cubic = {
            "extrapolation": {"order": {"gradient": "cubic", "value": "linear"}}
        }

        geometry = {"domain": {"x": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}
        submesh_types = {"domain": pybamm.Uniform1DSubMesh}
        var_pts = {"x": 10}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        spatial_methods_value_cubic = {"domain": pybamm.FiniteVolume(value_cubic)}
        spatial_methods_gradient_cubic = {"domain": pybamm.FiniteVolume(gradient_cubic)}

        disc_value_cubic = pybamm.Discretisation(mesh, spatial_methods_value_cubic)
        disc_gradient_cubic = pybamm.Discretisation(
            mesh, spatial_methods_gradient_cubic
        )

        var = pybamm.Variable("var", domain="domain")
        left_extrap = pybamm.BoundaryValue(var, "left")
        right_extrap = pybamm.BoundaryValue(var, "right")
        left_grad = pybamm.BoundaryGradient(var, "left")
        right_grad = pybamm.BoundaryGradient(var, "right")

        disc_value_cubic.set_variable_slices([var])
        disc_gradient_cubic.set_variable_slices([var])

        with pytest.raises(NotImplementedError):
            disc_value_cubic.process_symbol(left_extrap)
        with pytest.raises(NotImplementedError):
            disc_value_cubic.process_symbol(right_extrap)
        with pytest.raises(NotImplementedError):
            disc_gradient_cubic.process_symbol(left_grad)
        with pytest.raises(NotImplementedError):
            disc_gradient_cubic.process_symbol(right_grad)

    def test_convergence_without_bcs(self):
        # all tests are performed on x in [0, 1]
        linear = {"extrapolation": {"order": {"gradient": "linear", "value": "linear"}}}
        quad = {
            "extrapolation": {"order": {"gradient": "quadratic", "value": "quadratic"}}
        }

        def x_squared(x):
            y = x**2
            l_true = 0
            r_true = 1
            return y, l_true, r_true

        pts = 10 ** np.arange(1, 6, 1)
        dx = 1 / pts

        l_errors_lin, r_errors_lin, _l_grad_errors_lin, _r_grad_errors_lin = get_errors(
            x_squared, linear, pts
        )
        l_errors_quad, r_errors_quad, _l_grad_errors_quad, _r_grad_errors_quad = (
            get_errors(x_squared, quad, pts)
        )

        l_lin_rates = np.log(l_errors_lin[:-1] / l_errors_lin[1:]) / np.log(
            dx[:-1] / dx[1:]
        )

        r_lin_rates = np.log(r_errors_lin[:-1] / r_errors_lin[1:]) / np.log(
            dx[:-1] / dx[1:]
        )
        np.testing.assert_allclose(l_lin_rates, 2, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(r_lin_rates, 2, rtol=1e-7, atol=1e-6)

        # check quadratic is equal up to machine precision
        np.testing.assert_allclose(l_errors_quad, 0, rtol=1e-15, atol=1e-14)
        np.testing.assert_allclose(r_errors_quad, 0, rtol=1e-15, atol=1e-14)

        def x_cubed(x):
            y = x**3
            l_true = 0
            r_true = 1
            return y, l_true, r_true

        l_errors_lin, r_errors_lin, _l_grad_errors_lin, _r_grad_errors_lin = get_errors(
            x_squared, linear, pts
        )

        l_lin_rates = np.log(l_errors_lin[:-1] / l_errors_lin[1:]) / np.log(
            dx[:-1] / dx[1:]
        )

        r_lin_rates = np.log(r_errors_lin[:-1] / r_errors_lin[1:]) / np.log(
            dx[:-1] / dx[1:]
        )

        np.testing.assert_allclose(l_lin_rates, 2, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(r_lin_rates, 2, rtol=1e-7, atol=1e-6)

        # quadratic case
        pts = 5 ** np.arange(1, 7, 1)
        dx = 1 / pts
        l_errors_quad, r_errors_quad, _l_grad_errors_quad, _r_grad_errors_quad = (
            get_errors(x_cubed, quad, pts)
        )

        l_quad_rates = np.log(l_errors_quad[:-1] / l_errors_quad[1:]) / np.log(
            dx[:-1] / dx[1:]
        )

        r_quad_rates = np.log(r_errors_quad[:-1] / r_errors_quad[1:]) / np.log(
            dx[:-1] / dx[1:]
        )

        np.testing.assert_allclose(l_quad_rates, 3, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(r_quad_rates, 3, rtol=1e-4, atol=1e-3)

    def test_extrapolation_with_bcs_right_neumann(self):
        # simple particle with a flux bc

        pts = 10 ** np.arange(1, 6, 1)
        dx = 1 / pts

        left_val = 1
        right_flux = 2

        def x_cubed(x):
            n = 3
            f_x = x**n
            f_l = 0
            fp_r = n
            y = f_x + (right_flux - fp_r) * x + (left_val - f_l)

            true_left = left_val
            true_right = 1 + (right_flux - fp_r) + (left_val - f_l)

            return y, true_left, true_right

        bcs = {"left": (left_val, "Dirichlet"), "right": (right_flux, "Neumann")}

        linear = {
            "extrapolation": {
                "order": {"gradient": "linear", "value": "linear"},
                "use bcs": True,
            }
        }
        quad = {
            "extrapolation": {
                "order": {"gradient": "quadratic", "value": "quadratic"},
                "use bcs": True,
            }
        }
        for submesh_type in [pybamm.Uniform1DSubMesh, pybamm.SymbolicUniform1DSubMesh]:
            (
                l_errors_lin_no_bc,
                r_errors_lin_no_bc,
                _l_grad_errors_lin_no_bc,
                _r_grad_errors_lin_no_bc,
            ) = get_errors(x_cubed, linear, pts, submesh_type=submesh_type)
            (
                l_errors_quad_no_bc,
                r_errors_quad_no_bc,
                _l_grad_errors_quad_no_bc,
                _r_grad_errors_quad_no_bc,
            ) = get_errors(x_cubed, quad, pts, submesh_type=submesh_type)

            (
                l_errors_lin_with_bc,
                r_errors_lin_with_bc,
                _l_grad_errors_lin_with_bc,
                r_grad_errors_lin_with_bc,
            ) = get_errors(x_cubed, linear, pts, bcs, submesh_type=submesh_type)
            (
                l_errors_quad_with_bc,
                r_errors_quad_with_bc,
                _l_grad_errors_quad_with_bc,
                r_grad_errors_quad_with_bc,
            ) = get_errors(x_cubed, quad, pts, bcs, submesh_type=submesh_type)

            # test that with bc is better than without

            np.testing.assert_array_less(l_errors_lin_with_bc, l_errors_lin_no_bc)
            np.testing.assert_array_less(r_errors_lin_with_bc, r_errors_lin_no_bc)
            np.testing.assert_array_less(l_errors_quad_with_bc, l_errors_quad_no_bc)
            np.testing.assert_array_less(r_errors_quad_with_bc, r_errors_quad_no_bc)

            # Test that the RIGHT gradient is correct
            np.testing.assert_allclose(
                r_grad_errors_lin_with_bc, 0, rtol=1e-7, atol=1e-6
            )
            np.testing.assert_allclose(
                r_grad_errors_quad_with_bc, 0, rtol=1e-7, atol=1e-6
            )

            # note that with bcs we now obtain the left Dirichlet condition exactly

            r_lin_rates_bc = np.log(
                r_errors_lin_with_bc[:-1] / r_errors_lin_with_bc[1:]
            ) / np.log(dx[:-1] / dx[1:])
            r_quad_rates_bc = np.log(
                r_errors_quad_with_bc[:-1] / r_errors_quad_with_bc[1:]
            ) / np.log(dx[:-1] / dx[1:])

            # check convergence is about the correct order
            np.testing.assert_allclose(r_lin_rates_bc, 2, rtol=1e-3, atol=1e-2)
            np.testing.assert_allclose(r_quad_rates_bc, 3, rtol=1e-2, atol=1e-1)

    def test_extrapolation_with_bcs_left_neumann(self):
        # simple particle with a flux bc

        pts = 10 ** np.arange(1, 5, 1)
        dx = 1 / pts

        left_flux = 2
        right_val = 1

        def x_cubed(x):
            n = 3
            f_x = x**n
            fp_l = 0
            f_r = 1
            y = f_x + (left_flux - fp_l) * x + (right_val - f_r - left_flux + fp_l)

            true_left = right_val - f_r - left_flux + fp_l
            true_right = right_val

            return y, true_left, true_right

        bcs = {"left": (left_flux, "Neumann"), "right": (right_val, "Dirichlet")}

        linear = {
            "extrapolation": {
                "order": {"gradient": "linear", "value": "linear"},
                "use bcs": True,
            }
        }
        quad = {
            "extrapolation": {
                "order": {"gradient": "quadratic", "value": "quadratic"},
                "use bcs": True,
            }
        }
        for submesh_type in [pybamm.Uniform1DSubMesh, pybamm.SymbolicUniform1DSubMesh]:
            (
                l_errors_lin_no_bc,
                r_errors_lin_no_bc,
                _l_grad_errors_lin_no_bc,
                _r_grad_errors_lin_no_bc,
            ) = get_errors(x_cubed, linear, pts, submesh_type=submesh_type)
            (
                l_errors_quad_no_bc,
                r_errors_quad_no_bc,
                _l_grad_errors_quad_no_bc,
                _r_grad_errors_quad_no_bc,
            ) = get_errors(x_cubed, quad, pts, submesh_type=submesh_type)

            (
                l_errors_lin_with_bc,
                r_errors_lin_with_bc,
                l_grad_errors_lin_with_bc,
                _r_grad_errors_lin_with_bc,
            ) = get_errors(x_cubed, linear, pts, bcs, submesh_type=submesh_type)
            (
                l_errors_quad_with_bc,
                r_errors_quad_with_bc,
                l_grad_errors_quad_with_bc,
                _r_grad_errors_quad_with_bc,
            ) = get_errors(x_cubed, quad, pts, bcs, submesh_type=submesh_type)

            # test that with bc is better than without

            np.testing.assert_array_less(l_errors_lin_with_bc, l_errors_lin_no_bc)
            np.testing.assert_array_less(r_errors_lin_with_bc, r_errors_lin_no_bc)
            np.testing.assert_array_less(l_errors_quad_with_bc, l_errors_quad_no_bc)
            np.testing.assert_array_less(r_errors_quad_with_bc, r_errors_quad_no_bc)

            # assert that the LEFT gradient is correct
            np.testing.assert_allclose(
                l_grad_errors_lin_with_bc, 0, rtol=1e-7, atol=1e-6
            )
            np.testing.assert_allclose(
                l_grad_errors_quad_with_bc, 0, rtol=1e-7, atol=1e-6
            )

            # note that with bcs we now obtain the right Dirichlet condition exactly

            l_lin_rates_bc = np.log(
                l_errors_lin_with_bc[:-1] / l_errors_lin_with_bc[1:]
            ) / np.log(dx[:-1] / dx[1:])
            l_quad_rates_bc = np.log(
                l_errors_quad_with_bc[:-1] / l_errors_quad_with_bc[1:]
            ) / np.log(dx[:-1] / dx[1:])

            # check convergence is about the correct order
            np.testing.assert_array_less(2, l_lin_rates_bc)
            np.testing.assert_allclose(l_quad_rates_bc, 3, rtol=1e-2, atol=1e-1)

    def test_linear_extrapolate_left_right(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        method_options = {
            "extrapolation": {
                "order": {"gradient": "linear", "value": "linear"},
                "use bcs": True,
            }
        }
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(method_options),
            "negative particle": pybamm.FiniteVolume(method_options),
            "current collector": pybamm.ZeroDimensionalSpatialMethod(method_options),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        macro_submesh = mesh[whole_cell]
        micro_submesh = mesh["negative particle"]

        # Macroscale
        # create variable
        var = pybamm.Variable("var", domain=whole_cell)
        # boundary value should work with something more complicated than a variable
        extrap_left = pybamm.BoundaryValue(2 * var, "left")
        extrap_right = pybamm.BoundaryValue(4 - var, "right")
        disc.set_variable_slices([var])
        extrap_left_disc = disc.process_symbol(extrap_left)
        extrap_right_disc = disc.process_symbol(extrap_right)

        # check constant extrapolates to constant
        constant_y = np.ones_like(macro_submesh.nodes[:, np.newaxis])
        assert extrap_left_disc.evaluate(None, constant_y) == 2
        assert extrap_right_disc.evaluate(None, constant_y) == 3

        # check linear variable extrapolates correctly
        linear_y = macro_submesh.nodes
        np.testing.assert_allclose(
            extrap_left_disc.evaluate(None, linear_y), 0, rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            extrap_right_disc.evaluate(None, linear_y), 3, rtol=1e-7, atol=1e-6
        )

        # Fluxes
        extrap_flux_left = pybamm.BoundaryGradient(2 * var, "left")
        extrap_flux_right = pybamm.BoundaryGradient(1 - var, "right")
        extrap_flux_left_disc = disc.process_symbol(extrap_flux_left)
        extrap_flux_right_disc = disc.process_symbol(extrap_flux_right)

        # check constant extrapolates to constant
        np.testing.assert_allclose(extrap_flux_left_disc.evaluate(y=constant_y), 0)
        np.testing.assert_allclose(extrap_flux_right_disc.evaluate(y=constant_y), 0)

        # check linear variable extrapolates correctly
        np.testing.assert_allclose(extrap_flux_left_disc.evaluate(y=linear_y), 2)
        np.testing.assert_allclose(extrap_flux_right_disc.evaluate(y=linear_y), -1)

        # Microscale
        # create variable
        var = pybamm.Variable("var", domain="negative particle")
        surf_eqn = pybamm.surf(var)
        disc.set_variable_slices([var])
        surf_eqn_disc = disc.process_symbol(surf_eqn)

        # check constant extrapolates to constant
        constant_y = np.ones_like(micro_submesh.nodes[:, np.newaxis])
        assert surf_eqn_disc.evaluate(None, constant_y) == 1.0

        # check linear variable extrapolates correctly
        linear_y = micro_submesh.nodes
        y_surf = micro_submesh.edges[-1]
        np.testing.assert_allclose(
            surf_eqn_disc.evaluate(None, linear_y), y_surf, rtol=1e-7, atol=1e-6
        )

    def test_extrapolate_symbolic(self):
        mesh = get_mesh_for_testing_symbolic()
        method_options = {
            "extrapolation": {
                "order": {"gradient": "linear", "value": "linear"},
                "use bcs": False,
            }
        }
        spatial_methods = {
            "domain": pybamm.FiniteVolume(method_options),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var = pybamm.Variable("var", domain="domain")
        extrap_left = pybamm.BoundaryValue(var, "left")
        extrap_right = pybamm.BoundaryValue(var, "right")
        extrap_grad_left = pybamm.BoundaryGradient(var, "left")
        extrap_grad_right = pybamm.BoundaryGradient(var, "right")
        disc.set_variable_slices([var])
        extrap_left_disc = disc.process_symbol(extrap_left)
        extrap_right_disc = disc.process_symbol(extrap_right)
        extrap_grad_left_disc = disc.process_symbol(extrap_grad_left)
        extrap_grad_right_disc = disc.process_symbol(extrap_grad_right)

        # check constant extrapolates to constant
        constant_y = np.ones_like(mesh["domain"].nodes[:, np.newaxis])
        assert extrap_left_disc.evaluate(None, constant_y) == 1
        assert extrap_right_disc.evaluate(None, constant_y) == 1

        # check linear variable extrapolates correctly
        linear_y = mesh["domain"].nodes
        np.testing.assert_allclose(
            extrap_left_disc.evaluate(None, linear_y), 0, rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            extrap_right_disc.evaluate(None, linear_y), 1, rtol=1e-7, atol=1e-6
        )

        # check gradient extrapolates correctly
        np.testing.assert_allclose(
            extrap_grad_left_disc.evaluate(None, linear_y), 1 / 2, rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            extrap_grad_right_disc.evaluate(None, linear_y), 1 / 2, rtol=1e-7, atol=1e-6
        )

        method_options = {
            "extrapolation": {
                "order": {"gradient": "quadratic", "value": "quadratic"},
                "use bcs": False,
            }
        }
        spatial_methods = {"domain": pybamm.FiniteVolume(method_options)}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="domain")
        extrap_left = pybamm.BoundaryValue(var, "left")
        extrap_right = pybamm.BoundaryValue(var, "right")
        extrap_grad_left = pybamm.BoundaryGradient(var, "left")
        extrap_grad_right = pybamm.BoundaryGradient(var, "right")
        disc.set_variable_slices([var])
        extrap_left_disc = disc.process_symbol(extrap_left)
        extrap_right_disc = disc.process_symbol(extrap_right)
        extrap_grad_left_disc = disc.process_symbol(extrap_grad_left)
        extrap_grad_right_disc = disc.process_symbol(extrap_grad_right)

        # check constant extrapolates to constant
        np.testing.assert_allclose(
            extrap_left_disc.evaluate(None, constant_y), 1, rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            extrap_right_disc.evaluate(None, constant_y), 1, rtol=1e-7, atol=1e-6
        )

        # check linear variable extrapolates correctly
        np.testing.assert_allclose(
            extrap_left_disc.evaluate(None, linear_y), 0, rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            extrap_right_disc.evaluate(None, linear_y), 1, rtol=1e-7, atol=1e-6
        )

        # check gradient extrapolates correctly
        np.testing.assert_allclose(
            extrap_grad_left_disc.evaluate(None, linear_y), 1 / 2, rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            extrap_grad_right_disc.evaluate(None, linear_y), 1 / 2, rtol=1e-7, atol=1e-6
        )

    def test_quadratic_extrapolate_left_right(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        method_options = {
            "extrapolation": {
                "order": {"gradient": "quadratic", "value": "quadratic"},
                "use bcs": False,
            }
        }
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(method_options),
            "negative particle": pybamm.FiniteVolume(method_options),
            "current collector": pybamm.ZeroDimensionalSpatialMethod(method_options),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        macro_submesh = mesh[whole_cell]
        micro_submesh = mesh["negative particle"]

        # Macroscale
        # create variable
        var = pybamm.Variable("var", domain=whole_cell)
        # boundary value should work with something more complicated than a variable
        extrap_left = pybamm.BoundaryValue(2 * var, "left")
        extrap_right = pybamm.BoundaryValue(4 - var, "right")
        disc.set_variable_slices([var])
        extrap_left_disc = disc.process_symbol(extrap_left)
        extrap_right_disc = disc.process_symbol(extrap_right)

        # check constant extrapolates to constant
        constant_y = np.ones_like(macro_submesh.nodes[:, np.newaxis])
        np.testing.assert_allclose(
            extrap_left_disc.evaluate(None, constant_y), 2.0, rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            extrap_right_disc.evaluate(None, constant_y), 3.0, rtol=1e-7, atol=1e-6
        )

        # check linear variable extrapolates correctly
        linear_y = macro_submesh.nodes
        np.testing.assert_allclose(
            extrap_left_disc.evaluate(None, linear_y), 0, rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            extrap_right_disc.evaluate(None, linear_y), 3, rtol=1e-7, atol=1e-6
        )

        # Fluxes
        extrap_flux_left = pybamm.BoundaryGradient(2 * var, "left")
        extrap_flux_right = pybamm.BoundaryGradient(1 - var, "right")
        extrap_flux_left_disc = disc.process_symbol(extrap_flux_left)
        extrap_flux_right_disc = disc.process_symbol(extrap_flux_right)

        # check constant extrapolates to constant
        np.testing.assert_allclose(
            extrap_flux_left_disc.evaluate(None, constant_y), 0, rtol=1e-7, atol=1e-6
        )
        assert extrap_flux_right_disc.evaluate(None, constant_y) == 0

        # check linear variable extrapolates correctly
        np.testing.assert_allclose(
            extrap_flux_left_disc.evaluate(None, linear_y), 2, rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            extrap_flux_right_disc.evaluate(None, linear_y), -1, rtol=1e-7, atol=1e-6
        )

        # Microscale
        # create variable
        var = pybamm.Variable("var", domain="negative particle")
        surf_eqn = pybamm.surf(var)
        disc.set_variable_slices([var])
        surf_eqn_disc = disc.process_symbol(surf_eqn)

        # check constant extrapolates to constant
        constant_y = np.ones_like(micro_submesh.nodes[:, np.newaxis])
        np.testing.assert_allclose(
            surf_eqn_disc.evaluate(None, constant_y), 1, rtol=1e-7, atol=1e-6
        )

        # check linear variable extrapolates correctly
        linear_y = micro_submesh.nodes
        y_surf = micro_submesh.edges[-1]
        np.testing.assert_allclose(
            surf_eqn_disc.evaluate(None, linear_y), y_surf, rtol=1e-7, atol=1e-6
        )

    def test_extrapolate_on_nonuniform_grid(self):
        geometry = {
            "negative particle": {"r_n": {"min": 0, "max": 1}},
            "positive particle": {"r_p": {"min": 0, "max": 1}},
        }

        submesh_types = {
            "negative particle": pybamm.MeshGenerator(pybamm.Exponential1DSubMesh),
            "positive particle": pybamm.MeshGenerator(pybamm.Exponential1DSubMesh),
        }

        rpts = 10
        var_pts = {"r_n": rpts, "r_p": rpts}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        method_options = {
            "extrapolation": {
                "order": {"gradient": "linear", "value": "linear"},
                "use bcs": False,
            }
        }
        spatial_methods = {"negative particle": pybamm.FiniteVolume(method_options)}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var = pybamm.Variable("var", domain="negative particle")
        surf_eqn = pybamm.surf(var)
        disc.set_variable_slices([var])
        surf_eqn_disc = disc.process_symbol(surf_eqn)

        micro_submesh = mesh["negative particle"]

        # check constant extrapolates to constant
        constant_y = np.ones_like(micro_submesh.nodes[:, np.newaxis])
        np.testing.assert_allclose(
            surf_eqn_disc.evaluate(None, constant_y), 1, rtol=1e-7, atol=1e-6
        )

        # check linear variable extrapolates correctly
        linear_y = micro_submesh.nodes
        y_surf = micro_submesh.edges[-1]
        np.testing.assert_allclose(
            surf_eqn_disc.evaluate(None, linear_y), y_surf, rtol=1e-7, atol=1e-6
        )

    def test_extrapolate_2d_models(self):
        # create discretisation
        mesh = get_p2d_mesh_for_testing()
        method_options = {
            "extrapolation": {
                "order": {"gradient": "linear", "value": "linear"},
                "use bcs": False,
            }
        }
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(method_options),
            "negative particle": pybamm.FiniteVolume(method_options),
            "positive particle": pybamm.FiniteVolume(method_options),
            "current collector": pybamm.FiniteVolume(method_options),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Microscale
        var = pybamm.Variable(
            "var",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        extrap_right = pybamm.BoundaryValue(var, "right")
        disc.set_variable_slices([var])
        extrap_right_disc = disc.process_symbol(extrap_right)
        assert extrap_right_disc.domain == ["negative electrode"]
        # evaluate
        y_macro = mesh["negative electrode"].nodes
        y_micro = mesh["negative particle"].nodes
        y = np.outer(y_macro, y_micro).reshape(-1, 1)
        # extrapolate to r=0.5 --> should evaluate to 0.5*y_macro
        np.testing.assert_allclose(
            extrap_right_disc.evaluate(y=y)[:, 0], 0.5 * y_macro, rtol=1e-7, atol=1e-6
        )

        var = pybamm.Variable("var", domain="positive particle")
        extrap_right = pybamm.BoundaryValue(var, "right")
        disc.set_variable_slices([var])
        extrap_right_disc = disc.process_symbol(extrap_right)
        assert extrap_right_disc.domain == []

        # 2d macroscale
        mesh = get_1p1d_mesh_for_testing()
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="negative electrode")
        extrap_right = pybamm.BoundaryValue(var, "right")
        disc.set_variable_slices([var])
        extrap_right_disc = disc.process_symbol(extrap_right)
        assert extrap_right_disc.domain == []

        # test extrapolate to "negative tab" gives same as "left" and
        # "positive tab" gives same "right" (see get_mesh_for_testing)
        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])
        submesh = mesh["current collector"]
        constant_y = np.ones_like(submesh.nodes[:, np.newaxis])

        extrap_neg = pybamm.BoundaryValue(var, "negative tab")
        extrap_neg_disc = disc.process_symbol(extrap_neg)
        extrap_left = pybamm.BoundaryValue(var, "left")
        extrap_left_disc = disc.process_symbol(extrap_left)
        np.testing.assert_array_equal(
            extrap_neg_disc.evaluate(None, constant_y),
            extrap_left_disc.evaluate(None, constant_y),
        )

        extrap_pos = pybamm.BoundaryValue(var, "positive tab")
        extrap_pos_disc = disc.process_symbol(extrap_pos)
        extrap_right = pybamm.BoundaryValue(var, "right")
        extrap_right_disc = disc.process_symbol(extrap_right)
        np.testing.assert_array_equal(
            extrap_pos_disc.evaluate(None, constant_y),
            extrap_right_disc.evaluate(None, constant_y),
        )

    def test_boundary_mesh_size(self):
        """Test BoundaryMeshSize functionality."""
        # Create a simple 1D geometry
        geometry = {"domain": {"x": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}
        submesh_types = {"domain": pybamm.Uniform1DSubMesh}
        var_pts = {"x": 5}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        method_options = {
            "extrapolation": {
                "order": {"gradient": "linear", "value": "linear"},
                "use bcs": False,
            }
        }
        spatial_methods = {"domain": pybamm.FiniteVolume(method_options)}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Create a variable
        var = pybamm.Variable("var", domain="domain")

        # Test left boundary mesh size
        left_mesh_size = pybamm.BoundaryMeshSize(var, "left")
        right_mesh_size = pybamm.BoundaryMeshSize(var, "right")

        disc.set_variable_slices([var])
        left_mesh_size_disc = disc.process_symbol(left_mesh_size)
        right_mesh_size_disc = disc.process_symbol(right_mesh_size)

        # Get the submesh for direct comparison
        submesh = mesh["domain"]

        # Calculate expected values
        expected_left = submesh.d_nodes[0]
        expected_right = submesh.d_nodes[-1]

        # Test that the discretized symbols return the correct values
        np.testing.assert_allclose(
            left_mesh_size_disc.evaluate(), expected_left, rtol=1e-15, atol=1e-15
        )
        np.testing.assert_allclose(
            right_mesh_size_disc.evaluate(), expected_right, rtol=1e-15, atol=1e-15
        )

        # Test with different mesh sizes
        for npts in [10, 20, 50]:
            var_pts = {"x": npts}
            mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
            spatial_methods = {"domain": pybamm.FiniteVolume(method_options)}
            disc = pybamm.Discretisation(mesh, spatial_methods)

            var = pybamm.Variable("var", domain="domain")
            left_mesh_size = pybamm.BoundaryMeshSize(var, "left")
            right_mesh_size = pybamm.BoundaryMeshSize(var, "right")

            disc.set_variable_slices([var])
            left_mesh_size_disc = disc.process_symbol(left_mesh_size)
            right_mesh_size_disc = disc.process_symbol(right_mesh_size)

            submesh = mesh["domain"]
            expected_left = submesh.d_nodes[0]
            expected_right = submesh.d_nodes[-1]

            np.testing.assert_allclose(
                left_mesh_size_disc.evaluate(), expected_left, rtol=1e-15, atol=1e-15
            )
            np.testing.assert_allclose(
                right_mesh_size_disc.evaluate(), expected_right, rtol=1e-15, atol=1e-15
            )

        # Test with non-uniform mesh
        geometry_nonuniform = {
            "negative particle": {"r_n": {"min": 0, "max": 1}},
        }
        submesh_types_nonuniform = {
            "negative particle": pybamm.MeshGenerator(pybamm.Exponential1DSubMesh),
        }
        var_pts_nonuniform = {"r_n": 10}
        mesh_nonuniform = pybamm.Mesh(
            geometry_nonuniform, submesh_types_nonuniform, var_pts_nonuniform
        )

        spatial_methods_nonuniform = {
            "negative particle": pybamm.FiniteVolume(method_options)
        }
        disc_nonuniform = pybamm.Discretisation(
            mesh_nonuniform, spatial_methods_nonuniform
        )

        var_nonuniform = pybamm.Variable("var", domain="negative particle")
        left_mesh_size_nonuniform = pybamm.BoundaryMeshSize(var_nonuniform, "left")
        right_mesh_size_nonuniform = pybamm.BoundaryMeshSize(var_nonuniform, "right")

        disc_nonuniform.set_variable_slices([var_nonuniform])
        left_mesh_size_disc_nonuniform = disc_nonuniform.process_symbol(
            left_mesh_size_nonuniform
        )
        right_mesh_size_disc_nonuniform = disc_nonuniform.process_symbol(
            right_mesh_size_nonuniform
        )

        submesh_nonuniform = mesh_nonuniform["negative particle"]
        expected_left_nonuniform = submesh_nonuniform.d_nodes[0]
        expected_right_nonuniform = submesh_nonuniform.d_nodes[-1]

        np.testing.assert_allclose(
            left_mesh_size_disc_nonuniform.evaluate(),
            expected_left_nonuniform,
            rtol=1e-15,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            right_mesh_size_disc_nonuniform.evaluate(),
            expected_right_nonuniform,
            rtol=1e-15,
            atol=1e-15,
        )

        # Test with symbolic submesh
        geometry_symbolic = {
            "domain": {"x": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }
        submesh_types_symbolic = {"domain": pybamm.SymbolicUniform1DSubMesh}
        var_pts_symbolic = {"x": 8}
        mesh_symbolic = pybamm.Mesh(
            geometry_symbolic, submesh_types_symbolic, var_pts_symbolic
        )

        spatial_methods_symbolic = {"domain": pybamm.FiniteVolume(method_options)}
        disc_symbolic = pybamm.Discretisation(mesh_symbolic, spatial_methods_symbolic)

        var_symbolic = pybamm.Variable("var", domain="domain")
        left_mesh_size_symbolic = pybamm.BoundaryMeshSize(var_symbolic, "left")
        right_mesh_size_symbolic = pybamm.BoundaryMeshSize(var_symbolic, "right")

        disc_symbolic.set_variable_slices([var_symbolic])
        left_mesh_size_disc_symbolic = disc_symbolic.process_symbol(
            left_mesh_size_symbolic
        )
        right_mesh_size_disc_symbolic = disc_symbolic.process_symbol(
            right_mesh_size_symbolic
        )

        submesh_symbolic = mesh_symbolic["domain"]
        expected_left_symbolic = submesh_symbolic.d_nodes[0]
        expected_right_symbolic = submesh_symbolic.d_nodes[-1]

        np.testing.assert_allclose(
            left_mesh_size_disc_symbolic.evaluate(),
            expected_left_symbolic,
            rtol=1e-15,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            right_mesh_size_disc_symbolic.evaluate(),
            expected_right_symbolic,
            rtol=1e-15,
            atol=1e-15,
        )

        # Test error handling for invalid side
        var_error = pybamm.Variable("var", domain="domain")
        with pytest.raises(ValueError, match=r"Invalid side"):
            invalid_mesh_size = pybamm.BoundaryMeshSize(var_error, "invalid")
            disc.set_variable_slices([var_error])
            disc.process_symbol(invalid_mesh_size)

        # Test that the BoundaryMeshSize symbol has the correct properties
        left_mesh_size_symbol = pybamm.BoundaryMeshSize(var, "left")
        right_mesh_size_symbol = pybamm.BoundaryMeshSize(var, "right")

        # Check that it's a BoundaryMeshSize instance
        assert isinstance(left_mesh_size_symbol, pybamm.BoundaryMeshSize)
        assert isinstance(right_mesh_size_symbol, pybamm.BoundaryMeshSize)

        # Check side property
        assert left_mesh_size_symbol.side == "left"
        assert right_mesh_size_symbol.side == "right"

        # Check child property
        assert left_mesh_size_symbol.child == var
        assert right_mesh_size_symbol.child == var
