#
# Tests for integration using Finite Volume method
#

import numpy as np
import pytest

import pybamm
from tests.shared import get_mesh_for_testing_2d


class TestFiniteVolumeIntegration:
    def test_indefinite_integral(self):
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {"macroscale": pybamm.FiniteVolume2D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable(
            "var", domain=["negative electrode", "separator", "positive electrode"]
        )
        disc.set_variable_slices([var])
        x = pybamm.SpatialVariable(
            "x",
            domain=["negative electrode", "separator", "positive electrode"],
            direction="lr",
        )
        indef_integral = pybamm.IndefiniteIntegral(var, x)
        with pytest.raises(NotImplementedError):
            disc.process_symbol(indef_integral)
        spatial_method = pybamm.FiniteVolume2D()
        with pytest.raises(NotImplementedError):
            spatial_method.indefinite_integral_matrix_edges(
                "negative electrode", "left"
            )
        with pytest.raises(NotImplementedError):
            spatial_method.indefinite_integral_matrix_nodes(
                "negative electrode", "left"
            )

    def test_definite_integral(self):
        # create discretisation
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume2D(),
            "separator": pybamm.FiniteVolume2D(),
            "positive electrode": pybamm.FiniteVolume2D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        # lengths
        ln = mesh["negative electrode"].edges_lr[-1]
        ls = mesh["separator"].edges_lr[-1] - ln
        lp = mesh["positive electrode"].edges_lr[-1] - (ln + ls)
        l_tb = mesh["negative electrode"].edges_tb[-1]

        # macroscale variable (lr integration)
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable(
            "x", ["negative electrode", "separator"], direction="lr"
        )
        integral_eqn = pybamm.Integral(var, x)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)
        submesh = mesh[("negative electrode", "separator")]
        constant_y = np.ones(submesh.npts_lr * submesh.npts_tb)
        assert (integral_eqn_disc.evaluate(None, constant_y) == ln + ls).all()

        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        z = pybamm.SpatialVariable(
            "z", ["negative electrode", "separator"], direction="tb"
        )
        integral_eqn_tb = pybamm.Integral(var, z)
        disc.set_variable_slices([var])
        integral_eqn_disc_tb = disc.process_symbol(integral_eqn_tb)
        submesh = mesh[("negative electrode", "separator")]
        constant_y = np.ones(submesh.npts_lr * submesh.npts_tb)
        np.testing.assert_allclose(
            integral_eqn_disc_tb.evaluate(None, constant_y), l_tb
        )

        var = pybamm.Variable(
            "var", domain=["negative electrode", "separator", "positive electrode"]
        )
        x = pybamm.SpatialVariable(
            "x",
            ["negative electrode", "separator", "positive electrode"],
            direction="lr",
        )
        integral_eqn = pybamm.Integral(var, x)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)
        submesh = mesh[("negative electrode", "separator", "positive electrode")]
        LR, _TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        lr = LR.flatten()
        np.testing.assert_allclose(
            integral_eqn_disc.evaluate(None, lr),
            (ln + ls + lp) ** 2 / 2,
            rtol=1e-7,
            atol=1e-6,
        )

        spatial_method = pybamm.FiniteVolume2D()
        child = pybamm.Variable(
            "child",
            domain=["negative electrode", "separator", "positive electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        spatial_method.build(mesh)
        with pytest.raises(
            ValueError,
            match=r"Integration variable must be provided for 2D integration",
        ):
            spatial_method.definite_integral_matrix(child)

    def test_area_integral(self):
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {"macroscale": pybamm.FiniteVolume2D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable(
            "var", domain=["negative electrode", "separator", "positive electrode"]
        )
        disc.set_variable_slices([var])
        x = pybamm.SpatialVariable(
            "x",
            ["negative electrode", "separator", "positive electrode"],
            direction="lr",
        )
        z = pybamm.SpatialVariable(
            "z",
            ["negative electrode", "separator", "positive electrode"],
            direction="tb",
        )
        symbol_x_z = pybamm.Integral(var, [z, x])
        symbol_z_x = pybamm.Integral(var, [x, z])
        disc_symbol_x_z = disc.process_symbol(symbol_x_z)
        disc_symbol_z_x = disc.process_symbol(symbol_z_x)
        submesh = mesh[("negative electrode", "separator", "positive electrode")]
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        lr = LR.flatten()
        tb = TB.flatten()
        ln = mesh["negative electrode"].edges_lr[-1]
        ls = mesh["separator"].edges_lr[-1] - ln
        lp = mesh["positive electrode"].edges_lr[-1] - (ln + ls)
        l_tb = mesh["negative electrode"].edges_tb[-1]
        np.testing.assert_allclose(
            disc_symbol_x_z.evaluate(None, y=lr),
            l_tb * (ln + ls + lp) ** 2 / 2,
            rtol=1e-7,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            disc_symbol_z_x.evaluate(None, y=tb),
            l_tb * (ln + ls + lp) ** 2 / 2,
            rtol=1e-7,
            atol=1e-6,
        )

        with pytest.raises(
            ValueError, match=r"Integration variables must be in different directions"
        ):
            disc.process_symbol(pybamm.Integral(var, [x, x]))

    def test_boundary_integral(self):
        # create discretisation
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume2D(),
            "separator": pybamm.FiniteVolume2D(),
            "positive electrode": pybamm.FiniteVolume2D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Get mesh dimensions
        submesh = mesh[("negative electrode", "separator", "positive electrode")]
        l_lr = submesh.edges_lr[-1] - submesh.edges_lr[0]  # length in lr direction
        l_tb = submesh.edges_tb[-1] - submesh.edges_tb[0]  # length in tb direction

        # Test 1: Constant variable on left boundary
        var = pybamm.Variable(
            "var", domain=["negative electrode", "separator", "positive electrode"]
        )
        boundary_integral_left = pybamm.BoundaryIntegral(var, region="left")
        disc.set_variable_slices([var])
        boundary_integral_left_disc = disc.process_symbol(boundary_integral_left)

        # For a constant value of 1, the boundary integral should equal the length of the boundary
        constant_y = np.ones(submesh.npts)
        result_left = boundary_integral_left_disc.evaluate(None, constant_y)
        np.testing.assert_allclose(result_left, l_tb, rtol=1e-10)

        # Test 2: Constant variable on right boundary
        boundary_integral_right = pybamm.BoundaryIntegral(var, region="right")
        boundary_integral_right_disc = disc.process_symbol(boundary_integral_right)
        result_right = boundary_integral_right_disc.evaluate(None, constant_y)
        np.testing.assert_allclose(result_right, l_tb, rtol=1e-10)

        # Test 3: Constant variable on top boundary
        boundary_integral_top = pybamm.BoundaryIntegral(var, region="top")
        boundary_integral_top_disc = disc.process_symbol(boundary_integral_top)
        result_top = boundary_integral_top_disc.evaluate(None, constant_y)
        np.testing.assert_allclose(result_top, l_lr, rtol=1e-10)

        # Test 4: Constant variable on bottom boundary
        boundary_integral_bottom = pybamm.BoundaryIntegral(var, region="bottom")
        boundary_integral_bottom_disc = disc.process_symbol(boundary_integral_bottom)
        result_bottom = boundary_integral_bottom_disc.evaluate(None, constant_y)
        np.testing.assert_allclose(result_bottom, l_lr, rtol=1e-10)

        # Test 5: Variable function - test with z-dependent variable on left/right boundaries
        # Create a z-dependent variable (linear in z from 0 to 1)
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        z_values = TB.flatten()  # z varies from 0 to 1

        # For left/right boundaries, we integrate over z direction
        # Integral of z from 0 to 1 should be 1/2
        result_left_z = boundary_integral_left_disc.evaluate(None, z_values)
        np.testing.assert_allclose(result_left_z, 0.5, rtol=1e-6)

        result_right_z = boundary_integral_right_disc.evaluate(None, z_values)
        np.testing.assert_allclose(result_right_z, 0.5, rtol=1e-6)

        # Test 6: Variable function - test with x-dependent variable on top/bottom boundaries
        # Create an x-dependent variable (linear in x from 0 to 1)
        x_values = LR.flatten()  # x varies from 0 to 1

        # For top/bottom boundaries, we integrate over x direction
        # Integral of x from 0 to 1 should be 1/2
        result_top_x = boundary_integral_top_disc.evaluate(None, x_values)
        np.testing.assert_allclose(result_top_x, 0.5, rtol=1e-6)

        result_bottom_x = boundary_integral_bottom_disc.evaluate(None, x_values)
        np.testing.assert_allclose(result_bottom_x, 0.5, rtol=1e-6)

        # Error
        with pytest.raises(ValueError, match=r"not supported"):
            boundary_integral_bottom = disc.process_symbol(
                pybamm.BoundaryIntegral(var, region="tab")
            )

        bcs = {
            var: {
                "left": (
                    pybamm.Scalar(0),
                    "Neumann",
                ),
                "right": (
                    pybamm.Scalar(0),
                    "Neumann",
                ),
                "top": (
                    pybamm.Scalar(0),
                    "Neumann",
                ),
                "bottom": (
                    pybamm.Scalar(0),
                    "Neumann",
                ),
            }
        }
        k = pybamm.VectorField(pybamm.Scalar(1 + 1e-5), pybamm.Scalar(1 + 1e-5))
        disc.bcs = bcs
        symbol = pybamm.Magnitude(k * pybamm.Gradient(var) / pybamm.Scalar(-5), "tb")
        boundary_integral_left = pybamm.BoundaryIntegral(symbol, region="bottom")
        # Fix this test
        boundary_integral_left_disc = disc.process_symbol(boundary_integral_left)
        result_left = boundary_integral_left_disc.evaluate(
            None, y=np.ones(submesh.npts)
        )
        np.testing.assert_array_almost_equal(result_left, 0.0)

    def test_one_dimensional_integral(self):
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {"macroscale": pybamm.FiniteVolume2D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable(
            "var", domain=["negative electrode", "separator", "positive electrode"]
        )
        disc.set_variable_slices([var])
        x = pybamm.SpatialVariable(
            "x",
            ["negative electrode", "separator", "positive electrode"],
            direction="lr",
        )
        z = pybamm.SpatialVariable(
            "z",
            ["negative electrode", "separator", "positive electrode"],
            direction="tb",
        )
        x_boundary_value = pybamm.BoundaryValue(x, "bottom")
        z_boundary_value = pybamm.BoundaryValue(z, "left")
        symbol_x = pybamm.OneDimensionalIntegral(
            x_boundary_value,
            integration_domain=[
                "negative electrode",
                "separator",
                "positive electrode",
            ],
            direction="lr",
        )
        symbol_z = pybamm.OneDimensionalIntegral(
            z_boundary_value,
            integration_domain=[
                "negative electrode",
                "separator",
                "positive electrode",
            ],
            direction="tb",
        )
        disc_symbol_x = disc.process_symbol(symbol_x)
        disc_symbol_z = disc.process_symbol(symbol_z)
        submesh = mesh[("negative electrode", "separator", "positive electrode")]
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        lr = LR.flatten()
        tb = TB.flatten()
        l_tb = mesh["negative electrode"].edges_tb[-1]
        ln = mesh["negative electrode"].edges_lr[-1]
        ls = mesh["separator"].edges_lr[-1] - ln
        lp = mesh["positive electrode"].edges_lr[-1] - (ln + ls)
        np.testing.assert_allclose(
            disc_symbol_x.evaluate(None, y=lr),
            (ln + ls + lp) ** 2 / 2,
            rtol=1e-7,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            disc_symbol_z.evaluate(None, y=tb),
            (l_tb) ** 2 / 2,
            rtol=1e-7,
            atol=1e-6,
        )

        with pytest.raises(ValueError, match=r"not supported"):
            disc.process_symbol(
                pybamm.OneDimensionalIntegral(
                    pybamm.Scalar(0),
                    integration_domain=[
                        "negative electrode",
                        "separator",
                        "positive electrode",
                    ],
                    direction="asdf",
                )
            )

        child_edges = pybamm.Vector(submesh.edges_lr)
        child_edges._evaluates_on_edges = lambda _: True
        with pytest.raises(NotImplementedError, match=r"not implemented"):
            disc.process_symbol(
                pybamm.OneDimensionalIntegral(
                    child_edges,
                    integration_domain=[
                        "negative electrode",
                        "separator",
                        "positive electrode",
                    ],
                    direction="lr",
                )
            )
