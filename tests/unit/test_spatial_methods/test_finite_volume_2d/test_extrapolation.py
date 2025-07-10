import numpy as np
import pytest

import pybamm
from tests.shared import get_mesh_for_testing_2d


class TestExtrapolationFiniteVolume2D:
    @pytest.mark.parametrize("use_bcs", [True, False])
    @pytest.mark.parametrize("order", ["linear", "quadratic"])
    def test_boundary_value_finite_volume_2d(self, use_bcs, order):
        if order == "quadratic" and use_bcs:
            # Not implemented
            return
        # Create discretisation
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume2D(
                {
                    "extrapolation": {
                        "order": {"gradient": "linear", "value": order},
                        "use bcs": use_bcs,
                    }
                }
            ),
        }
        disc_LR = pybamm.Discretisation(mesh, spatial_methods)
        disc_TB = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[("negative electrode", "separator", "positive electrode")]
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)

        var_LR = pybamm.Variable(
            "x", ["negative electrode", "separator", "positive electrode"]
        )
        var_TB = pybamm.Variable(
            "y", ["negative electrode", "separator", "positive electrode"]
        )
        disc_LR.set_variable_slices([var_LR])
        disc_TB.set_variable_slices([var_TB])
        directions = [
            "left",
            "right",
            "top",
            "bottom",
            "top-left",
            "top-right",
            "bottom-left",
            "bottom-right",
        ]

        solutions_LR = {
            "left": np.zeros(submesh.nodes_tb.shape),
            "right": np.ones(submesh.nodes_tb.shape),
            "top": submesh.nodes_lr,
            "bottom": submesh.nodes_lr,
            "top-left": 0.0,
            "top-right": 1.0,
            "bottom-left": 0.0,
            "bottom-right": 1.0,
        }

        solutions_TB = {
            "left": submesh.nodes_tb,
            "right": submesh.nodes_tb,
            "top": np.zeros(submesh.nodes_lr.shape),
            "bottom": np.ones(submesh.nodes_lr.shape),
            "top-left": 0.0,
            "top-right": 0.0,
            "bottom-left": 1.0,
            "bottom-right": 1.0,
        }

        bcs_LR = {
            var_LR: {
                "left": (pybamm.Scalar(0.0), "Dirichlet"),
                "right": (pybamm.Scalar(1.0), "Dirichlet"),
                "top": (pybamm.Scalar(0.0), "Neumann"),
                "bottom": (pybamm.Scalar(0.0), "Neumann"),
            }
        }
        bcs_TB = {
            var_TB: {
                "left": (pybamm.Scalar(0.0), "Neumann"),
                "right": (pybamm.Scalar(0.0), "Neumann"),
                "top": (pybamm.Scalar(0.0), "Dirichlet"),
                "bottom": (pybamm.Scalar(1.0), "Dirichlet"),
            }
        }

        discretised_extrapolations_LR = {}
        for direction in directions:
            disc_LR.bcs = bcs_LR
            discretised_extrapolations_LR[direction] = disc_LR.process_symbol(
                pybamm.BoundaryValue(var_LR, direction)
            )
        discretised_extrapolations_TB = {}
        for direction in directions:
            disc_TB.bcs = bcs_TB
            discretised_extrapolations_TB[direction] = disc_TB.process_symbol(
                pybamm.BoundaryValue(var_TB, direction)
            )

        for direction in directions:
            np.testing.assert_array_almost_equal(
                discretised_extrapolations_LR[direction]
                .evaluate(y=LR.flatten())
                .flatten(),
                solutions_LR[direction],
            )

        for direction in directions:
            np.testing.assert_array_almost_equal(
                discretised_extrapolations_TB[direction]
                .evaluate(y=TB.flatten())
                .flatten(),
                solutions_TB[direction],
            )

    @pytest.mark.parametrize("use_bcs", [True, False])
    @pytest.mark.parametrize("order", ["linear", "quadratic"])
    def test_boundary_gradient_finite_volume_2d(self, use_bcs, order):
        # Create discretisation
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume2D(
                {
                    "extrapolation": {
                        "order": {"gradient": order, "value": "linear"},
                        "use bcs": use_bcs,
                    }
                }
            ),
        }
        disc_LR = pybamm.Discretisation(mesh, spatial_methods)
        disc_TB = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[("negative electrode", "separator", "positive electrode")]
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)

        var_LR = pybamm.Variable(
            "x", ["negative electrode", "separator", "positive electrode"]
        )
        var_TB = pybamm.Variable(
            "y", ["negative electrode", "separator", "positive electrode"]
        )
        disc_LR.set_variable_slices([var_LR])
        disc_TB.set_variable_slices([var_TB])

        # Test boundary gradients in lr and tb directions
        directions_LR = ["left", "right"]
        directions_TB = ["top", "bottom"]

        # For linear x function:
        # - left/right boundaries: normal gradient (lr direction) should be 1
        # - top/bottom boundaries: normal gradient (tb direction) should be 0
        # For linear z function:
        # - left/right boundaries: normal gradient (lr direction) should be 0
        # - top/bottom boundaries: normal gradient (tb direction) should be 1
        solutions_LR = {
            "left": np.ones(submesh.nodes_tb.shape),  # normal gradient d/dx of x = 1
            "right": np.ones(submesh.nodes_tb.shape),  # normal gradient d/dx of x = 1
            "top": np.zeros(submesh.nodes_lr.shape),  # normal gradient d/dz of x = 0
            "bottom": np.zeros(submesh.nodes_lr.shape),  # normal gradient d/dz of x = 0
        }

        solutions_TB = {
            "left": np.zeros(submesh.nodes_tb.shape),  # normal gradient d/dx of z = 0
            "right": np.zeros(submesh.nodes_tb.shape),  # normal gradient d/dx of z = 0
            "top": np.ones(submesh.nodes_lr.shape),  # normal gradient d/dz of z = 1
            "bottom": np.ones(submesh.nodes_lr.shape),  # normal gradient d/dz of z = 1
        }

        bcs_LR = {
            var_LR: {
                "left": (pybamm.Scalar(0.0), "Dirichlet"),
                "right": (pybamm.Scalar(1.0), "Dirichlet"),
                "top": (pybamm.Scalar(0.0), "Neumann"),
                "bottom": (pybamm.Scalar(0.0), "Neumann"),
            }
        }
        bcs_TB = {
            var_TB: {
                "left": (pybamm.Scalar(0.0), "Neumann"),
                "right": (pybamm.Scalar(0.0), "Neumann"),
                "top": (pybamm.Scalar(0.0), "Dirichlet"),
                "bottom": (pybamm.Scalar(1.0), "Dirichlet"),
            }
        }

        # Test boundary gradients for linear x function
        discretised_gradients_LR = {}
        for direction in directions_LR:
            disc_LR.bcs = bcs_LR
            discretised_gradients_LR[direction] = disc_LR.process_symbol(
                pybamm.BoundaryGradient(var_LR, direction)
            )

        # Test boundary gradients for linear z function
        discretised_gradients_TB = {}
        for direction in directions_TB:
            disc_TB.bcs = bcs_TB
            discretised_gradients_TB[direction] = disc_TB.process_symbol(
                pybamm.BoundaryGradient(var_TB, direction)
            )

        # Check results for linear x function
        for direction in directions_LR:
            np.testing.assert_array_almost_equal(
                discretised_gradients_LR[direction].evaluate(y=LR.flatten()).flatten(),
                solutions_LR[direction],
            )

        # Check results for linear z function
        for direction in directions_TB:
            np.testing.assert_array_almost_equal(
                discretised_gradients_TB[direction].evaluate(y=TB.flatten()).flatten(),
                solutions_TB[direction],
            )

    @pytest.mark.parametrize("use_bcs", [True, False])
    @pytest.mark.parametrize(
        "direction",
        [
            "left",
            "right",
            "top",
            "bottom",
            "top-left",
            "top-right",
            "bottom-left",
            "bottom-right",
        ],
    )
    def test_cubic_not_implemented(self, use_bcs, direction):
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume2D(
                {
                    "extrapolation": {
                        "order": {"gradient": "cubic", "value": "cubic"},
                        "use bcs": use_bcs,
                    }
                }
            ),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable(
            "x", ["negative electrode", "separator", "positive electrode"]
        )
        disc.set_variable_slices([var])
        with pytest.raises(NotImplementedError):
            disc.process_symbol(pybamm.BoundaryValue(var, direction))

    @pytest.mark.parametrize("use_bcs", [True, False])
    def test_boundary_gradient_quadratic_function_finite_volume_2d(self, use_bcs):
        # Create discretisation with quadratic gradient extrapolation
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume2D(
                {
                    "extrapolation": {
                        "order": {"gradient": "quadratic", "value": "quadratic"},
                        "use bcs": use_bcs,
                    }
                }
            ),
        }
        disc_LR = pybamm.Discretisation(mesh, spatial_methods)
        disc_TB = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[("negative electrode", "separator", "positive electrode")]
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)

        var_LR = pybamm.Variable(
            "x_squared", ["negative electrode", "separator", "positive electrode"]
        )
        var_TB = pybamm.Variable(
            "z_squared", ["negative electrode", "separator", "positive electrode"]
        )
        disc_LR.set_variable_slices([var_LR])
        disc_TB.set_variable_slices([var_TB])

        # Test boundary gradients in lr and tb directions for quadratic functions
        directions_LR = ["left", "right"]
        directions_TB = ["top", "bottom"]

        # For quadratic x^2 function:
        # - gradient is d/dx(x^2) = 2x
        # - at left boundary (x=0): gradient = 0
        # - at right boundary (x=1): gradient = 2
        # For quadratic z^2 function:
        # - gradient is d/dz(z^2) = 2z
        # - at top boundary (z=0): gradient = 0
        # - at bottom boundary (z=1): gradient = 2
        solutions_LR = {
            "left": np.zeros(submesh.nodes_tb.shape),  # gradient of x^2 at x=0 is 0
            "right": 2 * np.ones(submesh.nodes_tb.shape),  # gradient of x^2 at x=1 is 2
        }

        solutions_TB = {
            "top": np.zeros(submesh.nodes_lr.shape),  # gradient of z^2 at z=0 is 0
            "bottom": 2
            * np.ones(submesh.nodes_lr.shape),  # gradient of z^2 at z=1 is 2
        }

        # Set up boundary conditions for x^2 function
        # x^2 = 0 at x=0, x^2 = 1 at x=1
        bcs_LR = {
            var_LR: {
                "left": (pybamm.Scalar(0.0), "Dirichlet"),  # x^2 = 0 at x=0
                "right": (pybamm.Scalar(1.0), "Dirichlet"),  # x^2 = 1 at x=1
                "top": (pybamm.Scalar(0.0), "Neumann"),
                "bottom": (pybamm.Scalar(0.0), "Neumann"),
            }
        }

        # Set up boundary conditions for z^2 function
        # z^2 = 0 at z=0, z^2 = 1 at z=1
        bcs_TB = {
            var_TB: {
                "left": (pybamm.Scalar(0.0), "Neumann"),
                "right": (pybamm.Scalar(0.0), "Neumann"),
                "top": (pybamm.Scalar(0.0), "Dirichlet"),  # z^2 = 0 at z=0
                "bottom": (pybamm.Scalar(1.0), "Dirichlet"),  # z^2 = 1 at z=1
            }
        }

        # Test boundary gradients for quadratic x^2 function
        discretised_gradients_LR = {}
        for direction in directions_LR:
            disc_LR.bcs = bcs_LR
            discretised_gradients_LR[direction] = disc_LR.process_symbol(
                pybamm.BoundaryGradient(var_LR, direction)
            )

        # Test boundary gradients for quadratic z^2 function
        discretised_gradients_TB = {}
        for direction in directions_TB:
            disc_TB.bcs = bcs_TB
            discretised_gradients_TB[direction] = disc_TB.process_symbol(
                pybamm.BoundaryGradient(var_TB, direction)
            )

        # Create quadratic test data: x^2 and z^2
        x_squared_data = LR.flatten() ** 2
        z_squared_data = TB.flatten() ** 2

        # Check results for quadratic x^2 function
        for direction in directions_LR:
            np.testing.assert_array_almost_equal(
                discretised_gradients_LR[direction]
                .evaluate(y=x_squared_data)
                .flatten(),
                solutions_LR[direction],
                decimal=5,
            )

        # Check results for quadratic z^2 function
        for direction in directions_TB:
            np.testing.assert_array_almost_equal(
                discretised_gradients_TB[direction]
                .evaluate(y=z_squared_data)
                .flatten(),
                solutions_TB[direction],
                decimal=5,
            )
