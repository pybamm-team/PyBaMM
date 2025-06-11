import numpy as np

import pybamm
from tests.shared import get_mesh_for_testing_2d


class TestExtrapolationFiniteVolume2D:
    def test_extrapolation_finite_volume_2d_no_bcs(self):
        # Create discretisation
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume2D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[("negative electrode", "separator", "positive electrode")]
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)

        var = pybamm.Variable(
            "x", ["negative electrode", "separator", "positive electrode"]
        )
        disc.set_variable_slices([var])
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

        discretised_extrapolations = {}
        for direction in directions:
            discretised_extrapolations[direction] = disc.process_symbol(
                pybamm.BoundaryValue(var, direction)
            )

        for direction in directions:
            np.testing.assert_array_almost_equal(
                discretised_extrapolations[direction]
                .evaluate(y=LR.flatten())
                .flatten(),
                solutions_LR[direction],
            )

        for direction in directions:
            np.testing.assert_array_almost_equal(
                discretised_extrapolations[direction]
                .evaluate(y=TB.flatten())
                .flatten(),
                solutions_TB[direction],
            )
