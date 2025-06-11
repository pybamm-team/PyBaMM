import numpy as np
import pytest

import pybamm
from tests import get_mesh_for_testing_2d


class TestFiniteVolume2D:
    def test_node_to_edge_to_node(self):
        # Create discretisation
        mesh = get_mesh_for_testing_2d()
        fin_vol = pybamm.FiniteVolume2D()
        fin_vol.build(mesh)
        n_lr = mesh["negative electrode"].npts_lr
        n_tb = mesh["negative electrode"].npts_tb

        # node to edge
        c = pybamm.StateVector(slice(0, n_lr * n_tb), domain=["negative electrode"])
        y_test = np.ones(n_lr * n_tb) * 2
        diffusivity_c_ari_lr = fin_vol.node_to_edge(
            c, method="arithmetic", direction="lr"
        )
        np.testing.assert_array_equal(
            diffusivity_c_ari_lr.evaluate(None, y_test),
            np.ones(((n_lr + 1) * (n_tb), 1)) * 2,
        )
        diffusivity_c_har_lr = fin_vol.node_to_edge(
            c, method="harmonic", direction="lr"
        )
        np.testing.assert_array_equal(
            diffusivity_c_har_lr.evaluate(None, y_test),
            np.ones(((n_lr + 1) * (n_tb), 1)) * 2,
        )
        diffusivity_c_ari_tb = fin_vol.node_to_edge(
            c, method="arithmetic", direction="tb"
        )
        np.testing.assert_array_equal(
            diffusivity_c_ari_tb.evaluate(None, y_test),
            np.ones(((n_lr) * (n_tb + 1), 1)) * 2,
        )
        diffusivity_c_har_tb = fin_vol.node_to_edge(
            c, method="harmonic", direction="tb"
        )
        np.testing.assert_array_equal(
            diffusivity_c_har_tb.evaluate(None, y_test),
            np.ones(((n_lr) * (n_tb + 1), 1)) * 2,
        )

        # bad shift key
        with pytest.raises(ValueError, match="shift key"):
            fin_vol.shift(c, "bad shift key", "arithmetic")

        with pytest.raises(ValueError, match="shift key"):
            fin_vol.shift(c, "bad shift key", "harmonic")

        # bad method
        with pytest.raises(ValueError, match="method"):
            fin_vol.shift(c, "shift key", "bad method")

    def test_discretise_spatial_variable(self):
        # Create discretisation
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume2D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # macroscale
        x1 = pybamm.SpatialVariable("x", ["negative electrode"], direction="lr")
        x2 = pybamm.SpatialVariable("x2", ["negative electrode"], direction="tb")
        x1_disc = disc.process_symbol(x1)
        x2_disc = disc.process_symbol(x2)
        assert isinstance(x1_disc, pybamm.Vector)
        LR, TB = np.meshgrid(
            disc.mesh["negative electrode"].nodes_lr,
            disc.mesh["negative electrode"].nodes_tb,
        )
        np.testing.assert_array_equal(x1_disc.evaluate().flatten(), LR.flatten())
        np.testing.assert_array_equal(x2_disc.evaluate().flatten(), TB.flatten())
        # macroscale with concatenation
        x3 = pybamm.SpatialVariable(
            "x3", ["negative electrode", "separator"], direction="lr"
        )
        x4 = pybamm.SpatialVariable(
            "x4", ["negative electrode", "separator"], direction="tb"
        )
        x3_disc = disc.process_symbol(x3)
        x4_disc = disc.process_symbol(x4)
        assert isinstance(x2_disc, pybamm.Vector)
        submesh = disc.mesh[("negative electrode", "separator")]
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        np.testing.assert_array_equal(x3_disc.evaluate().flatten(), LR.flatten())
        np.testing.assert_array_equal(x4_disc.evaluate().flatten(), TB.flatten())

    def test_discretise_diffusivity_times_spatial_operator(self):
        # Setup mesh and discretisation
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {"macroscale": pybamm.FiniteVolume2D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        submesh = mesh[whole_cell]

        # Discretise some equations where averaging is needed
        var = pybamm.Variable("var", domain=whole_cell)
        disc.set_variable_slices([var])
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        for eqn in [
            var * pybamm.grad(var),
            var**2 * pybamm.grad(var),
            var * pybamm.grad(var) ** 2,
            var * (pybamm.grad(var) + 2),
            (pybamm.grad(var) + 2) * (-var),
            (pybamm.grad(var) + 2) * (2 * var),
            pybamm.grad(var) * pybamm.grad(var),
            (pybamm.grad(var) + 2) * pybamm.grad(var) ** 2,
        ]:
            # Check that the equation can be evaluated for different combinations
            # of boundary conditions
            # Dirichlet
            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Dirichlet"),
                    "right": (pybamm.Scalar(1), "Dirichlet"),
                    "top": (pybamm.Scalar(0), "Dirichlet"),
                    "bottom": (pybamm.Scalar(1), "Dirichlet"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.lr_field.evaluate(None, LR.flatten())
            eqn_disc.tb_field.evaluate(None, TB.flatten())
            # Neumann
            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(1), "Neumann"),
                    "top": (pybamm.Scalar(0), "Neumann"),
                    "bottom": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.lr_field.evaluate(None, LR.flatten())
            eqn_disc.tb_field.evaluate(None, TB.flatten())
            # One of each
            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Dirichlet"),
                    "right": (pybamm.Scalar(1), "Neumann"),
                    "top": (pybamm.Scalar(0), "Dirichlet"),
                    "bottom": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.lr_field.evaluate(None, LR.flatten())
            eqn_disc.tb_field.evaluate(None, TB.flatten())
            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(1), "Dirichlet"),
                    "top": (pybamm.Scalar(0), "Neumann"),
                    "bottom": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.lr_field.evaluate(None, LR.flatten())
            eqn_disc.tb_field.evaluate(None, TB.flatten())

        for eqn in [
            pybamm.div(pybamm.grad(var)),
            pybamm.div(pybamm.grad(var)) + 2,
            pybamm.div(pybamm.grad(var)) + var,
            pybamm.div(2 * pybamm.grad(var)),
            pybamm.div(2 * pybamm.grad(var)) + 3 * var,
            -2 * pybamm.div(var * pybamm.grad(var) + 2 * pybamm.grad(var)),
            pybamm.laplacian(var),
        ]:
            # Check that the equation can be evaluated for different combinations
            # of boundary conditions
            # Dirichlet
            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Dirichlet"),
                    "right": (pybamm.Scalar(1), "Dirichlet"),
                    "top": (pybamm.Scalar(0), "Dirichlet"),
                    "bottom": (pybamm.Scalar(1), "Dirichlet"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, LR.flatten())
            eqn_disc.evaluate(None, TB.flatten())
            # Neumann
            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(1), "Neumann"),
                    "top": (pybamm.Scalar(0), "Neumann"),
                    "bottom": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, LR.flatten())
            eqn_disc.evaluate(None, TB.flatten())
            # One of each
            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Dirichlet"),
                    "right": (pybamm.Scalar(1), "Neumann"),
                    "top": (pybamm.Scalar(0), "Dirichlet"),
                    "bottom": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, LR.flatten())
            eqn_disc.evaluate(None, TB.flatten())
            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(1), "Dirichlet"),
                    "top": (pybamm.Scalar(0), "Neumann"),
                    "bottom": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, LR.flatten())
            eqn_disc.evaluate(None, TB.flatten())
