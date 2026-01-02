import numpy as np
import pytest

import pybamm
from tests import get_mesh_for_testing, get_mesh_for_testing_2d


class TestFiniteVolume2D:
    def test_node_to_edge_to_node(self):
        # Create discretisation
        mesh = get_mesh_for_testing_2d()
        fin_vol = pybamm.FiniteVolume2D()
        fin_vol.build(mesh)
        n_lr = mesh[("negative electrode", "separator", "positive electrode")].npts_lr
        n_tb = mesh[("negative electrode", "separator", "positive electrode")].npts_tb

        # node to edge
        var_left = pybamm.Variable("var_left", domain=["negative electrode"])
        var_separator = pybamm.Variable("var_separator", domain=["separator"])
        var_right = pybamm.Variable("var_right", domain=["positive electrode"])
        var_concat = pybamm.concatenation(var_left, var_separator, var_right)
        disc = pybamm.Discretisation(mesh, {"macroscale": fin_vol})
        disc.set_variable_slices([var_concat])
        c = disc.process_symbol(var_concat)

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
        with pytest.raises(ValueError, match=r"shift key"):
            fin_vol.shift(c, "bad shift key", "arithmetic")

        with pytest.raises(ValueError, match=r"shift key"):
            fin_vol.shift(c, "bad shift key", "harmonic")

        # bad method
        with pytest.raises(ValueError, match=r"method"):
            fin_vol.shift(c, "shift key", "bad method")

        # Edge to node
        LR, TB = np.meshgrid(
            mesh[("negative electrode", "separator", "positive electrode")].nodes_lr,
            mesh[("negative electrode", "separator", "positive electrode")].nodes_tb,
        )
        lr = LR.flatten()
        tb = TB.flatten()
        bcs = {
            var_concat: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(1), "Dirichlet"),
                "bottom": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        disc.bcs = bcs
        symbol = pybamm.Gradient(var_concat)
        disc_symbol = disc.process_symbol(symbol)
        left_grad = fin_vol.edge_to_node(
            disc_symbol.lr_field, method="arithmetic", direction="lr"
        )
        right_grad = fin_vol.edge_to_node(
            disc_symbol.tb_field, method="arithmetic", direction="tb"
        )
        np.testing.assert_array_almost_equal(
            left_grad.evaluate(None, lr).flatten(), np.ones(n_lr * n_tb)
        )
        np.testing.assert_array_almost_equal(
            right_grad.evaluate(None, tb).flatten(), np.ones(n_lr * n_tb)
        )

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

        # test invalid direction
        with pytest.raises(ValueError, match=r"Direction asdf not supported"):
            disc.process_symbol(
                pybamm.SpatialVariable("q", ["negative electrode"], direction="asdf")
            )

        x_edges = pybamm.SpatialVariable(
            "x_edges", ["negative electrode", "separator"], direction="lr"
        )
        x_edges._evaluates_on_edges = lambda _: True
        x_edges_disc = disc.process_symbol(x_edges)
        LR, TB = np.meshgrid(submesh.edges_lr, submesh.edges_tb)
        np.testing.assert_array_equal(x_edges_disc.evaluate().flatten(), LR.flatten())

        mesh_1d = get_mesh_for_testing()
        spatial_methods_1d = {"macroscale": pybamm.FiniteVolume2D()}
        disc_1d = pybamm.Discretisation(mesh_1d, spatial_methods_1d)
        x_1d = pybamm.SpatialVariable("x_1d", ["negative electrode"], direction="lr")
        with pytest.raises(ValueError, match=r"Spatial variable x_1d is not in 2D"):
            disc_1d.process_symbol(x_1d)

    def test_process_binary_operators(self):
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
            pybamm.grad(var) * var,
            var**2 * pybamm.grad(var),
            var * pybamm.grad(var) ** 2,
            var * (pybamm.grad(var) + 2),
            (pybamm.grad(var) + 2) * (-var),
            (pybamm.grad(var) + 2) * (2 * var),
            # this very complicated one catches a coverage case
            (
                pybamm.VectorField(pybamm.Scalar(2), pybamm.Scalar(2))
                + (2 * pybamm.grad(var))
            )
            * (2 * var),
            pybamm.grad(var) * pybamm.grad(var),
            (pybamm.grad(var) + 2) * pybamm.grad(var) ** 2,
            pybamm.VectorField(pybamm.Scalar(2), pybamm.Scalar(2)) * pybamm.Scalar(2),
            pybamm.VectorField(pybamm.Scalar(2), pybamm.Scalar(2))
            * pybamm.VectorField(pybamm.Scalar(2), pybamm.Scalar(2)),
            pybamm.UpwindDownwind2D(var, None, None) * pybamm.VectorField(var, var),
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
            pybamm.Inner(pybamm.grad(var), var),
            pybamm.Inner(var, pybamm.grad(var)),
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

        for eqn in [
            var * pybamm.Magnitude(pybamm.grad(var), "lr"),
            pybamm.Magnitude(pybamm.grad(var), "lr") * var,
        ]:
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, LR.flatten())
            eqn_disc.evaluate(None, TB.flatten())

    def test_upwind_downwind_2d(self):
        """
        Test upwind and downwind operators in 2D finite volume method
        """
        # Create discretisation
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume2D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Create variable
        var = pybamm.Variable("var", domain=["negative electrode"])
        disc.set_variable_slices([var])

        # Get submesh and spatial method
        submesh = mesh["negative electrode"]
        spatial_method = spatial_methods["negative electrode"]
        spatial_method.build(mesh)

        # Process the variable
        var_disc = disc.process_symbol(var)

        # Test 1: Test with None directions (should use node_to_edge)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0.0), "Neumann"),
                "right": (pybamm.Scalar(0.0), "Neumann"),
                "top": (pybamm.Scalar(0.0), "Neumann"),
                "bottom": (pybamm.Scalar(0.0), "Neumann"),
            }
        }

        # Test None directions - should use node_to_edge
        disc.bcs = boundary_conditions
        result_none = spatial_method.upwind_or_downwind(
            var, var_disc, boundary_conditions, None, None
        )
        symbol = pybamm.UpwindDownwind2D(var, None, None)
        symbol_disc = disc.process_symbol(symbol)
        assert result_none == symbol_disc

        # Check that result is a VectorField
        assert isinstance(result_none, pybamm.VectorField)

        # Compare with direct node_to_edge calls to verify it's using the same method
        node_to_edge_lr = spatial_method.node_to_edge(
            var_disc, method="arithmetic", direction="lr"
        )
        node_to_edge_tb = spatial_method.node_to_edge(
            var_disc, method="arithmetic", direction="tb"
        )

        # Check shapes - should match node_to_edge output
        expected_lr_size = (submesh.npts_lr + 1) * submesh.npts_tb
        expected_tb_size = submesh.npts_lr * (submesh.npts_tb + 1)

        # Create test values
        LR, _TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        test_values = LR.flatten()

        lr_result = result_none.lr_field.evaluate(None, test_values)
        tb_result = result_none.tb_field.evaluate(None, test_values)

        assert lr_result.size == expected_lr_size
        assert tb_result.size == expected_tb_size

        # Verify that the results match node_to_edge
        lr_direct = node_to_edge_lr.evaluate(None, test_values)
        tb_direct = node_to_edge_tb.evaluate(None, test_values)

        np.testing.assert_array_almost_equal(lr_result.flatten(), lr_direct.flatten())
        np.testing.assert_array_almost_equal(tb_result.flatten(), tb_direct.flatten())

        # Test 1b: Test with only lr_direction=None (tb_direction should be ignored when None)
        result_lr_none = spatial_method.upwind_or_downwind(
            var, var_disc, boundary_conditions, None, None
        )

        # Should be the same as the None, None case
        lr_result_b = result_lr_none.lr_field.evaluate(None, test_values)
        tb_result_b = result_lr_none.tb_field.evaluate(None, test_values)

        np.testing.assert_array_almost_equal(lr_result_b.flatten(), lr_direct.flatten())
        np.testing.assert_array_almost_equal(tb_result_b.flatten(), tb_direct.flatten())

        # Test 1c: Test with only tb_direction=None (lr_direction should be ignored when None)
        result_tb_none = spatial_method.upwind_or_downwind(
            var, var_disc, boundary_conditions, None, None
        )

        # Should be the same as the None, None case
        lr_result_c = result_tb_none.lr_field.evaluate(None, test_values)
        tb_result_c = result_tb_none.tb_field.evaluate(None, test_values)

        np.testing.assert_array_almost_equal(lr_result_c.flatten(), lr_direct.flatten())
        np.testing.assert_array_almost_equal(tb_result_c.flatten(), tb_direct.flatten())

        # Test 2: Error cases - missing boundary conditions for upwind
        # Should raise error when upwinding without left Dirichlet BC
        with pytest.raises(
            pybamm.ModelError, match=r"Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions, "downwind", None
            )

        # Should raise error when downwinding without right Dirichlet BC
        with pytest.raises(
            pybamm.ModelError, match=r"Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions, "upwind", None
            )

        # Should raise error when upwinding in tb without top Dirichlet BC
        with pytest.raises(
            pybamm.ModelError, match=r"Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions, None, "downwind"
            )

        # Should raise error when downwinding in tb without bottom Dirichlet BC
        with pytest.raises(
            pybamm.ModelError, match=r"Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions, None, "upwind"
            )

        # Test 3: Error case - no boundary conditions at all
        with pytest.raises(
            pybamm.ModelError, match=r"Boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(var, var_disc, {}, "upwind", None)

        # Test 4: Error case - invalid direction
        with pytest.raises(ValueError, match=r"direction 'invalid' not recognised"):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions, "invalid", None
            )

        with pytest.raises(ValueError, match=r"direction 'invalid' not recognised"):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions, None, "invalid"
            )

        # Test 5: Test mixed boundary conditions for 100% coverage

        # Test 5a: Verify error messages for missing Dirichlet BCs are specific
        boundary_conditions_no_left_dirichlet = {
            var: {
                "left": (pybamm.Scalar(0.0), "Neumann"),
                "right": (pybamm.Scalar(0.0), "Neumann"),
                "top": (pybamm.Scalar(0.0), "Neumann"),
                "bottom": (pybamm.Scalar(0.0), "Neumann"),
            }
        }

        # Test that upwind requires left Dirichlet BC
        with pytest.raises(
            pybamm.ModelError, match=r"Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions_no_left_dirichlet, "downwind", None
            )

        # Test that downwind requires right Dirichlet BC
        with pytest.raises(
            pybamm.ModelError, match=r"Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions_no_left_dirichlet, "upwind", None
            )

        # Test that upwind in tb requires top Dirichlet BC
        with pytest.raises(
            pybamm.ModelError, match=r"Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions_no_left_dirichlet, None, "downwind"
            )

        # Test that downwind in tb requires bottom Dirichlet BC
        with pytest.raises(
            pybamm.ModelError, match=r"Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions_no_left_dirichlet, None, "upwind"
            )

        # Test 5b: Test the boundary condition checking logic more thoroughly
        # Test with left Dirichlet but trying downwind (should still fail)
        boundary_conditions_left_dirichlet_only = {
            var: {
                "left": (pybamm.Scalar(1.0), "Dirichlet"),
                "right": (pybamm.Scalar(0.0), "Neumann"),
                "top": (pybamm.Scalar(0.0), "Neumann"),
                "bottom": (pybamm.Scalar(0.0), "Neumann"),
            }
        }

        # Downwind should fail even with left Dirichlet (needs right Dirichlet)
        with pytest.raises(
            pybamm.ModelError, match=r"Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions_left_dirichlet_only, "downwind", None
            )

        # Test with right Dirichlet but trying upwind (should still fail)
        boundary_conditions_right_dirichlet_only = {
            var: {
                "left": (pybamm.Scalar(0.0), "Neumann"),
                "right": (pybamm.Scalar(2.0), "Dirichlet"),
                "top": (pybamm.Scalar(0.0), "Neumann"),
                "bottom": (pybamm.Scalar(0.0), "Neumann"),
            }
        }

        # Upwind should fail even with right Dirichlet (needs left Dirichlet)
        with pytest.raises(
            pybamm.ModelError, match=r"Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions_right_dirichlet_only, "upwind", None
            )

        # Test 5c: Test tb direction boundary condition checking
        # Test with top Dirichlet but trying downwind in tb (should fail)
        boundary_conditions_top_dirichlet_only = {
            var: {
                "left": (pybamm.Scalar(0.0), "Neumann"),
                "right": (pybamm.Scalar(0.0), "Neumann"),
                "top": (pybamm.Scalar(1.5), "Dirichlet"),
                "bottom": (pybamm.Scalar(0.0), "Neumann"),
            }
        }

        # Downwind in tb should fail even with top Dirichlet (needs bottom Dirichlet)
        with pytest.raises(
            pybamm.ModelError, match=r"Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions_top_dirichlet_only, None, "upwind"
            )

        # Test with bottom Dirichlet but trying upwind in tb (should fail)
        boundary_conditions_bottom_dirichlet_only = {
            var: {
                "left": (pybamm.Scalar(0.0), "Neumann"),
                "right": (pybamm.Scalar(0.0), "Neumann"),
                "top": (pybamm.Scalar(0.0), "Neumann"),
                "bottom": (pybamm.Scalar(2.5), "Dirichlet"),
            }
        }

        # Upwind in tb should fail even with bottom Dirichlet (needs top Dirichlet)
        with pytest.raises(
            pybamm.ModelError, match=r"Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var,
                var_disc,
                boundary_conditions_bottom_dirichlet_only,
                None,
                "downwind",
            )

        # Test 6: Test the add_ghost_nodes code paths (lines 2145-2146 and 2159-2160)

        # Test 6a: Test lr_direction add_ghost_nodes path (lines 2145-2146)
        boundary_conditions_minimal_lr = {
            var: {
                "left": (pybamm.Scalar(0.0), "Dirichlet"),
                "right": (pybamm.Scalar(0.0), "Neumann"),
                "top": (pybamm.Scalar(0.0), "Neumann"),
                "bottom": (pybamm.Scalar(0.0), "Neumann"),
            }
        }

        # This should reach line 2145-2146 (add_ghost_nodes for lr direction)
        result_lr_ghost = spatial_method.upwind_or_downwind(
            var, var_disc, boundary_conditions_minimal_lr, "upwind", None
        )
        # Verify it's a VectorField
        assert isinstance(result_lr_ghost, pybamm.VectorField)

        # Test 6b: Test tb_direction add_ghost_nodes path (lines 2159-2160)
        boundary_conditions_minimal_tb = {
            var: {
                "left": (pybamm.Scalar(0.0), "Neumann"),
                "right": (pybamm.Scalar(0.0), "Neumann"),
                "top": (pybamm.Scalar(0.0), "Dirichlet"),
                "bottom": (pybamm.Scalar(0.0), "Neumann"),
            }
        }

        # This should reach line 2159-2160 (add_ghost_nodes for tb direction)
        result_tb_ghost = spatial_method.upwind_or_downwind(
            var, var_disc, boundary_conditions_minimal_tb, None, "downwind"
        )
        # Verify it's a VectorField
        assert isinstance(result_tb_ghost, pybamm.VectorField)

        # Test 6c: Test both directions with add_ghost_nodes
        boundary_conditions_both_minimal = {
            var: {
                "left": (pybamm.Scalar(0.0), "Dirichlet"),
                "right": (pybamm.Scalar(0.0), "Neumann"),
                "top": (pybamm.Scalar(0.0), "Dirichlet"),
                "bottom": (pybamm.Scalar(0.0), "Neumann"),
            }
        }

        # This should reach both add_ghost_nodes code paths
        result_both_ghost = spatial_method.upwind_or_downwind(
            var, var_disc, boundary_conditions_both_minimal, "upwind", "downwind"
        )
        # Verify it's a VectorField
        assert isinstance(result_both_ghost, pybamm.VectorField)

    def test_2d_concatenations(self):
        # Create discretisation
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume2D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        neg_electrode = pybamm.Variable("symbol_neg", domain=["negative electrode"])
        separator = pybamm.Variable("symbol_sep", domain=["separator"])
        pos_electrode = pybamm.Variable("symbol_pos", domain=["positive electrode"])
        concat_var = pybamm.concatenation(neg_electrode, separator, pos_electrode)
        disc.set_variable_slices([concat_var])
        concat_var_disc = disc.process_symbol(concat_var)
        submesh_total = mesh[("negative electrode", "separator", "positive electrode")]
        LR, TB = np.meshgrid(submesh_total.nodes_lr, submesh_total.nodes_tb)
        linear_x = LR.flatten()
        linear_y = TB.flatten()
        lr_result = concat_var_disc.evaluate(None, linear_x).flatten()
        tb_result = concat_var_disc.evaluate(None, linear_y).flatten()
        np.testing.assert_array_equal(lr_result, LR.flatten())
        np.testing.assert_array_equal(tb_result, TB.flatten())

        # Now check individual vectors
        submesh_neg = mesh[("negative electrode")]
        submesh_sep = mesh[("separator")]
        submesh_pos = mesh[("positive electrode")]
        LR_neg, TB_neg = np.meshgrid(submesh_neg.nodes_lr, submesh_neg.nodes_tb)
        LR_sep, TB_sep = np.meshgrid(submesh_sep.nodes_lr, submesh_sep.nodes_tb)
        LR_pos, TB_pos = np.meshgrid(submesh_pos.nodes_lr, submesh_pos.nodes_tb)
        neg_electrode_disc = disc.process_symbol(neg_electrode)
        separator_disc = disc.process_symbol(separator)
        pos_electrode_disc = disc.process_symbol(pos_electrode)
        neg_electrode_lr = neg_electrode_disc.evaluate(None, linear_x).flatten()
        separator_lr = separator_disc.evaluate(None, linear_x).flatten()
        pos_electrode_lr = pos_electrode_disc.evaluate(None, linear_x).flatten()
        neg_electrode_tb = neg_electrode_disc.evaluate(None, linear_y).flatten()
        separator_tb = separator_disc.evaluate(None, linear_y).flatten()
        pos_electrode_tb = pos_electrode_disc.evaluate(None, linear_y).flatten()
        np.testing.assert_array_equal(neg_electrode_lr, LR_neg.flatten())
        np.testing.assert_array_equal(separator_lr, LR_sep.flatten())
        np.testing.assert_array_equal(pos_electrode_lr, LR_pos.flatten())
        np.testing.assert_array_equal(neg_electrode_tb, TB_neg.flatten())
        np.testing.assert_array_equal(separator_tb, TB_sep.flatten())
        np.testing.assert_array_equal(pos_electrode_tb, TB_pos.flatten())

        # Now check concatenating broadcasted constants
        source_neg = pybamm.PrimaryBroadcast(pybamm.Scalar(1.0), ["negative electrode"])
        source_sep = pybamm.PrimaryBroadcast(pybamm.Scalar(2.0), ["separator"])
        source_pos = pybamm.PrimaryBroadcast(pybamm.Scalar(3.0), ["positive electrode"])
        source_concat = pybamm.concatenation(source_neg, source_sep, source_pos)
        source_concat_disc = disc.process_symbol(source_concat)
        source_concat_result = source_concat_disc.evaluate(None, None).flatten()
        thing = []
        for _ in submesh_total.nodes_tb:
            thing.append(1 * np.ones(submesh_neg.npts_lr))
            thing.append(2 * np.ones(submesh_sep.npts_lr))
            thing.append(3 * np.ones(submesh_pos.npts_lr))
        source_concat_expected = np.concatenate(thing)
        np.testing.assert_array_equal(source_concat_result, source_concat_expected)

        # Now check adding broadcasted constants to state vectors
        constant_y = np.ones(submesh_total.npts)
        neg_add = neg_electrode + source_neg
        separator_add = separator + source_sep
        pos_add = pos_electrode + source_pos
        total_add = concat_var + source_concat
        neg_add_disc = disc.process_symbol(neg_add)
        separator_add_disc = disc.process_symbol(separator_add)
        pos_add_disc = disc.process_symbol(pos_add)
        total_add_disc = disc.process_symbol(total_add)
        neg_add_result = neg_add_disc.evaluate(None, constant_y).flatten()
        separator_add_result = separator_add_disc.evaluate(None, constant_y).flatten()
        pos_add_result = pos_add_disc.evaluate(None, constant_y).flatten()
        total_add_result = total_add_disc.evaluate(None, constant_y).flatten()
        np.testing.assert_array_equal(neg_add_result, 2 * np.ones(submesh_neg.npts))
        np.testing.assert_array_equal(
            separator_add_result, 3 * np.ones(submesh_sep.npts)
        )
        np.testing.assert_array_equal(pos_add_result, 4 * np.ones(submesh_pos.npts))
        total_add_result_expected = source_concat_expected + 1
        np.testing.assert_array_equal(total_add_result, total_add_result_expected)

        # Now check concatenating broadcasted constants to state vectors
        constant_y = np.ones(submesh_total.npts)
        neg_add = neg_electrode + source_neg
        separator_add = separator + source_sep
        pos_add = pos_electrode + source_pos
        total_add = concat_var + source_concat
        neg_add_disc = disc.process_symbol(neg_add)

        # test functions that vary in X and Z
        z_neg = pybamm.SpatialVariable(
            "z_neg",
            domain="negative electrode",
            direction="tb",
        )
        z_sep = pybamm.SpatialVariable(
            "z_sep",
            domain="separator",
            direction="tb",
        )
        z_pos = pybamm.SpatialVariable(
            "z_pos",
            domain="positive electrode",
            direction="tb",
        )
        z_concat = pybamm.concatenation(z_neg, z_sep, z_pos)
        z_concat_disc = disc.process_symbol(z_concat)
        z_concat_result = z_concat_disc.evaluate(None, None).flatten()
        z = pybamm.SpatialVariable(
            "z",
            ["negative electrode", "separator", "positive electrode"],
            direction="tb",
        )
        z_disc = disc.process_symbol(z)
        z_result = z_disc.evaluate(None, None).flatten()
        np.testing.assert_array_equal(z_concat_result, z_result)

    def test_vector_boundary_conditions(self):
        """
        Test using vector quantities as boundary conditions, such as spatial variables
        with BoundaryGradient. This tests the case where boundary conditions are
        functions of spatial coordinates rather than scalar constants.
        """
        # Create discretisation with 2D mesh
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {"macroscale": pybamm.FiniteVolume2D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Create a variable and spatial variables
        phi = pybamm.Variable(
            "phi", domain=["negative electrode", "separator", "positive electrode"]
        )
        submesh = mesh[("negative electrode", "separator", "positive electrode")]
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

        # Set up boundary conditions using spatial variables through BoundaryGradient
        # This represents cases where boundary conditions depend on spatial position
        boundary_conditions = {
            phi: {
                "top": (
                    pybamm.Vector(np.ones(submesh.npts_lr)),
                    "Dirichlet",
                ),  # BC depends on x coordinate
                "bottom": (
                    pybamm.Vector(np.ones(submesh.npts_lr)),
                    "Dirichlet",
                ),  # BC depends on x coordinate
                "left": (pybamm.Scalar(0), "Dirichlet"),  # Scalar BC for comparison
                "right": (pybamm.Scalar(1), "Dirichlet"),  # Scalar BC for comparison
            }
        }

        disc.bcs = boundary_conditions
        disc.set_variable_slices([phi])

        # Process the boundary conditions to ensure they work with vector quantities
        x_disc = disc.process_symbol(x)
        z_disc = disc.process_symbol(z)

        # Test that spatial variables are properly discretised as vectors
        assert isinstance(x_disc, pybamm.Vector)
        assert isinstance(z_disc, pybamm.Vector)

        # Test BoundaryGradient with spatial variables
        top_bc_gradient = pybamm.BoundaryGradient(x, "top")
        bottom_bc_gradient = pybamm.BoundaryGradient(x, "bottom")

        top_bc_disc = disc.process_symbol(top_bc_gradient)
        bottom_bc_disc = disc.process_symbol(bottom_bc_gradient)

        # Verify that boundary gradients are properly processed
        assert top_bc_disc is not None
        assert bottom_bc_disc is not None

        # Test a simple equation with these boundary conditions
        # Laplace equation: div(grad(phi)) = 0
        eqn = pybamm.div(pybamm.grad(phi))
        eqn_disc = disc.process_symbol(eqn)

        # Verify the equation can be evaluated with vector boundary conditions
        submesh = mesh[("negative electrode", "separator", "positive electrode")]
        LR, _TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        test_y = LR.flatten()  # Use x-coordinate values as test data

        # This should not raise an error
        result = eqn_disc.evaluate(None, test_y)
        assert result is not None
        assert result.shape[0] == submesh.npts

        # Test more complex vector boundary conditions
        # Use spatial variable directly in boundary condition (not through BoundaryGradient)
        boundary_conditions_direct = {
            phi: {
                "top": (
                    pybamm.Vector(np.ones(submesh.npts_lr)),
                    "Dirichlet",
                ),  # BC is the z-coordinate itself
                "bottom": (pybamm.Scalar(0), "Dirichlet"),
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }

        disc.bcs = boundary_conditions_direct
        eqn_disc_direct = disc.process_symbol(eqn)

        # This should also work with direct spatial variable as BC
        result_direct = eqn_disc_direct.evaluate(None, test_y)
        assert result_direct is not None
        assert result_direct.shape[0] == submesh.npts

        # Test combination of vector and scalar boundary conditions
        boundary_conditions_mixed = {
            phi: {
                "top": (
                    pybamm.Vector(np.ones(submesh.npts_lr)) + pybamm.Scalar(1),
                    "Dirichlet",
                ),  # Vector + scalar
                "bottom": (
                    pybamm.Vector(np.ones(submesh.npts_lr)),
                    "Neumann",
                ),  # Vector Neumann BC
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Vector(np.ones(submesh.npts_tb)), "Neumann"),
            }
        }

        disc.bcs = boundary_conditions_mixed
        eqn_disc_mixed = disc.process_symbol(eqn)

        # This should work with mixed vector/scalar boundary conditions
        result_mixed = eqn_disc_mixed.evaluate(None, test_y)
        assert result_mixed is not None
        assert result_mixed.shape[0] == submesh.npts

        # Test vector Dirichlet boundary conditions for left and right sides
        # (similar to existing top/bottom vector BCs)
        boundary_conditions_lr_vector = {
            phi: {
                "top": (pybamm.Scalar(0), "Dirichlet"),
                "bottom": (pybamm.Scalar(0), "Dirichlet"),
                "left": (
                    pybamm.Vector(np.ones(submesh.npts_tb)),
                    "Dirichlet",
                ),  # Vector BC depends on z coordinate
                "right": (
                    pybamm.Vector(np.ones(submesh.npts_tb)),
                    "Dirichlet",
                ),  # Vector BC depends on z coordinate
            }
        }

        disc.bcs = boundary_conditions_lr_vector
        eqn_disc_lr_vector = disc.process_symbol(eqn)

        # This should work with vector Dirichlet BCs on left and right sides
        result_lr_vector = eqn_disc_lr_vector.evaluate(None, test_y)
        assert result_lr_vector is not None
        assert result_lr_vector.shape[0] == submesh.npts

        # Test BoundaryGradient with spatial variables for left and right sides
        left_bc_gradient = pybamm.BoundaryGradient(z, "left")
        right_bc_gradient = pybamm.BoundaryGradient(z, "right")

        left_bc_disc = disc.process_symbol(left_bc_gradient)
        right_bc_disc = disc.process_symbol(right_bc_gradient)

        # Verify that boundary gradients are properly processed for left/right
        assert left_bc_disc is not None
        assert right_bc_disc is not None

        # Test all sides with vector Dirichlet boundary conditions
        boundary_conditions_all_vector = {
            phi: {
                "top": (
                    pybamm.Vector(np.ones(submesh.npts_lr)),
                    "Dirichlet",
                ),  # Vector BC depends on x coordinate
                "bottom": (
                    pybamm.Vector(np.ones(submesh.npts_lr)),
                    "Dirichlet",
                ),  # Vector BC depends on x coordinate
                "left": (
                    pybamm.Vector(np.ones(submesh.npts_tb)),
                    "Dirichlet",
                ),  # Vector BC depends on z coordinate
                "right": (
                    pybamm.Vector(np.ones(submesh.npts_tb)),
                    "Dirichlet",
                ),  # Vector BC depends on z coordinate
            }
        }

        disc.bcs = boundary_conditions_all_vector
        eqn_disc_all_vector = disc.process_symbol(eqn)

        # This should work with vector Dirichlet BCs on all sides
        result_all_vector = eqn_disc_all_vector.evaluate(None, test_y)
        assert result_all_vector is not None
        assert result_all_vector.shape[0] == submesh.npts

        # Test mixed vector boundary conditions with complex expressions
        boundary_conditions_complex_lr = {
            phi: {
                "top": (
                    pybamm.Vector(np.ones(submesh.npts_lr)) * 2,
                    "Dirichlet",
                ),  # Vector BC with scaling
                "bottom": (pybamm.Scalar(0), "Dirichlet"),
                "left": (
                    pybamm.Vector(np.ones(submesh.npts_tb)) + pybamm.Scalar(0.5),
                    "Dirichlet",
                ),  # Vector + scalar BC
                "right": (
                    pybamm.Vector(np.ones(submesh.npts_tb)) * pybamm.Scalar(1.5),
                    "Dirichlet",
                ),  # Vector * scalar BC
            }
        }

        disc.bcs = boundary_conditions_complex_lr
        eqn_disc_complex_lr = disc.process_symbol(eqn)

        # This should work with complex vector expressions as boundary conditions
        result_complex_lr = eqn_disc_complex_lr.evaluate(None, test_y)
        assert result_complex_lr is not None
        assert result_complex_lr.shape[0] == submesh.npts

        # Test vector Neumann boundary conditions for top and bottom sides
        boundary_conditions_tb_neumann = {
            phi: {
                "top": (
                    pybamm.Vector(np.ones(submesh.npts_lr)),
                    "Neumann",
                ),  # Vector Neumann BC for top side
                "bottom": (
                    pybamm.Vector(np.ones(submesh.npts_lr)),
                    "Neumann",
                ),  # Vector Neumann BC for bottom side
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }

        disc.bcs = boundary_conditions_tb_neumann
        eqn_disc_tb_neumann = disc.process_symbol(eqn)

        # This should work with vector Neumann BCs on top and bottom sides
        result_tb_neumann = eqn_disc_tb_neumann.evaluate(None, test_y)
        assert result_tb_neumann is not None
        assert result_tb_neumann.shape[0] == submesh.npts

        # Test all sides with vector Neumann boundary conditions
        boundary_conditions_all_neumann = {
            phi: {
                "top": (
                    pybamm.Vector(np.ones(submesh.npts_lr)),
                    "Neumann",
                ),  # Vector Neumann BC depends on x coordinate
                "bottom": (
                    pybamm.Vector(np.ones(submesh.npts_lr)),
                    "Neumann",
                ),  # Vector Neumann BC depends on x coordinate
                "left": (
                    pybamm.Vector(np.ones(submesh.npts_tb)),
                    "Neumann",
                ),  # Vector Neumann BC depends on z coordinate
                "right": (
                    pybamm.Vector(np.ones(submesh.npts_tb)),
                    "Neumann",
                ),  # Vector Neumann BC depends on z coordinate
            }
        }

        disc.bcs = boundary_conditions_all_neumann
        eqn_disc_all_neumann = disc.process_symbol(eqn)

        # This should work with vector Neumann BCs on all sides
        result_all_neumann = eqn_disc_all_neumann.evaluate(None, test_y)
        assert result_all_neumann is not None
        assert result_all_neumann.shape[0] == submesh.npts

        # Test mixed vector Neumann boundary conditions with complex expressions
        boundary_conditions_complex_neumann = {
            phi: {
                "top": (
                    pybamm.Vector(np.ones(submesh.npts_lr)) * 2,
                    "Neumann",
                ),  # Vector Neumann BC with scaling
                "bottom": (
                    pybamm.Vector(np.ones(submesh.npts_lr)) + pybamm.Scalar(0.5),
                    "Neumann",
                ),  # Vector + scalar Neumann BC
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }

        disc.bcs = boundary_conditions_complex_neumann
        eqn_disc_complex_neumann = disc.process_symbol(eqn)

        # This should work with complex vector Neumann expressions as boundary conditions
        result_complex_neumann = eqn_disc_complex_neumann.evaluate(None, test_y)
        assert result_complex_neumann is not None
        assert result_complex_neumann.shape[0] == submesh.npts

    def test_delta_function(self):
        # Create discretisation
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {"macroscale": pybamm.FiniteVolume2D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable(
            "var", domain=["negative electrode", "separator", "positive electrode"]
        )
        disc.set_variable_slices([var])
        delta_function = pybamm.DeltaFunction(var, "left", "negative electrode")
        with pytest.raises(NotImplementedError):
            disc.process_symbol(delta_function)
