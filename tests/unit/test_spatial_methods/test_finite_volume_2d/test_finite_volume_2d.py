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
        result_none = spatial_method.upwind_or_downwind(
            var, var_disc, boundary_conditions, None, None
        )

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
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
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
            pybamm.ModelError, match="Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions, "upwind", None
            )

        # Should raise error when downwinding without right Dirichlet BC
        with pytest.raises(
            pybamm.ModelError, match="Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions, "downwind", None
            )

        # Should raise error when upwinding in tb without top Dirichlet BC
        with pytest.raises(
            pybamm.ModelError, match="Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions, None, "upwind"
            )

        # Should raise error when downwinding in tb without bottom Dirichlet BC
        with pytest.raises(
            pybamm.ModelError, match="Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions, None, "downwind"
            )

        # Test 3: Error case - no boundary conditions at all
        with pytest.raises(
            pybamm.ModelError, match="Boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(var, var_disc, {}, "upwind", None)

        # Test 4: Error case - invalid direction
        with pytest.raises(ValueError, match="direction 'invalid' not recognised"):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions, "invalid", None
            )

        with pytest.raises(ValueError, match="direction 'invalid' not recognised"):
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
            pybamm.ModelError, match="Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions_no_left_dirichlet, "upwind", None
            )

        # Test that downwind requires right Dirichlet BC
        with pytest.raises(
            pybamm.ModelError, match="Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions_no_left_dirichlet, "downwind", None
            )

        # Test that upwind in tb requires top Dirichlet BC
        with pytest.raises(
            pybamm.ModelError, match="Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions_no_left_dirichlet, None, "upwind"
            )

        # Test that downwind in tb requires bottom Dirichlet BC
        with pytest.raises(
            pybamm.ModelError, match="Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions_no_left_dirichlet, None, "downwind"
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
            pybamm.ModelError, match="Dirichlet boundary conditions must be provided"
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
            pybamm.ModelError, match="Dirichlet boundary conditions must be provided"
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
            pybamm.ModelError, match="Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions_top_dirichlet_only, None, "downwind"
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
            pybamm.ModelError, match="Dirichlet boundary conditions must be provided"
        ):
            spatial_method.upwind_or_downwind(
                var, var_disc, boundary_conditions_bottom_dirichlet_only, None, "upwind"
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
            var, var_disc, boundary_conditions_minimal_tb, None, "upwind"
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
            var, var_disc, boundary_conditions_both_minimal, "upwind", "upwind"
        )
        # Verify it's a VectorField
        assert isinstance(result_both_ghost, pybamm.VectorField)
