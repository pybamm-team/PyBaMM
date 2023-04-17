#
# Test for adding ghost nodes in finite volumes class
#
from tests import TestCase
import pybamm
from tests import get_mesh_for_testing, get_p2d_mesh_for_testing
import numpy as np
import unittest


class TestGhostNodes(TestCase):
    def test_add_ghost_nodes(self):
        # Set up

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Add ghost nodes
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        disc.set_variable_slices([var])
        discretised_symbol = pybamm.StateVector(*disc.y_slices[var])
        bcs = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(3), "Dirichlet"),
        }

        # Test
        sp_meth = pybamm.FiniteVolume()
        sp_meth.build(mesh)
        sym_ghost, _ = sp_meth.add_ghost_nodes(var, discretised_symbol, bcs)
        submesh = mesh[whole_cell]
        y_test = np.linspace(0, 1, submesh.npts)
        np.testing.assert_array_equal(
            sym_ghost.evaluate(y=y_test)[1:-1], discretised_symbol.evaluate(y=y_test)
        )
        self.assertEqual(
            (sym_ghost.evaluate(y=y_test)[0] + sym_ghost.evaluate(y=y_test)[1]) / 2, 0
        )
        self.assertEqual(
            (sym_ghost.evaluate(y=y_test)[-2] + sym_ghost.evaluate(y=y_test)[-1]) / 2, 3
        )

        # test errors
        bcs = {"left": (pybamm.Scalar(0), "x"), "right": (pybamm.Scalar(3), "Neumann")}
        with self.assertRaisesRegex(ValueError, "boundary condition must be"):
            sp_meth.add_ghost_nodes(var, discretised_symbol, bcs)
        with self.assertRaisesRegex(ValueError, "boundary condition must be"):
            sp_meth.add_neumann_values(var, discretised_symbol, bcs, var.domain)
        bcs = {"left": (pybamm.Scalar(0), "Neumann"), "right": (pybamm.Scalar(3), "x")}
        with self.assertRaisesRegex(ValueError, "boundary condition must be"):
            sp_meth.add_ghost_nodes(var, discretised_symbol, bcs)
        with self.assertRaisesRegex(ValueError, "boundary condition must be"):
            sp_meth.add_neumann_values(var, discretised_symbol, bcs, var.domain)

    def test_add_ghost_nodes_concatenation(self):
        # Set up

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Add ghost nodes
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var_n = pybamm.Variable("var", domain=["negative electrode"])
        var_s = pybamm.Variable("var", domain=["separator"])
        var_p = pybamm.Variable("var", domain=["positive electrode"])
        var = pybamm.concatenation(var_n, var_s, var_p)
        disc.set_variable_slices([var])
        discretised_symbol = disc.process_symbol(var)
        bcs = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(3), "Dirichlet"),
        }

        # Test
        submesh = mesh[whole_cell]
        y_test = np.ones_like(submesh.nodes[:, np.newaxis])

        # both
        sp_meth = pybamm.FiniteVolume()
        sp_meth.build(mesh)
        symbol_plus_ghost_both, _ = sp_meth.add_ghost_nodes(
            var, discretised_symbol, bcs
        )
        np.testing.assert_array_equal(
            symbol_plus_ghost_both.evaluate(None, y_test)[1:-1],
            discretised_symbol.evaluate(None, y_test),
        )
        self.assertEqual(
            (
                symbol_plus_ghost_both.evaluate(None, y_test)[0]
                + symbol_plus_ghost_both.evaluate(None, y_test)[1]
            )
            / 2,
            0,
        )
        self.assertEqual(
            (
                symbol_plus_ghost_both.evaluate(None, y_test)[-2]
                + symbol_plus_ghost_both.evaluate(None, y_test)[-1]
            )
            / 2,
            3,
        )

    def test_p2d_add_ghost_nodes(self):
        # create discretisation
        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # add ghost nodes
        c_s_n = pybamm.Variable(
            "c_s_n",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )
        c_s_p = pybamm.Variable(
            "c_s_p",
            domain=["positive particle"],
            auxiliary_domains={"secondary": "positive electrode"},
        )

        disc.set_variable_slices([c_s_n])
        disc_c_s_n = pybamm.StateVector(*disc.y_slices[c_s_n])

        disc.set_variable_slices([c_s_p])
        disc_c_s_p = pybamm.StateVector(*disc.y_slices[c_s_p])
        bcs = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(3), "Dirichlet"),
        }
        sp_meth = pybamm.FiniteVolume()
        sp_meth.build(mesh)
        c_s_n_plus_ghost, _ = sp_meth.add_ghost_nodes(c_s_n, disc_c_s_n, bcs)
        c_s_p_plus_ghost, _ = sp_meth.add_ghost_nodes(c_s_p, disc_c_s_p, bcs)

        mesh_s_n = mesh["negative particle"]
        mesh_s_p = mesh["positive particle"]

        n_prim_pts = mesh_s_n.npts
        n_sec_pts = mesh["negative electrode"].npts

        p_prim_pts = mesh_s_p.npts
        p_sec_pts = mesh["positive electrode"].npts

        y_s_n_test = np.kron(np.ones(n_sec_pts), np.ones(n_prim_pts))
        y_s_p_test = np.kron(np.ones(p_sec_pts), np.ones(p_prim_pts))

        # evaluate with and without ghost points
        c_s_n_eval = disc_c_s_n.evaluate(None, y_s_n_test)
        c_s_n_ghost_eval = c_s_n_plus_ghost.evaluate(None, y_s_n_test)

        c_s_p_eval = disc_c_s_p.evaluate(None, y_s_p_test)
        c_s_p_ghost_eval = c_s_p_plus_ghost.evaluate(None, y_s_p_test)

        # reshape to make easy to deal with
        c_s_n_eval = np.reshape(c_s_n_eval, [n_sec_pts, n_prim_pts])
        c_s_n_ghost_eval = np.reshape(c_s_n_ghost_eval, [n_sec_pts, n_prim_pts + 2])

        c_s_p_eval = np.reshape(c_s_p_eval, [p_sec_pts, p_prim_pts])
        c_s_p_ghost_eval = np.reshape(c_s_p_ghost_eval, [p_sec_pts, p_prim_pts + 2])

        np.testing.assert_array_equal(c_s_n_ghost_eval[:, 1:-1], c_s_n_eval)
        np.testing.assert_array_equal(c_s_p_ghost_eval[:, 1:-1], c_s_p_eval)

        np.testing.assert_array_equal(
            (c_s_n_ghost_eval[:, 0] + c_s_n_ghost_eval[:, 1]) / 2, 0
        )
        np.testing.assert_array_equal(
            (c_s_p_ghost_eval[:, 0] + c_s_p_ghost_eval[:, 1]) / 2, 0
        )

        np.testing.assert_array_equal(
            (c_s_n_ghost_eval[:, -2] + c_s_n_ghost_eval[:, -1]) / 2, 3
        )
        np.testing.assert_array_equal(
            (c_s_p_ghost_eval[:, -2] + c_s_p_ghost_eval[:, -1]) / 2, 3
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
