#
# Test for the operator class
#
import pybamm
import tests.shared as shared

import numpy as np
import unittest


class TestFiniteVolumeDiscretisation(unittest.TestCase):
    def test_node_to_edge(self):
        a = pybamm.Symbol("a")

        def arithmetic_mean(array):
            return (array[1:] + array[:-1]) / 2

        ava = pybamm.NodeToEdge(a, arithmetic_mean)
        self.assertEqual(ava.name, "node to edge (arithmetic_mean)")
        self.assertEqual(ava.children[0].name, a.name)

        b = pybamm.Scalar(-4)
        avb = pybamm.NodeToEdge(b, arithmetic_mean)
        self.assertEqual(avb.evaluate(), -4)

        c = pybamm.Vector(np.ones(10))
        avc = pybamm.NodeToEdge(c, arithmetic_mean)
        np.testing.assert_array_equal(avc.evaluate(), np.ones(9))

        d = pybamm.StateVector(slice(0, 10))
        y_test = np.ones(10)
        avd = pybamm.NodeToEdge(d, arithmetic_mean)
        np.testing.assert_array_equal(avd.evaluate(None, y_test), np.ones(9))

    def test_discretise_diffusivity_times_spatial_operator(self):
        # Set up
        whole_cell = ["negative electrode", "separator", "positive electrode"]

        # create discretisation
        defaults = shared.TestDefaults1DMacro()
        disc = pybamm.FiniteVolumeDiscretisation(
            defaults.mesh_type, defaults.submesh_pts, defaults.submesh_types
        )
        disc.mesh_geometry(defaults.geometry)
        mesh = disc.mesh

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        # Discretise some equations where averaging is needed
        var = pybamm.Variable("var", domain=whole_cell)
        y_slices = disc.get_variable_slices([var])
        y_test = np.ones_like(combined_submesh.nodes)
        for eqn in [
            var * pybamm.grad(var),
            var ** 2 * pybamm.grad(var),
            var * pybamm.grad(var) ** 2,
            var * (pybamm.grad(var) + 2),
            (pybamm.grad(var) + 2) * (-var),
            (pybamm.grad(var) + 2) * (2 * var),
            pybamm.grad(var) * pybamm.grad(var),
            (pybamm.grad(var) + 2) * pybamm.grad(var) ** 2,
        ]:
            eqn_disc = disc.process_symbol(eqn, y_slices, {})
            # Check that the equation can be evaluated
            eqn_disc.evaluate(None, y_test)

        # more testing, with boundary conditions
        for flux, eqn in zip(
            [
                pybamm.grad(var),
                pybamm.grad(var),
                pybamm.grad(var),
                2 * pybamm.grad(var),
                2 * pybamm.grad(var),
                var * pybamm.grad(var) + 2 * pybamm.grad(var),
            ],
            [
                pybamm.div(pybamm.grad(var)),
                pybamm.div(pybamm.grad(var)) + 2,
                pybamm.div(pybamm.grad(var)) + var,
                pybamm.div(2 * pybamm.grad(var)),
                pybamm.div(2 * pybamm.grad(var)) + 3 * var,
                -2 * pybamm.div(var * pybamm.grad(var) + 2 * pybamm.grad(var)),
            ],
        ):
            eqn_disc = disc.process_symbol(
                eqn,
                y_slices,
                {flux.id: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(1)}},
            )
            # Check that the equation can be evaluated
            eqn_disc.evaluate(None, y_test)

    def test_add_ghost_nodes(self):
        # Set up

        # create discretisation
        defaults = shared.TestDefaults1DMacro()
        disc = pybamm.FiniteVolumeDiscretisation(
            defaults.mesh_type, defaults.submesh_pts, defaults.submesh_types
        )
        disc.mesh_geometry(defaults.geometry)
        mesh = disc.mesh

        # Add ghost nodes
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        y_slices = disc.get_variable_slices([var])
        discretised_symbol = pybamm.StateVector(y_slices[var.id])
        lbc = pybamm.Scalar(0)
        rbc = pybamm.Scalar(3)
        symbol_plus_ghost = disc.add_ghost_nodes(discretised_symbol, lbc, rbc)

        # Test
        combined_submesh = mesh.combine_submeshes(*whole_cell)
        y_test = np.ones_like(combined_submesh.nodes)
        np.testing.assert_array_equal(
            symbol_plus_ghost.evaluate(None, y_test)[1:-1],
            discretised_symbol.evaluate(None, y_test),
        )
        self.assertEqual(
            (
                symbol_plus_ghost.evaluate(None, y_test)[0]
                + symbol_plus_ghost.evaluate(None, y_test)[1]
            )
            / 2,
            0,
        )
        self.assertEqual(
            (
                symbol_plus_ghost.evaluate(None, y_test)[-2]
                + symbol_plus_ghost.evaluate(None, y_test)[-1]
            )
            / 2,
            3,
        )

    def test_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        """
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        # create discretisation
        defaults = shared.TestDefaults1DMacro()
        disc = pybamm.FiniteVolumeDiscretisation(
            defaults.mesh_type, defaults.submesh_pts, defaults.submesh_types
        )
        disc.mesh_geometry(defaults.geometry)
        mesh = disc.mesh

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        mesh.add_ghost_meshes()
        disc.mesh.add_ghost_meshes()

        # grad
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {"left": pybamm.Scalar(1), "right": pybamm.Scalar(1)}
        }
        y_slices = disc.get_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn, y_slices, boundary_conditions)

        constant_y = np.ones_like(combined_submesh.nodes)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh.edges),
        )

        # div: test on linear y (should have laplacian zero) so change bcs
        linear_y = combined_submesh.nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(1)}
        }

        grad_eqn_disc = disc.process_symbol(grad_eqn, y_slices, boundary_conditions)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y), np.ones_like(combined_submesh.edges)
        )

        div_eqn_disc = disc.process_symbol(div_eqn, y_slices, boundary_conditions)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y), np.zeros_like(combined_submesh.nodes)
        )

    def test_grad_div_shapes_Neumann_bcs(self):
        """Test grad and div with Neumann boundary conditions (applied by div on N)"""
        whole_cell = ["negative electrode", "separator", "positive electrode"]

        # create discretisation
        defaults = shared.TestDefaults1DMacro()
        disc = pybamm.FiniteVolumeDiscretisation(
            defaults.mesh_type, defaults.submesh_pts, defaults.submesh_types
        )
        disc.mesh_geometry(defaults.geometry)
        mesh = disc.mesh

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        # grad
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        y_slices = disc.get_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn, y_slices, {})

        constant_y = np.ones_like(combined_submesh.nodes)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh.edges[1:-1]),
        )

        # div
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(1), "right": pybamm.Scalar(1)}
        }
        div_eqn_disc = disc.process_symbol(div_eqn, y_slices, boundary_conditions)

        # Linear y should have laplacian zero
        linear_y = combined_submesh.nodes
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh.edges[1:-1]),
        )
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y), np.zeros_like(combined_submesh.nodes)
        )

    def test_grad_div_shapes_mixed_domain(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        """
        # create discretisation
        defaults = shared.TestDefaults1DMacro()
        disc = pybamm.FiniteVolumeDiscretisation(
            defaults.mesh_type, defaults.submesh_pts, defaults.submesh_types
        )
        disc.mesh_geometry(defaults.geometry)
        mesh = disc.mesh

        mesh.add_ghost_meshes()
        disc.mesh.add_ghost_meshes()

        # grad
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {"left": pybamm.Scalar(1), "right": pybamm.Scalar(1)}
        }
        y_slices = disc.get_variable_slices([var])

        grad_eqn_disc = disc.process_symbol(grad_eqn, y_slices, boundary_conditions)

        combined_submesh = mesh.combine_submeshes("negative electrode", "separator")
        constant_y = np.ones_like(combined_submesh.nodes)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh.edges),
        )

        # div: test on linear y (should have laplacian zero) so change bcs
        linear_y = combined_submesh.nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {
                "left": pybamm.Scalar(0),
                "right": pybamm.Scalar(combined_submesh.edges[-1]),
            }
        }

        grad_eqn_disc = disc.process_symbol(grad_eqn, y_slices, boundary_conditions)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y), np.ones_like(combined_submesh.edges)
        )

        div_eqn_disc = disc.process_symbol(div_eqn, y_slices, boundary_conditions)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y), np.zeros_like(combined_submesh.nodes)
        )

    def test_grad_convergence_without_bcs(self):
        # Convergence
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)

        # Function for convergence testing
        def get_l2_error(n):
            # Set up discretisation
            n = 3 * round(n / 3)
            # create discretisation
            defaults = shared.TestDefaults1DMacro()
            defaults.set_equal_pts(n)
            disc = pybamm.FiniteVolumeDiscretisation(
                defaults.mesh_type, defaults.submesh_pts, defaults.submesh_types
            )
            disc.mesh_geometry(defaults.geometry)
            mesh = disc.mesh

            combined_submesh = mesh.combine_submeshes(*whole_cell)
            # Define exact solutions
            y = np.sin(combined_submesh.nodes)
            grad_exact = np.cos(combined_submesh.edges[1:-1])

            # Discretise and evaluate
            y_slices = disc.get_variable_slices([var])
            grad_eqn_disc = disc.process_symbol(grad_eqn, y_slices, {})
            grad_approx = grad_eqn_disc.evaluate(None, y)

            # Calculate errors
            return np.linalg.norm(grad_approx - grad_exact) / np.linalg.norm(grad_exact)

        # Get errors
        ns = 100 * (2 ** np.arange(2, 7))
        errs = np.array([get_l2_error(int(n)) for n in ns])

        # Get rates: expect h**2 convergence
        rates = np.log2(errs[:-1] / errs[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)

    def test_grad_convergence_with_bcs(self):
        # Convergence
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {
                "left": pybamm.Scalar(np.sin(0)),
                "right": pybamm.Scalar(np.sin(1)),
            }
        }

        # Function for convergence testing
        def get_l2_error(n):
            # create discretisation
            defaults = shared.TestDefaults1DMacro()
            defaults.set_equal_pts(n)
            disc = pybamm.FiniteVolumeDiscretisation(
                defaults.mesh_type, defaults.submesh_pts, defaults.submesh_types
            )
            disc.mesh_geometry(defaults.geometry)
            mesh = disc.mesh

            # Set up discretisation
            whole_cell = ["negative electrode", "separator", "positive electrode"]
            combined_submesh = mesh.combine_submeshes(*whole_cell)

            mesh.add_ghost_meshes()
            disc.mesh.add_ghost_meshes()

            # Define exact solutions
            y = np.sin(combined_submesh.nodes)
            grad_exact = np.cos(combined_submesh.edges)

            # Discretise and evaluate
            y_slices = disc.get_variable_slices([var])
            grad_eqn_disc = disc.process_symbol(grad_eqn, y_slices, boundary_conditions)
            grad_approx = grad_eqn_disc.evaluate(None, y)

            # Calculate errors
            return np.linalg.norm(grad_approx - grad_exact) / np.linalg.norm(grad_exact)

        # Get errors
        ns = 100 * (2 ** np.arange(2, 7))
        ns = 3 * np.round(ns / 3)
        errs = np.array([get_l2_error(int(n)) for n in ns])

        # Get rates: expect h**1.5 convergence
        rates = np.log2(errs[:-1] / errs[1:])
        np.testing.assert_array_less(1.49 * np.ones_like(rates), rates)

    def test_div_convergence_internal(self):
        # Convergence
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(np.cos(0)), "right": pybamm.Scalar(np.cos(1))}
        }

        # Function for convergence testing
        def get_l2_error(n):

            # create discretisation
            defaults = shared.TestDefaults1DMacro()
            defaults.set_equal_pts(n)
            disc = pybamm.FiniteVolumeDiscretisation(
                defaults.mesh_type, defaults.submesh_pts, defaults.submesh_types
            )
            disc.mesh_geometry(defaults.geometry)
            mesh = disc.mesh

            whole_cell = ["negative electrode", "separator", "positive electrode"]
            combined_submesh = mesh.combine_submeshes(*whole_cell)

            # Define exact solutions
            y = np.sin(combined_submesh.nodes)
            div_exact_internal = -np.sin(combined_submesh.nodes[1:-1])

            # Discretise and evaluate
            y_slices = disc.get_variable_slices([var])
            div_eqn_disc = disc.process_symbol(div_eqn, y_slices, boundary_conditions)
            div_approx_internal = div_eqn_disc.evaluate(None, y)[1:-1]

            # Calculate errors
            return np.linalg.norm(
                div_approx_internal - div_exact_internal
            ) / np.linalg.norm(div_exact_internal)

        # Get errors
        ns = 10 * (2 ** np.arange(2, 6))
        ns = 3 * np.round(ns / 3)
        errs = np.array([get_l2_error(int(n)) for n in ns])

        # Get rates: expect h**2 convergence
        rates = np.log2(errs[:-1] / errs[1:])
        np.testing.assert_array_less(1.9 * np.ones_like(rates), rates)

    def test_div_convergence(self):
        # Convergence
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(np.cos(0)), "right": pybamm.Scalar(np.cos(1))}
        }

        # Function for convergence testing
        def get_l2_error(n):
            whole_cell = ["negative electrode", "separator", "positive electrode"]
            # create discretisation
            defaults = shared.TestDefaults1DMacro()
            defaults.set_equal_pts(n)
            disc = pybamm.FiniteVolumeDiscretisation(
                defaults.mesh_type, defaults.submesh_pts, defaults.submesh_types
            )
            disc.mesh_geometry(defaults.geometry)
            mesh = disc.mesh

            combined_submesh = mesh.combine_submeshes(*whole_cell)

            # Define exact solutions
            y = np.sin(combined_submesh.nodes)
            div_exact_internal = -np.sin(combined_submesh.nodes[1:-1])

            # Discretise and evaluate
            y_slices = disc.get_variable_slices([var])
            div_eqn_disc = disc.process_symbol(div_eqn, y_slices, boundary_conditions)
            div_approx_internal = div_eqn_disc.evaluate(None, y)[1:-1]

            # Calculate errors
            return np.linalg.norm(
                div_approx_internal - div_exact_internal
            ) / np.linalg.norm(div_exact_internal)

        # Get errors
        ns = 10 * (2 ** np.arange(2, 6))
        ns = 3 * np.round(ns / 3)
        errs = np.array([get_l2_error(int(n)) for n in ns])

        # Get rates: expect h**1.5 convergence because of boundary conditions
        rates = np.log2(errs[:-1] / errs[1:])
        np.testing.assert_array_less(1.49 * np.ones_like(rates), rates)

    def test_spherical_operators(self):
        # test div( grad( sin(r) )) == (2/r)*cos(r) - *sin(r)

        domain = ["negative particle"]
        c = pybamm.Variable("c", domain=domain)
        N = pybamm.grad(c)
        eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(np.cos(0)), "right": pybamm.Scalar(np.cos(1))}
        }

        def get_l2_error(n):
            defaults = shared.TestDefaults1DParticle(n)

            disc = pybamm.FiniteVolumeDiscretisation(
                defaults.mesh_type, defaults.submesh_pts, defaults.submesh_types
            )
            disc.mesh_geometry(defaults.geometry)
            mesh = disc.mesh["negative particle"]
            r = mesh.nodes

            # exact solution
            y = np.sin(r)
            exact = (2 / r) * np.cos(r) - np.sin(r)
            exact_internal = exact[1:-1]

            # discretise and evaluate
            y_slices = disc.get_variable_slices([c])
            eqn_disc = disc.process_symbol(eqn, y_slices, boundary_conditions)
            approx_internal = eqn_disc.evaluate(None, y)[1:-1]

            # error
            error = np.linalg.norm(approx_internal - exact_internal) / np.linalg.norm(
                exact_internal
            )
            return error

        # Get errors
        ns = 10 * (2 ** np.arange(2, 7))
        errs = np.array([get_l2_error(int(n)) for n in ns])

        # Get rates: expect h**1.5 convergence because of boundary conditions
        rates = np.log2(errs[:-1] / errs[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
