#
# Tests for the Finite Volume Method
#
from tests import TestCase
import pybamm
from tests import (
    get_mesh_for_testing,
    get_p2d_mesh_for_testing,
    get_1p1d_mesh_for_testing,
)
import numpy as np
from scipy.sparse import kron, eye
import unittest


class TestFiniteVolume(TestCase):
    def test_node_to_edge_to_node(self):
        # Create discretisation
        mesh = get_mesh_for_testing()
        fin_vol = pybamm.FiniteVolume()
        fin_vol.build(mesh)
        n = mesh["negative electrode"].npts

        # node to edge
        c = pybamm.StateVector(slice(0, n), domain=["negative electrode"])
        y_test = np.ones(n)
        diffusivity_c_ari = fin_vol.node_to_edge(c, method="arithmetic")
        np.testing.assert_array_equal(
            diffusivity_c_ari.evaluate(None, y_test), np.ones((n + 1, 1))
        )
        diffusivity_c_har = fin_vol.node_to_edge(c, method="harmonic")
        np.testing.assert_array_equal(
            diffusivity_c_har.evaluate(None, y_test), np.ones((n + 1, 1))
        )

        # edge to node
        d = pybamm.StateVector(slice(0, n + 1), domain=["negative electrode"])
        y_test = np.ones(n + 1)
        diffusivity_d_ari = fin_vol.edge_to_node(d, method="arithmetic")
        np.testing.assert_array_equal(
            diffusivity_d_ari.evaluate(None, y_test), np.ones((n, 1))
        )
        diffusivity_d_har = fin_vol.edge_to_node(d, method="harmonic")
        np.testing.assert_array_equal(
            diffusivity_d_har.evaluate(None, y_test), np.ones((n, 1))
        )

        # bad shift key
        with self.assertRaisesRegex(ValueError, "shift key"):
            fin_vol.shift(c, "bad shift key", "arithmetic")

        with self.assertRaisesRegex(ValueError, "shift key"):
            fin_vol.shift(c, "bad shift key", "harmonic")

        # bad method
        with self.assertRaisesRegex(ValueError, "method"):
            fin_vol.shift(c, "shift key", "bad method")

    def test_concatenation(self):
        mesh = get_mesh_for_testing()
        fin_vol = pybamm.FiniteVolume()
        fin_vol.build(mesh)

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        edges = [pybamm.Vector(mesh[dom].edges, domain=dom) for dom in whole_cell]
        # Concatenation of edges should get averaged to nodes first, using edge_to_node
        v_disc = fin_vol.concatenation(edges)
        np.testing.assert_array_equal(v_disc.evaluate()[:, 0], mesh[whole_cell].nodes)

        # test for bad shape
        edges = [
            pybamm.Vector(np.ones(mesh[dom].npts + 2), domain=dom) for dom in whole_cell
        ]
        with self.assertRaisesRegex(pybamm.ShapeError, "child must have size n_nodes"):
            fin_vol.concatenation(edges)

    def test_discretise_diffusivity_times_spatial_operator(self):
        # Setup mesh and discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        submesh = mesh[whole_cell]

        # Discretise some equations where averaging is needed
        var = pybamm.Variable("var", domain=whole_cell)
        disc.set_variable_slices([var])
        y_test = np.ones_like(submesh.nodes[:, np.newaxis])
        for eqn in [
            var * pybamm.grad(var),
            var**2 * pybamm.grad(var),
            var * pybamm.grad(var) ** 2,
            var * (pybamm.grad(var) + 2),
            (pybamm.grad(var) + 2) * (-var),
            (pybamm.grad(var) + 2) * (2 * var),
            pybamm.grad(var) * pybamm.grad(var),
            (pybamm.grad(var) + 2) * pybamm.grad(var) ** 2,
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
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            # Neumann
            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            # One of each
            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Dirichlet"),
                    "right": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(1), "Dirichlet"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)

    def test_discretise_spatial_variable(self):
        # Create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # macroscale
        x1 = pybamm.SpatialVariable("x", ["negative electrode"])
        x1_disc = disc.process_symbol(x1)
        self.assertIsInstance(x1_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            x1_disc.evaluate(), disc.mesh["negative electrode"].nodes[:, np.newaxis]
        )
        # macroscale with concatenation
        x2 = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        x2_disc = disc.process_symbol(x2)
        self.assertIsInstance(x2_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            x2_disc.evaluate(),
            disc.mesh[("negative electrode", "separator")].nodes[:, np.newaxis],
        )
        # microscale
        r = 3 * pybamm.SpatialVariable("r", ["negative particle"])
        r_disc = disc.process_symbol(r)
        self.assertIsInstance(r_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            r_disc.evaluate(), 3 * disc.mesh["negative particle"].nodes[:, np.newaxis]
        )

    def test_mass_matrix_shape(self):
        # Create model
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        N = pybamm.grad(c)
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N)}
        model.initial_conditions = {c: pybamm.Scalar(0)}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        model.variables = {"c": c, "N": N}

        # Create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]
        disc.process_model(model)

        # Mass matrix
        mass = np.eye(submesh.npts)
        np.testing.assert_array_equal(mass, model.mass_matrix.entries.toarray())

    def test_p2d_mass_matrix_shape(self):
        # Create model
        c = pybamm.Variable(
            "c",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )
        N = pybamm.grad(c)
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N)}
        model.initial_conditions = {c: pybamm.Scalar(0)}
        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Dirichlet")}
        }
        model.variables = {"c": c, "N": N}

        # Create discretisation
        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        # Mass matrix
        prim_pts = mesh["negative particle"].npts
        sec_pts = mesh["negative electrode"].npts
        mass_local = eye(prim_pts)
        mass = kron(eye(sec_pts), mass_local)
        np.testing.assert_array_equal(
            mass.toarray(), model.mass_matrix.entries.toarray()
        )

    def test_jacobian(self):
        # Create discretisation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]
        spatial_method = pybamm.FiniteVolume()
        spatial_method.build(mesh)

        # Setup variable
        var = pybamm.Variable("var", domain=whole_cell)
        disc.set_variable_slices([var])
        y = pybamm.StateVector(slice(0, submesh.npts))
        y_test = np.ones_like(submesh.nodes[:, np.newaxis])

        # grad
        eqn = pybamm.grad(var)
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(2), "Dirichlet"),
            }
        }
        eqn_disc = disc.process_symbol(eqn)
        eqn_jac = eqn_disc.jac(y)
        jacobian = eqn_jac.evaluate(y=y_test)
        grad_matrix = spatial_method.gradient_matrix(
            whole_cell, {"primary": whole_cell}
        ).entries
        np.testing.assert_allclose(jacobian.toarray()[1:-1], grad_matrix.toarray())
        np.testing.assert_allclose(
            jacobian.toarray()[0, 0], grad_matrix.toarray()[0][0] * -2
        )
        np.testing.assert_allclose(
            jacobian.toarray()[-1, -1], grad_matrix.toarray()[-1][-1] * -2
        )

        # grad with averaging
        eqn = var * pybamm.grad(var)
        eqn_disc = disc.process_symbol(eqn)
        eqn_jac = eqn_disc.jac(y)
        eqn_jac.evaluate(y=y_test)

        # div(grad)
        flux = pybamm.grad(var)
        eqn = pybamm.div(flux)
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(2), "Neumann"),
            }
        }
        eqn_disc = disc.process_symbol(eqn)
        eqn_jac = eqn_disc.jac(y)
        eqn_jac.evaluate(y=y_test)

        # div(grad) with averaging
        flux = var * pybamm.grad(var)
        eqn = pybamm.div(flux)
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(2), "Neumann"),
            }
        }
        eqn_disc = disc.process_symbol(eqn)
        eqn_jac = eqn_disc.jac(y)
        eqn_jac.evaluate(y=y_test)

    def test_boundary_value_domain(self):
        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        c_s_n = pybamm.Variable(
            "c_s_n",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )
        c_s_p = pybamm.Variable(
            "c_s_p",
            domain=["positive particle"],
            auxiliary_domains={"secondary": ["positive electrode"]},
        )

        disc.set_variable_slices([c_s_n, c_s_p])

        # surface values
        c_s_n_surf = pybamm.surf(c_s_n)
        c_s_p_surf = pybamm.surf(c_s_p)
        c_s_n_surf_disc = disc.process_symbol(c_s_n_surf)
        c_s_p_surf_disc = disc.process_symbol(c_s_p_surf)
        self.assertEqual(c_s_n_surf_disc.domain, ["negative electrode"])
        self.assertEqual(c_s_p_surf_disc.domain, ["positive electrode"])

    def test_delta_function(self):
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var = pybamm.Variable("var")
        delta_fn_left = pybamm.DeltaFunction(var, "left", "negative electrode")
        delta_fn_right = pybamm.DeltaFunction(var, "right", "negative electrode")
        disc.set_variable_slices([var])
        delta_fn_left_disc = disc.process_symbol(delta_fn_left)
        delta_fn_right_disc = disc.process_symbol(delta_fn_right)

        # Basic shape and type tests
        y = np.ones_like(mesh["negative electrode"].nodes[:, np.newaxis])
        # Left
        self.assertEqual(delta_fn_left_disc.domains, delta_fn_left.domains)
        self.assertIsInstance(delta_fn_left_disc, pybamm.Multiplication)
        self.assertIsInstance(delta_fn_left_disc.left, pybamm.Matrix)
        np.testing.assert_array_equal(delta_fn_left_disc.left.evaluate()[:, 1:], 0)
        self.assertEqual(delta_fn_left_disc.shape, y.shape)
        # Right
        self.assertEqual(delta_fn_right_disc.domains, delta_fn_right.domains)
        self.assertIsInstance(delta_fn_right_disc, pybamm.Multiplication)
        self.assertIsInstance(delta_fn_right_disc.left, pybamm.Matrix)
        np.testing.assert_array_equal(delta_fn_right_disc.left.evaluate()[:, :-1], 0)
        self.assertEqual(delta_fn_right_disc.shape, y.shape)

        # Value tests
        # Delta function should integrate to the same thing as variable
        var_disc = disc.process_symbol(var)
        x = pybamm.standard_spatial_vars.x_n
        delta_fn_int_disc = disc.process_symbol(pybamm.Integral(delta_fn_left, x))
        np.testing.assert_allclose(
            var_disc.evaluate(y=y) * mesh["negative electrode"].edges[-1],
            np.sum(delta_fn_int_disc.evaluate(y=y)),
        )

    def test_heaviside(self):
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var = pybamm.Variable("var", domain="negative electrode")
        heav = var > 1

        disc.set_variable_slices([var])
        # process_binary_operators should work with heaviside
        disc_heav = disc.process_symbol(heav * var)
        nodes = mesh["negative electrode"].nodes
        self.assertEqual(disc_heav.size, nodes.size)
        np.testing.assert_array_equal(disc_heav.evaluate(y=2 * np.ones_like(nodes)), 2)
        np.testing.assert_array_equal(disc_heav.evaluate(y=-2 * np.ones_like(nodes)), 0)

    def test_upwind_downwind(self):
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        n = mesh["negative electrode"].npts
        var = pybamm.StateVector(slice(0, n), domain="negative electrode")
        upwind = pybamm.upwind(var)
        downwind = pybamm.downwind(var)

        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(5), "Dirichlet"),
                "right": (pybamm.Scalar(3), "Dirichlet"),
            }
        }

        disc_upwind = disc.process_symbol(upwind)
        disc_downwind = disc.process_symbol(downwind)

        nodes = mesh["negative electrode"].nodes
        n = mesh["negative electrode"].npts
        self.assertEqual(disc_upwind.size, nodes.size + 1)
        self.assertEqual(disc_downwind.size, nodes.size + 1)

        y_test = 2 * np.ones_like(nodes)
        np.testing.assert_array_equal(
            disc_upwind.evaluate(y=y_test),
            np.concatenate([np.array([5, 0.5]), 2 * np.ones(n - 1)])[:, np.newaxis],
        )
        np.testing.assert_array_equal(
            disc_downwind.evaluate(y=y_test),
            np.concatenate([2 * np.ones(n - 1), np.array([1.5, 3])])[:, np.newaxis],
        )

        # Remove boundary conditions and check error is raised
        disc.bcs = {}
        disc._discretised_symbols = {}
        with self.assertRaisesRegex(pybamm.ModelError, "Boundary conditions"):
            disc.process_symbol(upwind)

        # Set wrong boundary conditions and check error is raised
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(5), "Neumann"),
                "right": (pybamm.Scalar(3), "Neumann"),
            }
        }
        with self.assertRaisesRegex(pybamm.ModelError, "Dirichlet boundary conditions"):
            disc.process_symbol(upwind)
        with self.assertRaisesRegex(pybamm.ModelError, "Dirichlet boundary conditions"):
            disc.process_symbol(downwind)

    def test_grad_div_with_bcs_on_tab(self):
        # Create discretisation
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # var
        y_test = np.ones(mesh["current collector"].npts)
        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])
        # grad
        grad_eqn = pybamm.grad(var)
        # div
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)

        # bcs (on each tab)
        boundary_conditions = {
            var: {
                "negative tab": (pybamm.Scalar(1), "Dirichlet"),
                "positive tab": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        grad_eqn_disc.evaluate(None, y_test)
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eqn_disc.evaluate(None, y_test)

        # bcs (one pos, one not tab)
        boundary_conditions = {
            var: {
                "no tab": (pybamm.Scalar(1), "Dirichlet"),
                "positive tab": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        grad_eqn_disc.evaluate(None, y_test)
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eqn_disc.evaluate(None, y_test)

        # bcs (one neg, one not tab)
        boundary_conditions = {
            var: {
                "negative tab": (pybamm.Scalar(1), "Neumann"),
                "no tab": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        grad_eqn_disc.evaluate(None, y_test)
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eqn_disc.evaluate(None, y_test)

    def test_neg_pos_bcs(self):
        # Create discretisation
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # var
        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])
        # grad
        grad_eqn = pybamm.grad(var)

        # bcs (on each tab)
        boundary_conditions = {
            var: {
                "negative tab": (pybamm.Scalar(1), "Dirichlet"),
                "positive tab": (pybamm.Scalar(0), "Neumann"),
                "no tab": (pybamm.Scalar(8), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        # check after disc that negative tab goes to left and positive tab goes
        # to right
        disc.process_symbol(grad_eqn)
        self.assertEqual(disc.bcs[var]["left"][0], pybamm.Scalar(1))
        self.assertEqual(disc.bcs[var]["left"][1], "Dirichlet")
        self.assertEqual(disc.bcs[var]["right"][0], pybamm.Scalar(0))
        self.assertEqual(disc.bcs[var]["right"][1], "Neumann")

    def test_full_broadcast_domains(self):
        model = pybamm.BaseModel()
        var = pybamm.Variable(
            "var", domain=["negative electrode", "separator"], scale=100
        )
        model.rhs = {var: 0}
        a = pybamm.InputParameter("a")
        ic = pybamm.concatenation(
            pybamm.FullBroadcast(a * 100, "negative electrode"),
            pybamm.FullBroadcast(100, "separator"),
        )
        model.initial_conditions = {var: ic}

        mesh = get_mesh_for_testing()
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume(),
            "separator": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
