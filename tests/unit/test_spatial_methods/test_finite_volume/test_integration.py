#
# Tests for integration using Finite Volume method
#
import pybamm
from tests import (
    get_mesh_for_testing,
    get_1p1d_mesh_for_testing,
    get_cylindrical_mesh_for_testing,
)
import numpy as np
import unittest


class TestFiniteVolumeIntegration(unittest.TestCase):
    def test_definite_integral(self):
        # create discretisation
        mesh = get_mesh_for_testing(xpts=200, rpts=200)
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.ZeroDimensionalSpatialMethod(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        # lengths
        ln = mesh["negative electrode"].edges[-1]
        ls = mesh["separator"].edges[-1] - ln
        lp = 1 - (ln + ls)

        # macroscale variable
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        integral_eqn = pybamm.Integral(var, x)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)
        submesh = mesh[("negative electrode", "separator")]

        constant_y = np.ones_like(submesh.nodes[:, np.newaxis])
        self.assertEqual(integral_eqn_disc.evaluate(None, constant_y), ln + ls)
        linear_y = submesh.nodes
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, linear_y), (ln + ls) ** 2 / 2
        )
        cos_y = np.cos(submesh.nodes[:, np.newaxis])
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, cos_y), np.sin(ln + ls), decimal=4
        )

        # domain not starting at zero
        var = pybamm.Variable("var", domain=["separator", "positive electrode"])
        x = pybamm.SpatialVariable("x", ["separator", "positive electrode"])
        integral_eqn = pybamm.Integral(var, x)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)
        submesh = mesh[("separator", "positive electrode")]

        constant_y = np.ones_like(submesh.nodes[:, np.newaxis])
        self.assertEqual(integral_eqn_disc.evaluate(None, constant_y), ls + lp)
        linear_y = submesh.nodes
        self.assertAlmostEqual(
            integral_eqn_disc.evaluate(None, linear_y)[0][0], (1 - (ln) ** 2) / 2
        )
        cos_y = np.cos(submesh.nodes[:, np.newaxis])
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, cos_y), np.sin(1) - np.sin(ln), decimal=4
        )

        # microscale variable
        var = pybamm.Variable("var", domain=["negative particle"])
        r = pybamm.SpatialVariable("r", ["negative particle"])
        integral_eqn = pybamm.Integral(var, r)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        constant_y = np.ones_like(mesh["negative particle"].nodes[:, np.newaxis])
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, constant_y), 4 * np.pi / 3
        )
        linear_y = mesh["negative particle"].nodes
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, linear_y), np.pi, decimal=4
        )
        one_over_y = 1 / mesh["negative particle"].nodes
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, one_over_y), 2 * np.pi, decimal=3
        )

        # cylindrical coordinates
        mesh = get_cylindrical_mesh_for_testing(rcellpts=200)
        spatial_methods = {"current collector": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var = pybamm.Variable("var", domain=["current collector"])
        r = pybamm.SpatialVariable("r", ["current collector"])
        integral_eqn = pybamm.Integral(var, r)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)
        pts = mesh["current collector"].nodes
        r0 = mesh["current collector"].edges[0]

        constant_y = np.ones_like(pts[:, np.newaxis])
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, constant_y), np.pi * (1 - r0**2)
        )
        linear_y = pts
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, linear_y),
            2 * np.pi / 3 * (1 - r0**3),
            decimal=4,
        )
        one_over_y = 1 / pts
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, one_over_y),
            2 * np.pi * (1 - r0),
            decimal=3,
        )

        # test failure for secondary dimension column form
        finite_volume = pybamm.FiniteVolume()
        finite_volume.build(mesh)
        with self.assertRaisesRegex(
            NotImplementedError,
            "Integral in secondary vector only implemented in 'row' form",
        ):
            finite_volume.definite_integral_matrix(var, "column", "secondary")

    def test_integral_secondary_domain(self):
        # create discretisation
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        # lengths
        ln = mesh["negative electrode"].edges[-1]
        ls = mesh["separator"].edges[-1] - ln
        lp = 1 - (ln + ls)

        var = pybamm.Variable(
            "var",
            domain="positive particle",
            auxiliary_domains={
                "secondary": "positive electrode",
                "tertiary": "current collector",
            },
        )
        x = pybamm.SpatialVariable("x", "positive electrode")
        integral_eqn = pybamm.Integral(var, x)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        submesh = mesh["positive particle"]
        constant_y = np.ones(
            (
                submesh.npts
                * mesh["positive electrode"].npts
                * mesh["current collector"].npts,
                1,
            )
        )
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, constant_y),
            lp * np.ones((submesh.npts * mesh["current collector"].npts, 1)),
        )
        linear_in_x = np.tile(
            np.repeat(mesh["positive electrode"].nodes, submesh.npts),
            mesh["current collector"].npts,
        )
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, linear_in_x),
            (1 - (ln + ls) ** 2)
            / 2
            * np.ones((submesh.npts * mesh["current collector"].npts, 1)),
        )
        linear_in_r = np.tile(
            submesh.nodes,
            mesh["positive electrode"].npts * mesh["current collector"].npts,
        )
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, linear_in_r).flatten(),
            lp * np.tile(submesh.nodes, mesh["current collector"].npts),
        )
        cos_y = np.cos(linear_in_x)
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, cos_y),
            (np.sin(1) - np.sin(ln + ls))
            * np.ones((submesh.npts * mesh["current collector"].npts, 1)),
            decimal=4,
        )

    def test_integral_primary_then_secondary_same_result(self):
        # Test that integrating in r then in x gives the same result as integrating in
        # x then in r
        # create discretisation
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var = pybamm.Variable(
            "var",
            domain="positive particle",
            auxiliary_domains={
                "secondary": "positive electrode",
                "tertiary": "current collector",
            },
        )
        x = pybamm.SpatialVariable("x", "positive electrode")
        r = pybamm.SpatialVariable("r", "positive particle")
        integral_eqn_x_then_r = pybamm.Integral(pybamm.Integral(var, x), r)
        integral_eqn_r_then_x = pybamm.Integral(pybamm.Integral(var, r), x)

        # discretise
        disc.set_variable_slices([var])
        integral_eqn_x_then_r_disc = disc.process_symbol(integral_eqn_x_then_r)
        integral_eqn_r_then_x_disc = disc.process_symbol(integral_eqn_r_then_x)

        # test
        submesh = mesh["positive particle"]
        cos_y = np.cos(
            np.tile(
                submesh.nodes,
                mesh["positive electrode"].npts * mesh["current collector"].npts,
            )
        )
        np.testing.assert_array_almost_equal(
            integral_eqn_x_then_r_disc.evaluate(None, cos_y),
            integral_eqn_r_then_x_disc.evaluate(None, cos_y),
            decimal=4,
        )

    def test_integral_secondary_domain_on_edges_in_primary_domain(self):
        # create discretisation
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        # lengths
        ln = mesh["negative electrode"].edges[-1]
        ls = mesh["separator"].edges[-1] - ln
        lp = 1 - (ln + ls)

        r_edge = pybamm.SpatialVariableEdge(
            "r_p",
            domain="positive particle",
            auxiliary_domains={
                "secondary": "positive electrode",
                "tertiary": "current collector",
            },
        )

        x = pybamm.SpatialVariable("x", "positive electrode")
        integral_eqn = pybamm.Integral(r_edge, x)
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        submesh = mesh["positive particle"]
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate().flatten(),
            lp
            * np.tile(
                np.linspace(0, 1, submesh.npts + 1), mesh["current collector"].npts
            ),
        )

    def test_definite_integral_vector(self):
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="negative electrode")
        disc.set_variable_slices([var])

        # row (default)
        vec = pybamm.DefiniteIntegralVector(var)
        vec_disc = disc.process_symbol(vec)
        self.assertEqual(vec_disc.shape[0], 1)
        self.assertEqual(vec_disc.shape[1], mesh["negative electrode"].npts)

        # column
        vec = pybamm.DefiniteIntegralVector(var, vector_type="column")
        vec_disc = disc.process_symbol(vec)
        self.assertEqual(vec_disc.shape[0], mesh["negative electrode"].npts)
        self.assertEqual(vec_disc.shape[1], 1)

    def test_indefinite_integral(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.ZeroDimensionalSpatialMethod(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # input a phi, take grad, then integrate to recover phi approximation
        # (need to test this way as check evaluated on edges using if has grad
        # and no div)
        phi = pybamm.Variable("phi", domain=["negative electrode", "separator"])
        i = pybamm.grad(phi)  # create test current (variable on edges)

        x = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        int_grad_phi = pybamm.IndefiniteIntegral(i, x)
        disc.set_variable_slices([phi])  # i is not a fundamental variable
        # Set boundary conditions (required for shape but don't matter)
        disc.bcs = {
            phi: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        int_grad_phi_disc = disc.process_symbol(int_grad_phi)
        left_boundary_value = pybamm.BoundaryValue(int_grad_phi, "left")
        left_boundary_value_disc = disc.process_symbol(left_boundary_value)

        submesh = mesh[("negative electrode", "separator")]

        # constant case
        phi_exact = np.ones((submesh.npts, 1))
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        phi_approx += 1  # add constant of integration
        np.testing.assert_array_equal(phi_exact, phi_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=phi_exact), 0)
        # linear case
        phi_exact = submesh.nodes[:, np.newaxis]
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_almost_equal(phi_exact, phi_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=phi_exact), 0)

        # sine case
        phi_exact = np.sin(submesh.nodes[:, np.newaxis])
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_almost_equal(phi_exact, phi_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=phi_exact), 0)

        # --------------------------------------------------------------------
        # region which doesn't start at zero
        phi = pybamm.Variable("phi", domain=["separator", "positive electrode"])
        i = pybamm.grad(phi)  # create test current (variable on edges)
        x = pybamm.SpatialVariable("x", ["separator", "positive electrode"])
        int_grad_phi = pybamm.IndefiniteIntegral(i, x)
        disc.set_variable_slices([phi])  # i is not a fundamental variable
        disc.bcs = {
            phi: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        int_grad_phi_disc = disc.process_symbol(int_grad_phi)
        left_boundary_value = pybamm.BoundaryValue(int_grad_phi, "left")
        left_boundary_value_disc = disc.process_symbol(left_boundary_value)
        submesh = mesh[("separator", "positive electrode")]

        # constant case
        phi_exact = np.ones((submesh.npts, 1))
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        phi_approx += 1  # add constant of integration
        np.testing.assert_array_equal(phi_exact, phi_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=phi_exact), 0)

        # linear case
        phi_exact = submesh.nodes[:, np.newaxis] - submesh.edges[0]
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_almost_equal(phi_exact, phi_approx)
        np.testing.assert_array_almost_equal(
            left_boundary_value_disc.evaluate(y=phi_exact), 0
        )

        # sine case
        phi_exact = np.sin(submesh.nodes[:, np.newaxis] - submesh.edges[0])
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_almost_equal(phi_exact, phi_approx)
        np.testing.assert_array_almost_equal(
            left_boundary_value_disc.evaluate(y=phi_exact), 0
        )

        # --------------------------------------------------------------------
        # indefinite integral of a spatial variable
        x = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        x_edge = pybamm.SpatialVariableEdge("x", ["negative electrode", "separator"])
        int_x = pybamm.IndefiniteIntegral(x, x)
        int_x_edge = pybamm.IndefiniteIntegral(x_edge, x)

        x_disc = disc.process_symbol(x)
        x_edge_disc = disc.process_symbol(x_edge)
        int_x_disc = disc.process_symbol(int_x)
        int_x_edge_disc = disc.process_symbol(int_x_edge)

        np.testing.assert_almost_equal(
            int_x_disc.evaluate(), x_edge_disc.evaluate() ** 2 / 2
        )
        np.testing.assert_almost_equal(
            int_x_edge_disc.evaluate(), x_disc.evaluate() ** 2 / 2, decimal=4
        )

        # --------------------------------------------------------------------
        # micrsoscale case
        c = pybamm.Variable("c", domain=["negative particle"])
        N = pybamm.grad(c)  # create test flux (variable on edges)
        r_n = pybamm.SpatialVariable("r_n", ["negative particle"])
        c_integral = pybamm.IndefiniteIntegral(N, r_n)
        disc.set_variable_slices([c])  # N is not a fundamental variable
        disc.bcs = {
            c: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }

        c_integral_disc = disc.process_symbol(c_integral)
        left_boundary_value = pybamm.BoundaryValue(c_integral, "left")
        left_boundary_value_disc = disc.process_symbol(left_boundary_value)
        submesh = mesh["negative particle"]

        # constant case
        c_exact = np.ones((submesh.npts, 1))
        c_approx = c_integral_disc.evaluate(None, c_exact)
        c_approx += 1  # add constant of integration
        np.testing.assert_array_equal(c_exact, c_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=c_exact), 0)

        # linear case
        c_exact = submesh.nodes[:, np.newaxis]
        c_approx = c_integral_disc.evaluate(None, c_exact)
        np.testing.assert_array_almost_equal(c_exact, c_approx)
        np.testing.assert_array_almost_equal(
            left_boundary_value_disc.evaluate(y=c_exact), 0
        )

        # sine case
        c_exact = np.sin(submesh.nodes[:, np.newaxis])
        c_approx = c_integral_disc.evaluate(None, c_exact)
        np.testing.assert_array_almost_equal(c_exact, c_approx, decimal=3)
        np.testing.assert_array_almost_equal(
            left_boundary_value_disc.evaluate(y=c_exact), 0
        )

    def test_backward_indefinite_integral(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # --------------------------------------------------------------------
        # region which doesn't start at zero
        phi = pybamm.Variable("phi", domain=["negative electrode", "separator"])
        i = pybamm.grad(phi)  # create test current (variable on edges)
        x = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        int_grad_phi = pybamm.BackwardIndefiniteIntegral(i, x)
        disc.set_variable_slices([phi])  # i is not a fundamental variable
        disc.bcs = {
            phi: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        int_grad_phi_disc = disc.process_symbol(int_grad_phi)
        right_boundary_value = pybamm.BoundaryValue(int_grad_phi, "right")
        right_boundary_value_disc = disc.process_symbol(right_boundary_value)
        submesh = mesh[("negative electrode", "separator")]

        # Test that the backward_integral(grad(phi)) = -phi
        # constant case
        phi_exact = np.ones((submesh.npts, 1))
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        phi_approx += 1  # add constant of integration
        np.testing.assert_array_equal(phi_exact, phi_approx)
        self.assertEqual(right_boundary_value_disc.evaluate(y=phi_exact), 0)

        # linear case
        phi_exact = submesh.nodes - submesh.edges[-1]
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact).flatten()
        np.testing.assert_array_almost_equal(phi_exact, -phi_approx)
        np.testing.assert_array_almost_equal(
            right_boundary_value_disc.evaluate(y=phi_exact), 0
        )

        # sine case
        phi_exact = np.sin(submesh.nodes - submesh.edges[-1])
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact).flatten()
        np.testing.assert_array_almost_equal(phi_exact, -phi_approx)
        np.testing.assert_array_almost_equal(
            right_boundary_value_disc.evaluate(y=phi_exact), 0
        )

    def test_indefinite_integral_of_broadcasted_to_cell_edges(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.ZeroDimensionalSpatialMethod(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # make a variable 'phi' and a vector 'i' which is broadcast onto edges
        # the integral of this should then be put onto the nodes
        phi = pybamm.Variable("phi", domain=["negative electrode", "separator"])
        i = pybamm.PrimaryBroadcastToEdges(1, phi.domain)
        x = pybamm.SpatialVariable("x", phi.domain)
        disc.set_variable_slices([phi])
        submesh = mesh[("negative electrode", "separator")]
        x_end = submesh.edges[-1]

        # take indefinite integral
        int_phi = pybamm.IndefiniteIntegral(i * phi, x)
        # take integral again
        int_int_phi = pybamm.Integral(int_phi, x)
        int_int_phi_disc = disc.process_symbol(int_int_phi)

        # constant case
        phi_exact = np.ones_like(submesh.nodes)
        phi_approx = int_int_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_almost_equal(x_end**2 / 2, phi_approx)

        # linear case
        phi_exact = submesh.nodes[:, np.newaxis]
        phi_approx = int_int_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_almost_equal(x_end**3 / 6, phi_approx, decimal=4)

    def test_indefinite_integral_on_nodes(self):
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        phi = pybamm.Variable("phi", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", ["negative electrode", "separator"])

        int_phi = pybamm.IndefiniteIntegral(phi, x)
        disc.set_variable_slices([phi])
        int_phi_disc = disc.process_symbol(int_phi)

        submesh = mesh[("negative electrode", "separator")]

        # constant case
        phi_exact = np.ones((submesh.npts, 1))
        int_phi_exact = submesh.edges
        int_phi_approx = int_phi_disc.evaluate(None, phi_exact).flatten()
        np.testing.assert_array_equal(int_phi_exact, int_phi_approx)
        # linear case
        phi_exact = submesh.nodes
        int_phi_exact = submesh.edges**2 / 2
        int_phi_approx = int_phi_disc.evaluate(None, phi_exact).flatten()
        np.testing.assert_array_almost_equal(int_phi_exact, int_phi_approx)
        # cos case
        phi_exact = np.cos(submesh.nodes)
        int_phi_exact = np.sin(submesh.edges)
        int_phi_approx = int_phi_disc.evaluate(None, phi_exact).flatten()
        np.testing.assert_array_almost_equal(int_phi_exact, int_phi_approx, decimal=5)

        # microscale case should fail
        mesh = get_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        c = pybamm.Variable("c", domain=["negative particle"])
        r = pybamm.SpatialVariable("r", ["negative particle"])

        int_c = pybamm.IndefiniteIntegral(c, r)
        disc.set_variable_slices([c])
        with self.assertRaisesRegex(
            NotImplementedError,
            "Indefinite integral on a spherical polar domain is not implemented",
        ):
            disc.process_symbol(int_c)

    def test_backward_indefinite_integral_on_nodes(self):
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        phi = pybamm.Variable("phi", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", ["negative electrode", "separator"])

        back_int_phi = pybamm.BackwardIndefiniteIntegral(phi, x)
        disc.set_variable_slices([phi])
        back_int_phi_disc = disc.process_symbol(back_int_phi)

        submesh = mesh[("negative electrode", "separator")]
        edges = submesh.edges

        # constant case
        phi_exact = np.ones((submesh.npts, 1))
        back_int_phi_exact = edges[-1] - edges
        back_int_phi_approx = back_int_phi_disc.evaluate(None, phi_exact).flatten()
        np.testing.assert_array_almost_equal(back_int_phi_exact, back_int_phi_approx)
        # linear case
        phi_exact = submesh.nodes
        back_int_phi_exact = edges[-1] ** 2 / 2 - edges**2 / 2
        back_int_phi_approx = back_int_phi_disc.evaluate(None, phi_exact).flatten()
        np.testing.assert_array_almost_equal(back_int_phi_exact, back_int_phi_approx)
        # cos case
        phi_exact = np.cos(submesh.nodes)
        back_int_phi_exact = np.sin(edges[-1]) - np.sin(edges)
        back_int_phi_approx = back_int_phi_disc.evaluate(None, phi_exact).flatten()
        np.testing.assert_array_almost_equal(
            back_int_phi_exact, back_int_phi_approx, decimal=5
        )

    def test_forward_plus_backward_integral(self):
        # Test that forward integral + backward integral = integral
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # On nodes
        phi = pybamm.Variable("phi", domain=["separator", "positive electrode"])
        x = pybamm.SpatialVariable("x", ["separator", "positive electrode"])

        disc.set_variable_slices([phi])

        full_int_phi = pybamm.PrimaryBroadcastToEdges(
            pybamm.Integral(phi, x), ["separator", "positive electrode"]
        )
        full_int_phi_disc = disc.process_symbol(full_int_phi)
        int_plus_back_int_phi = pybamm.IndefiniteIntegral(
            phi, x
        ) + pybamm.BackwardIndefiniteIntegral(phi, x)
        int_plus_back_int_phi_disc = disc.process_symbol(int_plus_back_int_phi)

        submesh = mesh[("separator", "positive electrode")]

        # test
        for phi_exact in [
            np.ones((submesh.npts, 1)),
            submesh.nodes,
            np.cos(submesh.nodes),
        ]:
            np.testing.assert_array_almost_equal(
                full_int_phi_disc.evaluate(y=phi_exact).flatten(),
                int_plus_back_int_phi_disc.evaluate(y=phi_exact).flatten(),
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
