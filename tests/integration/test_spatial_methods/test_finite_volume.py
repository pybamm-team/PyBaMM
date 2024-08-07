#
# Test for the operator class
#

import pybamm
from tests import (
    get_mesh_for_testing,
    get_p2d_mesh_for_testing,
    get_cylindrical_mesh_for_testing,
)

import numpy as np
import unittest


class TestFiniteVolumeConvergence(unittest.TestCase):
    def test_grad_div_broadcast(self):
        # create mesh and discretisation
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        mesh = get_mesh_for_testing()
        disc = pybamm.Discretisation(mesh, spatial_methods)

        a = pybamm.PrimaryBroadcast(1, "negative electrode")
        grad_a = disc.process_symbol(pybamm.grad(a))
        np.testing.assert_array_equal(grad_a.evaluate(), 0)

        a_edge = pybamm.PrimaryBroadcastToEdges(1, "negative electrode")
        div_a = disc.process_symbol(pybamm.div(a_edge))
        np.testing.assert_array_equal(div_a.evaluate(), 0)

        div_grad_a = disc.process_symbol(pybamm.div(pybamm.grad(a)))
        np.testing.assert_array_equal(div_grad_a.evaluate(), 0)

    def test_cartesian_spherical_grad_convergence(self):
        # note that grad function is the same for cartesian and spherical
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        whole_cell = ["negative electrode", "separator", "positive electrode"]

        # Define variable
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(np.sin(1) ** 2), "Dirichlet"),
            }
        }

        # Function for convergence testing
        def get_error(n):
            # create mesh and discretisation
            mesh = get_mesh_for_testing(n)
            disc = pybamm.Discretisation(mesh, spatial_methods)
            disc.bcs = boundary_conditions
            disc.set_variable_slices([var])

            # Define exact solutions
            submesh = mesh[whole_cell]
            x = submesh.nodes
            y = np.sin(x) ** 2
            # var = sin(x)**2 --> dvardx = 2*sin(x)*cos(x)
            x_edge = submesh.edges
            grad_exact = 2 * np.sin(x_edge) * np.cos(x_edge)

            # Discretise and evaluate
            grad_eqn_disc = disc.process_symbol(grad_eqn)
            grad_approx = grad_eqn_disc.evaluate(y=y)

            # Return difference between approx and exact
            return grad_approx[:, 0] - grad_exact

        # Get errors
        ns = 100 * 2 ** np.arange(6)
        errs = {n: get_error(int(n)) for n in ns}
        # expect quadratic convergence at internal points
        errs_internal = np.array([np.linalg.norm(errs[n][1:-1], np.inf) for n in ns])
        rates = np.log2(errs_internal[:-1] / errs_internal[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)
        # expect linear convergence at the boundaries
        for idx in [0, -1]:
            err_boundary = np.array([errs[n][idx] for n in ns])
            rates = np.log2(err_boundary[:-1] / err_boundary[1:])
            np.testing.assert_array_less(0.98 * np.ones_like(rates), rates)

    def test_cartesian_div_convergence(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}

        # Function for convergence testing
        def get_error(n):
            # create mesh and discretisation
            mesh = get_mesh_for_testing(n)
            disc = pybamm.Discretisation(mesh, spatial_methods)
            submesh = mesh[whole_cell]
            x = submesh.nodes
            x_edge = pybamm.standard_spatial_vars.x_edge

            # Define flux and eqn
            N = x_edge**2 * pybamm.cos(x_edge)
            div_eqn = pybamm.div(N)
            # Define exact solutions
            # N = x**2 * cos(x) --> dNdx = x*(2cos(x) - xsin(x))
            div_exact = x * (2 * np.cos(x) - x * np.sin(x))

            # Discretise and evaluate
            div_eqn_disc = disc.process_symbol(div_eqn)
            div_approx = div_eqn_disc.evaluate()

            # Return difference between approx and exact
            return div_approx[:, 0] - div_exact

        # Get errors
        ns = 10 * 2 ** np.arange(1, 6)
        errs = {n: get_error(int(n)) for n in ns}
        # expect quadratic convergence everywhere
        err_norm = np.array([np.linalg.norm(errs[n], np.inf) for n in ns])
        rates = np.log2(err_norm[:-1] / err_norm[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)

    def test_cylindrical_div_convergence_quadratic(self):
        # N = sin(r) --> div(N) = sin(r)/r + cos(r)
        spatial_methods = {"current collector": pybamm.FiniteVolume()}

        # Function for convergence testing
        def get_error(n):
            # create mesh and discretisation (single particle)
            mesh = get_cylindrical_mesh_for_testing(rcellpts=n)
            disc = pybamm.Discretisation(mesh, spatial_methods)
            submesh = mesh["current collector"]
            r = submesh.nodes
            r_edge = pybamm.SpatialVariableEdge("r", domain=["current collector"])

            # Define flux and eqn
            N = pybamm.sin(r_edge)
            div_eqn = pybamm.div(N)
            # Define exact solutions
            div_exact = np.sin(r) / r + np.cos(r)

            # Discretise and evaluate
            div_eqn_disc = disc.process_symbol(div_eqn)
            div_approx = div_eqn_disc.evaluate()

            # Return difference between approx and exact
            return div_approx[:, 0] - div_exact

        # Get errors
        ns = 10 * 2 ** np.arange(1, 7)
        errs = {n: get_error(int(n)) for n in ns}
        # expect quadratic convergence everywhere
        err_norm = np.array([np.linalg.norm(errs[n], np.inf) for n in ns])
        rates = np.log2(err_norm[:-1] / err_norm[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)

    def test_spherical_div_convergence_quadratic(self):
        # N = sin(r) --> div(N) = 2*sin(r)/r + cos(r)
        spatial_methods = {"negative particle": pybamm.FiniteVolume()}

        # Function for convergence testing
        def get_error(n):
            # create mesh and discretisation (single particle)
            mesh = get_mesh_for_testing(rpts=n)
            disc = pybamm.Discretisation(mesh, spatial_methods)
            submesh = mesh["negative particle"]
            r = submesh.nodes
            r_edge = pybamm.SpatialVariableEdge("r_n", domain=["negative particle"])

            # Define flux and eqn
            N = pybamm.sin(r_edge)
            div_eqn = pybamm.div(N)
            # Define exact solutions
            div_exact = 2 / r * np.sin(r) + np.cos(r)

            # Discretise and evaluate
            div_eqn_disc = disc.process_symbol(div_eqn)
            div_approx = div_eqn_disc.evaluate()

            # Return difference between approx and exact
            return div_approx[:, 0] - div_exact

        # Get errors
        ns = 10 * 2 ** np.arange(1, 7)
        errs = {n: get_error(int(n)) for n in ns}
        # expect quadratic convergence everywhere
        err_norm = np.array([np.linalg.norm(errs[n], np.inf) for n in ns])
        rates = np.log2(err_norm[:-1] / err_norm[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)

    def test_spherical_div_convergence_linear(self):
        # N = r*sin(r) --> div(N) = 3*sin(r) + r*cos(r)
        spatial_methods = {"negative particle": pybamm.FiniteVolume()}

        # Function for convergence testing
        def get_error(n):
            # create mesh and discretisation (single particle)
            mesh = get_mesh_for_testing(rpts=n)
            disc = pybamm.Discretisation(mesh, spatial_methods)
            submesh = mesh["negative particle"]
            r = submesh.nodes
            r_edge = pybamm.SpatialVariableEdge("r_n", domain=["negative particle"])

            # Define flux and eqn
            N = r_edge * pybamm.sin(r_edge)
            div_eqn = pybamm.div(N)
            # Define exact solutions
            div_exact = 3 * np.sin(r) + r * np.cos(r)

            # Discretise and evaluate
            div_eqn_disc = disc.process_symbol(div_eqn)
            div_approx = div_eqn_disc.evaluate()

            # Return difference between approx and exact
            return div_approx[:, 0] - div_exact

        # Get errors
        ns = 10 * 2 ** np.arange(6)
        errs = {n: get_error(int(n)) for n in ns}
        # expect linear convergence everywhere
        err_norm = np.array([np.linalg.norm(errs[n], np.inf) for n in ns])
        rates = np.log2(err_norm[:-1] / err_norm[1:])
        np.testing.assert_array_less(0.99 * np.ones_like(rates), rates)

    def test_p2d_spherical_convergence_quadratic(self):
        # test div( r**2 * sin(r) ) == 2/r*sin(r) + cos(r)
        spatial_methods = {"negative particle": pybamm.FiniteVolume()}

        # Function for convergence testing
        def get_error(m):
            # create mesh and discretisation p2d, uniform in x
            mesh = get_p2d_mesh_for_testing(3, m)
            disc = pybamm.Discretisation(mesh, spatial_methods)
            submesh = mesh["negative particle"]
            r = submesh.nodes
            r_edge = pybamm.standard_spatial_vars.r_n_edge

            # Define flux and eqn
            N = pybamm.sin(r_edge)
            div_eqn = pybamm.div(N)
            # Define exact solutions
            # N = sin(r) --> div(N) = 1/r2 * d/dr(r2*N) = 2/r*sin(r) + cos(r)
            div_exact = 2 / r * np.sin(r) + np.cos(r)
            div_exact = np.kron(np.ones(mesh["negative electrode"].npts), div_exact)

            # Discretise and evaluate
            div_eqn_disc = disc.process_symbol(div_eqn)
            div_approx = div_eqn_disc.evaluate()

            return div_approx[:, 0] - div_exact

        # Get errors
        ns = 10 * 2 ** np.arange(1, 7)
        errs = {n: get_error(int(n)) for n in ns}
        # expect quadratic convergence everywhere
        err_norm = np.array([np.linalg.norm(errs[n], np.inf) for n in ns])
        rates = np.log2(err_norm[:-1] / err_norm[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)

    def test_p2d_with_x_dep_bcs_spherical_convergence(self):
        # test div_r( (r**2 * sin(r)) * x ) == (2*sin(r)/r + cos(r)) * x
        spatial_methods = {
            "negative particle": pybamm.FiniteVolume(),
            "negative electrode": pybamm.FiniteVolume(),
        }

        # Function for convergence testing
        def get_error(m):
            # create mesh and discretisation p2d, x-dependent
            mesh = get_p2d_mesh_for_testing(6, m)
            disc = pybamm.Discretisation(mesh, spatial_methods)
            submesh_r = mesh["negative particle"]
            r = submesh_r.nodes
            r_edge = pybamm.standard_spatial_vars.r_n_edge
            x = pybamm.standard_spatial_vars.x_n

            # Define flux and eqn
            N = pybamm.PrimaryBroadcast(x, "negative particle") * pybamm.sin(r_edge)
            div_eqn = pybamm.div(N)
            # Define exact solutions
            # N = r**2*sin(r) --> div(N) = 2*sin(r)/r + cos(r)
            div_exact = 2 / r * np.sin(r) + np.cos(r)
            div_exact = np.kron(mesh["negative electrode"].nodes, div_exact)

            # Discretise and evaluate
            div_eqn_disc = disc.process_symbol(div_eqn)
            div_approx = div_eqn_disc.evaluate()

            return div_approx[:, 0] - div_exact

        # Get errors
        ns = 10 * 2 ** np.arange(1, 7)
        errs = {n: get_error(int(n)) for n in ns}
        # expect quadratic convergence everywhere
        err_norm = np.array([np.linalg.norm(errs[n], np.inf) for n in ns])
        rates = np.log2(err_norm[:-1] / err_norm[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)


def solve_laplace_equation(coord_sys="cartesian"):
    model = pybamm.BaseModel()
    r = pybamm.SpatialVariable("r", domain="domain", coord_sys=coord_sys)
    u = pybamm.Variable("u", domain="domain")
    del_u = pybamm.div(pybamm.grad(u))
    model.boundary_conditions = {
        u: {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(1), "Dirichlet"),
        }
    }
    model.algebraic = {u: del_u}
    model.initial_conditions = {u: pybamm.Scalar(0)}
    model.variables = {"u": u, "r": r}
    geometry = {"domain": {r: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}}}
    submesh_types = {"domain": pybamm.Uniform1DSubMesh}
    var_pts = {r: 500}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
    spatial_methods = {"domain": pybamm.FiniteVolume()}
    disc = pybamm.Discretisation(mesh, spatial_methods)
    disc.process_model(model)
    solver = pybamm.CasadiAlgebraicSolver()
    return solver.solve(model)


class TestFiniteVolumeLaplacian(unittest.TestCase):
    def test_laplacian_cartesian(self):
        solution = solve_laplace_equation(coord_sys="cartesian")
        np.testing.assert_array_almost_equal(
            solution["u"].entries, solution["r"].entries - 1, decimal=10
        )

    def test_laplacian_cylindrical(self):
        solution = solve_laplace_equation(coord_sys="cylindrical polar")
        np.testing.assert_array_almost_equal(
            solution["u"].entries, np.log(solution["r"].entries) / np.log(2), decimal=5
        )

    def test_laplacian_spherical(self):
        solution = solve_laplace_equation(coord_sys="spherical polar")
        np.testing.assert_array_almost_equal(
            solution["u"].entries, 2 - 2 / solution["r"].entries, decimal=5
        )


def solve_advection_equation(direction="upwind", source=1, bc=0):
    model = pybamm.BaseModel()
    x = pybamm.SpatialVariable("x", domain="domain", coord_sys="cartesian")
    u = pybamm.Variable("u", domain="domain")
    if direction == "upwind":
        bc_side = "left"
        y = x
        v = pybamm.PrimaryBroadcastToEdges(1, ["domain"])
        rhs = -pybamm.div(pybamm.upwind(u) * v) + source
    elif direction == "downwind":
        bc_side = "right"
        y = 1 - x
        v = pybamm.PrimaryBroadcastToEdges(-1, ["domain"])
        rhs = -pybamm.div(pybamm.downwind(u) * v) + source

    u_an = (bc + source * y) - (bc + source * (y - pybamm.t)) * ((y - pybamm.t) > 0)
    model.boundary_conditions = {
        u: {
            bc_side: (pybamm.Scalar(bc), "Dirichlet"),
        }
    }
    model.rhs = {u: rhs}
    model.initial_conditions = {u: pybamm.Scalar(0)}
    model.variables = {"u": u, "x": x, "analytical": u_an}
    geometry = {"domain": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}
    submesh_types = {"domain": pybamm.Uniform1DSubMesh}
    var_pts = {x: 1000}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
    spatial_methods = {"domain": pybamm.FiniteVolume()}
    disc = pybamm.Discretisation(mesh, spatial_methods)
    disc.process_model(model)
    solver = pybamm.CasadiSolver()
    return solver.solve(model, [0, 1])


class TestUpwindDownwind(unittest.TestCase):
    def test_upwind(self):
        solution = solve_advection_equation("upwind")
        np.testing.assert_array_almost_equal(
            solution["u"].entries, solution["analytical"].entries, decimal=2
        )

    def test_downwind(self):
        solution = solve_advection_equation("downwind")
        np.testing.assert_array_almost_equal(
            solution["u"].entries, solution["analytical"].entries, decimal=2
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
