#
# Test for the operator class
#

import pybamm

import numpy as np
import unittest


def get_mesh_for_testing(
    xpts=None, rpts=10, ypts=15, zpts=15, geometry=None, cc_submesh=None, order=2
):
    param = pybamm.ParameterValues(
        values={
            "Electrode width [m]": 0.4,
            "Electrode height [m]": 0.5,
            "Negative tab width [m]": 0.1,
            "Negative tab centre y-coordinate [m]": 0.1,
            "Negative tab centre z-coordinate [m]": 0.0,
            "Positive tab width [m]": 0.1,
            "Positive tab centre y-coordinate [m]": 0.3,
            "Positive tab centre z-coordinate [m]": 0.5,
            "Negative electrode thickness [m]": 1 / 3,
            "Separator thickness [m]": 1 / 3,
            "Positive electrode thickness [m]": 1 / 3,
            "Negative particle radius [m]": 0.5,
            "Positive particle radius [m]": 0.5,
        }
    )

    if geometry is None:
        geometry = pybamm.battery_geometry()
    param.process_geometry(geometry)

    submesh_types = {
        "negative electrode": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh, {"order": order}
        ),
        "separator": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh, {"order": order}
        ),
        "positive electrode": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh, {"order": order}
        ),
        "negative particle": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh, {"order": order}
        ),
        "positive particle": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh, {"order": order}
        ),
        "current collector": pybamm.SubMesh0D,
    }
    if cc_submesh:
        submesh_types["current collector"] = cc_submesh

    if xpts is None:
        xn_pts, xs_pts, xp_pts = 40, 25, 35
    else:
        xn_pts, xs_pts, xp_pts = xpts, xpts, xpts
    var_pts = {
        "x_n": xn_pts,
        "x_s": xs_pts,
        "x_p": xp_pts,
        "r_n": rpts,
        "r_p": rpts,
        "y": ypts,
        "z": zpts,
    }

    return pybamm.Mesh(geometry, submesh_types, var_pts)


def get_p2d_mesh_for_testing(xpts=None, rpts=10):
    geometry = pybamm.battery_geometry()
    return get_mesh_for_testing(xpts=xpts, rpts=rpts, geometry=geometry)


class TestSpectralVolumeConvergence(unittest.TestCase):
    def test_grad_div_broadcast(self):
        # create mesh and discretisation
        spatial_methods = {"macroscale": pybamm.SpectralVolume()}
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
        order = 2
        spatial_methods = {"macroscale": pybamm.SpectralVolume(order=order)}
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
            mesh = get_mesh_for_testing(n, order=order)
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
        ns = 100 * 2 ** np.arange(5)
        errs = {n: get_error(int(n)) for n in ns}
        # expect linear convergence at internal points
        # (the higher-order convergence is in the integral means,
        #  not in the edge values)
        errs_internal = np.array([np.linalg.norm(errs[n][1:-1], np.inf) for n in ns])
        rates = np.log2(errs_internal[:-1] / errs_internal[1:])
        np.testing.assert_array_less(0.99 * np.ones_like(rates), rates)
        # expect linear convergence at the boundaries
        for idx in [0, -1]:
            err_boundary = np.array([errs[n][idx] for n in ns])
            rates = np.log2(err_boundary[:-1] / err_boundary[1:])
            np.testing.assert_array_less(0.98 * np.ones_like(rates), rates)

    def test_cartesian_div_convergence(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        spatial_methods = {"macroscale": pybamm.SpectralVolume()}

        # Function for convergence testing
        def get_error(n):
            # create mesh and discretisation
            mesh = get_mesh_for_testing(n)
            disc = pybamm.Discretisation(mesh, spatial_methods)
            submesh = mesh[whole_cell]
            x = submesh.nodes
            x_edge = pybamm.standard_spatial_vars.x_edge

            # Define flux and bcs
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
        ns = 10 * 2 ** np.arange(6)
        errs = {n: get_error(int(n)) for n in ns}
        # expect quadratic convergence everywhere
        err_norm = np.array([np.linalg.norm(errs[n], np.inf) for n in ns])
        rates = np.log2(err_norm[:-1] / err_norm[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)

    def test_spherical_div_convergence_quadratic(self):
        # test div( r**2 * sin(r) ) == 2/r*sin(r) + cos(r)
        spatial_methods = {"negative particle": pybamm.SpectralVolume()}

        # Function for convergence testing
        def get_error(n):
            # create mesh and discretisation (single particle)
            mesh = get_mesh_for_testing(rpts=n)
            disc = pybamm.Discretisation(mesh, spatial_methods)
            submesh = mesh["negative particle"]
            r = submesh.nodes
            r_edge = pybamm.SpatialVariableEdge("r_n", domain=["negative particle"])

            # Define flux and bcs
            N = pybamm.sin(r_edge)
            div_eqn = pybamm.div(N)
            # Define exact solutions
            # N = r**3 --> div(N) = 5 * r**2
            div_exact = 2 / r * np.sin(r) + np.cos(r)

            # Discretise and evaluate
            div_eqn_disc = disc.process_symbol(div_eqn)
            div_approx = div_eqn_disc.evaluate()

            # Return difference between approx and exact
            return div_approx[:, 0] - div_exact

        # Get errors
        ns = 10 * 2 ** np.arange(6)
        errs = {n: get_error(int(n)) for n in ns}
        # expect quadratic convergence everywhere
        err_norm = np.array([np.linalg.norm(errs[n], np.inf) for n in ns])
        rates = np.log2(err_norm[:-1] / err_norm[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)

    def test_spherical_div_convergence_linear(self):
        # test div( r*sin(r) ) == 3*sin(r) + r*cos(r)
        spatial_methods = {"negative particle": pybamm.SpectralVolume()}

        # Function for convergence testing
        def get_error(n):
            # create mesh and discretisation (single particle)
            mesh = get_mesh_for_testing(rpts=n)
            disc = pybamm.Discretisation(mesh, spatial_methods)
            submesh = mesh["negative particle"]
            r = submesh.nodes
            r_edge = pybamm.SpatialVariableEdge("r_n", domain=["negative particle"])

            # Define flux and bcs
            N = r_edge * pybamm.sin(r_edge)
            div_eqn = pybamm.div(N)
            # Define exact solutions
            # N = r*sin(r) --> div(N) = 3*sin(r) + r*cos(r)
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
        spatial_methods = {"negative particle": pybamm.SpectralVolume()}

        # Function for convergence testing
        def get_error(m):
            # create mesh and discretisation p2d, uniform in x
            mesh = get_p2d_mesh_for_testing(3, m)
            disc = pybamm.Discretisation(mesh, spatial_methods)
            submesh = mesh["negative particle"]
            r = submesh.nodes
            r_edge = pybamm.standard_spatial_vars.r_n_edge

            N = pybamm.sin(r_edge)
            div_eqn = pybamm.div(N)
            # Define exact solutions
            # N = r**2*sin(r) --> div(N) = 2/r*sin(r) + cos(r)
            div_exact = 2 / r * np.sin(r) + np.cos(r)
            div_exact = np.kron(np.ones(mesh["negative electrode"].npts), div_exact)

            # Discretise and evaluate
            div_eqn_disc = disc.process_symbol(div_eqn)
            div_approx = div_eqn_disc.evaluate()

            return div_approx[:, 0] - div_exact

        # Get errors
        ns = 10 * 2 ** np.arange(6)
        errs = {n: get_error(int(n)) for n in ns}
        # expect quadratic convergence everywhere
        err_norm = np.array([np.linalg.norm(errs[n], np.inf) for n in ns])
        rates = np.log2(err_norm[:-1] / err_norm[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)

    def test_p2d_with_x_dep_bcs_spherical_convergence(self):
        # test div_r( sin(r) * x ) == (2/r*sin(r) + cos(r)) * x
        spatial_methods = {
            "negative particle": pybamm.SpectralVolume(),
            "negative electrode": pybamm.SpectralVolume(),
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

            N = pybamm.PrimaryBroadcast(x, "negative particle") * (pybamm.sin(r_edge))
            div_eqn = pybamm.div(N)
            # Define exact solutions
            # N = sin(r) --> div(N) = 2/r*sin(r) + cos(r)
            div_exact = 2 / r * np.sin(r) + np.cos(r)
            div_exact = np.kron(mesh["negative electrode"].nodes, div_exact)

            # Discretise and evaluate
            div_eqn_disc = disc.process_symbol(div_eqn)
            div_approx = div_eqn_disc.evaluate()

            return div_approx[:, 0] - div_exact

        # Get errors
        ns = 10 * 2 ** np.arange(6)
        errs = {n: get_error(int(n)) for n in ns}
        # expect quadratic convergence everywhere
        err_norm = np.array([np.linalg.norm(errs[n], np.inf) for n in ns])
        rates = np.log2(err_norm[:-1] / err_norm[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
