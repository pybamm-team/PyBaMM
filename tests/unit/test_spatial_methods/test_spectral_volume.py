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
            "Negative electrode thickness [m]": 0.3,
            "Separator thickness [m]": 0.4,
            "Positive electrode thickness [m]": 0.3,
            "Negative particle radius [m]": 2,
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


def get_1p1d_mesh_for_testing(
    xpts=None,
    rpts=10,
    zpts=15,
    cc_submesh=pybamm.Uniform1DSubMesh,
):
    geometry = pybamm.battery_geometry(options={"dimensionality": 1})
    return get_mesh_for_testing(
        xpts=xpts, rpts=rpts, zpts=zpts, geometry=geometry, cc_submesh=cc_submesh
    )


class TestSpectralVolume(unittest.TestCase):
    def test_exceptions(self):
        sp_meth = pybamm.SpectralVolume()
        with self.assertRaises(ValueError):
            sp_meth.chebyshev_differentiation_matrices(3, 3)

        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.SpectralVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        disc.set_variable_slices([var])
        discretised_symbol = pybamm.StateVector(*disc.y_slices[var])
        sp_meth.build(mesh)

        bcs = {"left": (pybamm.Scalar(0), "x"), "right": (pybamm.Scalar(3), "Neumann")}
        with self.assertRaisesRegex(ValueError, "boundary condition must be"):
            sp_meth.replace_dirichlet_values(var, discretised_symbol, bcs)
        with self.assertRaisesRegex(ValueError, "boundary condition must be"):
            sp_meth.replace_neumann_values(var, discretised_symbol, bcs)
        bcs = {"left": (pybamm.Scalar(0), "Neumann"), "right": (pybamm.Scalar(3), "x")}
        with self.assertRaisesRegex(ValueError, "boundary condition must be"):
            sp_meth.replace_dirichlet_values(var, discretised_symbol, bcs)
        with self.assertRaisesRegex(ValueError, "boundary condition must be"):
            sp_meth.replace_neumann_values(var, discretised_symbol, bcs)

    def test_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions and also test the
        case where only one Spectral Volume is discretised
        """
        # Create discretisation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        mesh = get_mesh_for_testing(1)
        spatial_methods = {"macroscale": pybamm.SpectralVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]

        # Test gradient of constant is zero
        # grad(1) = 0
        constant_y = np.ones_like(submesh.nodes[:, np.newaxis])
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(submesh.edges[:, np.newaxis]),
        )

        # Test operations on linear x
        linear_y = submesh.nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        # grad(x) = 1
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(submesh.edges[:, np.newaxis]),
        )
        # div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(submesh.nodes[:, np.newaxis]),
        )

    def test_spherical_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions in spherical polar
        coordinates
        """
        # Create discretisation
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.SpectralVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh["negative particle"]
        npts = submesh.npts
        sec_npts = mesh["negative electrode"].npts * mesh["current collector"].npts
        total_npts = npts * sec_npts
        total_npts_edges = (npts + 1) * sec_npts

        # Test gradient
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={
                "secondary": "negative electrode",
                "tertiary": "current collector",
            },
        )
        grad_eqn = pybamm.grad(var)
        # grad(1) = 0
        constant_y = np.ones((total_npts, 1))
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, constant_y), np.zeros((total_npts_edges, 1))
        )
        # grad(r) == 1
        y_linear = np.tile(
            submesh.nodes,
            mesh["negative electrode"].npts * mesh["current collector"].npts,
        )
        # bcs: r = 0, r = 2
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(2), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, y_linear), np.ones((total_npts_edges, 1))
        )

        # Test divergence of gradient
        # div (grad r^2) = 6
        y_squared = np.tile(
            submesh.nodes**2,
            mesh["negative electrode"].npts * mesh["current collector"].npts,
        )
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eval = div_eqn_disc.evaluate(None, y_squared)
        div_eval = np.reshape(div_eval, [sec_npts, npts])
        np.testing.assert_array_almost_equal(
            div_eval[:, 2:-2], 6 * np.ones([sec_npts, npts - 4])
        )

    def test_p2d_spherical_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions in the pseudo
        2-dimensional case
        """
        # Create discretisation
        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.SpectralVolume(),
            "negative particle": pybamm.SpectralVolume(),
            "positive particle": pybamm.SpectralVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        prim_pts = mesh["negative particle"].npts
        sec_pts = mesh["negative electrode"].npts

        # Test gradient of a constant is zero
        # grad(1) = 0
        constant_y = np.kron(np.ones(sec_pts), np.ones(prim_pts))
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        grad_eval = grad_eqn_disc.evaluate(None, constant_y)
        grad_eval = np.reshape(grad_eval, [sec_pts, prim_pts + 1])
        np.testing.assert_array_almost_equal(
            grad_eval, np.zeros([sec_pts, prim_pts + 1])
        )

        # Test divergence of gradient
        # div(grad r^2) = 6
        y_squared = np.tile(mesh["negative particle"].nodes ** 2, sec_pts)
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eval = div_eqn_disc.evaluate(None, y_squared)
        div_eval = np.reshape(div_eval, [sec_pts, prim_pts])
        np.testing.assert_array_almost_equal(
            div_eval[:, 2:-2], 6 * np.ones([sec_pts, prim_pts - 4])
        )

    def test_grad_div_shapes_Neumann_bcs(self):
        """
        Test grad and div with Neumann boundary conditions in Cartesian coordinates
        """
        # Create discretisation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.SpectralVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]

        # Test gradient of constant is zero
        # grad(1) = 0
        constant_y = np.ones_like(submesh.nodes[:, np.newaxis])
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(submesh.edges[:, np.newaxis]),
        )

        # Test operations on linear x
        linear_y = submesh.nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        # grad(x) = 1
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(submesh.edges[:, np.newaxis]),
        )
        # div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(submesh.nodes[:, np.newaxis]),
        )

    def test_grad_div_shapes_Dirichlet_and_Neumann_bcs(self):
        """
        Test grad and div with a Dirichlet boundary condition on one side and
        a Neumann boundary conditions on the other side in Cartesian coordinates
        """
        # Create discretisation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.SpectralVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]

        # Test gradient and divergence of a constant
        constant_y = np.ones_like(submesh.nodes[:, np.newaxis])
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        # grad(1) = 0
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(submesh.edges[:, np.newaxis]),
        )
        # div(grad(1)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(submesh.nodes[:, np.newaxis]),
        )

        # Test gradient and divergence of linear x
        linear_y = submesh.nodes
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        # grad(x) = 1
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(submesh.edges[:, np.newaxis]),
        )
        # div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(submesh.nodes[:, np.newaxis]),
        )

    def test_spherical_grad_div_shapes_Neumann_bcs(self):
        """
        Test grad and div with Neumann boundary conditions spherical polar
        coordinates
        """
        # Create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.SpectralVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh["negative particle"]

        # Test gradient
        var = pybamm.Variable("var", domain="negative particle")
        grad_eqn = pybamm.grad(var)
        # grad(1) = 0
        constant_y = np.ones_like(submesh.nodes[:, np.newaxis])
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(submesh.edges[:, np.newaxis]),
        )
        # grad(r) == 1
        linear_y = submesh.nodes
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(submesh.edges[:, np.newaxis]),
        )

        # Test divergence of gradient
        # div(grad(r^2)) = 6 , N_left = 2*r[0], N_right = 2 * r[-1]
        quadratic_y = submesh.nodes**2
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(2 * submesh.edges[-1]), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, quadratic_y),
            6 * np.ones((submesh.npts, 1)),
        )

    def test_p2d_spherical_grad_div_shapes_Neumann_bcs(self):
        """
        Test grad and div with Neumann boundary conditions in the pseudo
        2-dimensional case
        """
        # Create discretisation
        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.SpectralVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        prim_pts = mesh["negative particle"].npts
        sec_pts = mesh["negative electrode"].npts

        # Test gradient of a constant is zero
        # grad(1) = 0
        constant_y = np.kron(np.ones(sec_pts), np.ones(prim_pts))
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        grad_eval = grad_eqn_disc.evaluate(None, constant_y)
        grad_eval = np.reshape(grad_eval, [sec_pts, prim_pts + 1])
        np.testing.assert_array_equal(grad_eval, np.zeros([sec_pts, prim_pts + 1]))

        # Test divergence of gradient
        # div(grad r^2) = 6, N_left = 2 * r[0], N_right = 2 * r[-1]
        submesh = mesh["negative particle"]
        y_squared = np.tile(submesh.nodes**2, sec_pts)
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(2 * submesh.edges[-1]), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eval = div_eqn_disc.evaluate(None, y_squared)
        div_eval = np.reshape(div_eval, [sec_pts, prim_pts])
        np.testing.assert_array_almost_equal(div_eval, 6 * np.ones([sec_pts, prim_pts]))

    def test_grad_div_shapes_mixed_domain(self):
        # Create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.SpectralVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[("negative electrode", "separator")]

        # Test gradient of constant
        # grad(1) = 0
        constant_y = np.ones_like(submesh.nodes[:, np.newaxis])
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(submesh.edges[:, np.newaxis]),
        )

        # Test operations on linear x
        linear_y = submesh.nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(submesh.edges[-1]), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        # grad(x) = 1
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(submesh.edges[:, np.newaxis]),
        )
        # div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(submesh.nodes[:, np.newaxis]),
        )

    def test_grad_1plus1d(self):
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.SpectralVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        a = pybamm.Variable(
            "a",
            domain=["negative electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        b = pybamm.Variable(
            "b",
            domain=["separator"],
            auxiliary_domains={"secondary": "current collector"},
        )
        c = pybamm.Variable(
            "c",
            domain=["positive electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        var = pybamm.concatenation(a, b, c)
        boundary_conditions = {
            var: {
                "left": (pybamm.Vector(np.linspace(0, 1, 15)), "Neumann"),
                "right": (pybamm.Vector(np.linspace(0, 1, 15)), "Neumann"),
            }
        }

        # Discretise
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(pybamm.grad(var))

        # Evaulate
        submesh = mesh[var.domain]
        linear_y = np.outer(np.linspace(0, 1, 15), submesh.nodes).reshape(-1, 1)
        expected = np.outer(np.linspace(0, 1, 15), np.ones_like(submesh.edges)).reshape(
            -1, 1
        )
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y), expected
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
