#
# Test for the operator class
#
import pybamm

import numpy as np
import unittest


def get_mesh_for_testing(
    xpts=None, rpts=10, ypts=15, zpts=15, geometry=None, cc_submesh=None,
    order=2
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
            "Separator thickness [m]": 0.3,
            "Positive electrode thickness [m]": 0.3,
        }
    )

    if geometry is None:
        geometry = pybamm.battery_geometry()
    param.process_geometry(geometry)

    submesh_types = {
        "negative electrode": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order}
        ),
        "separator": pybamm.MeshGenerator(pybamm.SpectralVolume1DSubMesh,
                                          {"order": order}),
        "positive electrode": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order}
        ),
        "negative particle": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order}
        ),
        "positive particle": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order}
        ),
        "current collector": pybamm.MeshGenerator(pybamm.SubMesh0D),
    }
    if cc_submesh:
        submesh_types["current collector"] = cc_submesh

    if xpts is None:
        xn_pts, xs_pts, xp_pts = 40, 25, 35
    else:
        xn_pts, xs_pts, xp_pts = xpts, xpts, xpts
    var = pybamm.standard_spatial_vars
    var_pts = {
        var.x_n: xn_pts,
        var.x_s: xs_pts,
        var.x_p: xp_pts,
        var.r_n: rpts,
        var.r_p: rpts,
        var.y: ypts,
        var.z: zpts,
    }

    return pybamm.Mesh(geometry, submesh_types, var_pts)


def get_p2d_mesh_for_testing(xpts=None, rpts=10):
    geometry = pybamm.battery_geometry()
    return get_mesh_for_testing(xpts=xpts, rpts=rpts, geometry=geometry)


def get_1p1d_mesh_for_testing(
    xpts=None,
    rpts=10,
    zpts=15,
    cc_submesh=pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
):
    geometry = pybamm.battery_geometry(current_collector_dimension=1)
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
        discretised_symbol = pybamm.StateVector(*disc.y_slices[var.id])
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
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        and also test the case where only one Spectral Volume is discretised
        """
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        # create discretisation
        mesh = get_mesh_for_testing(1)
        spatial_methods = {"macroscale": pybamm.SpectralVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        # grad
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        constant_y = np.ones_like(combined_submesh.nodes[:, np.newaxis])
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh.edges[:, np.newaxis]),
        )

        # div: test on linear y (should have laplacian zero) so change bcs
        linear_y = combined_submesh.nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }

        disc.bcs = boundary_conditions

        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh.edges[:, np.newaxis]),
        )

        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(combined_submesh.nodes[:, np.newaxis]),
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
        var = pybamm.Concatenation(a, b, c)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Vector(np.linspace(0, 1, 15)), "Neumann"),
                "right": (pybamm.Vector(np.linspace(0, 1, 15)), "Neumann"),
            }
        }

        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(pybamm.grad(var))

        # Evaulate
        combined_submesh = mesh.combine_submeshes(*var.domain)
        linear_y = np.outer(np.linspace(0, 1, 15), combined_submesh.nodes).reshape(
            -1, 1
        )

        expected = np.outer(
            np.linspace(0, 1, 15), np.ones_like(combined_submesh.edges)
        ).reshape(-1, 1)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y), expected
        )

    def test_spherical_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        """
        # create discretisation
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.SpectralVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        submesh = mesh["negative particle"]

        # grad
        # grad(r) == 1
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={
                "secondary": "negative electrode",
                "tertiary": "current collector",
            },
        )
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }

        disc.bcs = boundary_conditions

        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        total_npts = (
            submesh.npts
            * mesh["negative electrode"].npts
            * mesh["current collector"].npts
        )
        total_npts_edges = (
            (submesh.npts + 1)
            * mesh["negative electrode"].npts
            * mesh["current collector"].npts
        )
        constant_y = np.ones((total_npts, 1))
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y), np.zeros((total_npts_edges, 1))
        )

        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        y_linear = np.tile(
            submesh.nodes,
            mesh["negative electrode"].npts * mesh["current collector"].npts,
        )
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, y_linear), np.ones((total_npts_edges, 1))
        )

        # div: test on linear r^2
        # div (grad r^2) = 6
        const = 6 * np.ones((total_npts, 1))
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(6), "Dirichlet"),
                "right": (pybamm.Scalar(6), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, const),
            np.zeros(
                (
                    submesh.npts
                    * mesh["negative electrode"].npts
                    * mesh["current collector"].npts,
                    1,
                )
            ),
        )

    def test_p2d_spherical_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        in the pseudo 2-dimensional case
        """

        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.SpectralVolume(),
            "negative particle": pybamm.SpectralVolume(),
            "positive particle": pybamm.SpectralVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        n_mesh = mesh["negative particle"]

        mesh.add_ghost_meshes()
        disc.mesh.add_ghost_meshes()

        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        prim_pts = n_mesh.npts
        sec_pts = mesh["negative electrode"].npts
        constant_y = np.kron(np.ones(sec_pts), np.ones(prim_pts))

        grad_eval = grad_eqn_disc.evaluate(None, constant_y)
        grad_eval = np.reshape(grad_eval, [sec_pts, prim_pts + 1])

        np.testing.assert_array_equal(grad_eval, np.zeros([sec_pts, prim_pts + 1]))

        # div
        # div (grad r^2) = 6, N_left = N_right = 0
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        bc_var = disc.process_symbol(
            pybamm.SpatialVariable("x_n", domain="negative electrode")
        )
        boundary_conditions = {
            var.id: {"left": (bc_var, "Neumann"), "right": (bc_var, "Neumann")}
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)

        const = 6 * np.ones(sec_pts * prim_pts)
        div_eval = div_eqn_disc.evaluate(None, const)
        div_eval = np.reshape(div_eval, [sec_pts, prim_pts])
        np.testing.assert_array_almost_equal(
            div_eval[:, :-1], np.zeros([sec_pts, prim_pts - 1])
        )

    def test_grad_div_shapes_Neumann_bcs(self):
        """Test grad and div with Neumann boundary conditions (applied by div on N)"""
        whole_cell = ["negative electrode", "separator", "positive electrode"]

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.SpectralVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        # grad
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        constant_y = np.ones_like(combined_submesh.nodes[:, np.newaxis])
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh.edges[:][:, np.newaxis]),
        )

        # div
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)

        # Linear y should have laplacian zero
        linear_y = combined_submesh.nodes
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh.edges[:][:, np.newaxis]),
        )
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(combined_submesh.nodes[:, np.newaxis]),
        )

    def test_grad_div_shapes_Dirichlet_and_Neumann_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on c) on
        one side and Neumann boundary conditions (applied by div on N) on the other
        """
        whole_cell = ["negative electrode", "separator", "positive electrode"]

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.SpectralVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        # grad
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        disc.set_variable_slices([var])

        # div
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        div_eqn_disc = disc.process_symbol(div_eqn)

        # Constant y should have gradient and laplacian zero
        constant_y = np.ones_like(combined_submesh.nodes[:, np.newaxis])
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh.edges[:, np.newaxis]),
        )
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh.nodes[:, np.newaxis]),
        )

        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        div_eqn_disc = disc.process_symbol(div_eqn)

        # Linear y should have gradient one and laplacian zero
        linear_y = combined_submesh.nodes
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh.edges[:, np.newaxis]),
        )
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(combined_submesh.nodes[:, np.newaxis]),
        )

    def test_spherical_grad_div_shapes_Neumann_bcs(self):
        """Test grad and div with Neumann boundary conditions (applied by div on N)"""

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.SpectralVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes("negative particle")

        # grad
        var = pybamm.Variable("var", domain="negative particle")
        grad_eqn = pybamm.grad(var)
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        constant_y = np.ones_like(combined_submesh.nodes[:, np.newaxis])
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh.edges[:][:, np.newaxis]),
        )

        linear_y = combined_submesh.nodes
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh.edges[:][:, np.newaxis]),
        )
        # div
        # div ( grad(r^2) ) == 6 , N_left = N_right = 0
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)

        linear_y = combined_submesh.nodes
        const = 6 * np.ones(combined_submesh.npts)

        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, const), np.zeros((combined_submesh.npts, 1))
        )

    def test_p2d_spherical_grad_div_shapes_Neumann_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        in the pseudo 2-dimensional case
        """

        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.SpectralVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        n_mesh = mesh["negative particle"]

        mesh.add_ghost_meshes()
        disc.mesh.add_ghost_meshes()

        # test grad
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )
        grad_eqn = pybamm.grad(var)
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        prim_pts = n_mesh.npts
        sec_pts = mesh["negative electrode"].npts
        constant_y = np.kron(np.ones(sec_pts), np.ones(prim_pts))

        grad_eval = grad_eqn_disc.evaluate(None, constant_y)
        grad_eval = np.reshape(grad_eval, [sec_pts, prim_pts + 1])

        np.testing.assert_array_equal(grad_eval, np.zeros([sec_pts, prim_pts + 1]))

        # div
        # div (grad r^2) = 6, N_left = N_right = 0
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)

        const = 6 * np.ones(sec_pts * prim_pts)
        div_eval = div_eqn_disc.evaluate(None, const)
        div_eval = np.reshape(div_eval, [sec_pts, prim_pts])
        np.testing.assert_array_almost_equal(div_eval, np.zeros([sec_pts, prim_pts]))

    def test_grad_div_shapes_mixed_domain(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        """
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.SpectralVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # grad
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        disc.set_variable_slices([var])

        grad_eqn_disc = disc.process_symbol(grad_eqn)

        combined_submesh = mesh.combine_submeshes("negative electrode", "separator")
        constant_y = np.ones_like(combined_submesh.nodes[:, np.newaxis])
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh.edges[:, np.newaxis]),
        )

        # div: test on linear y (should have laplacian zero) so change bcs
        linear_y = combined_submesh.nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(combined_submesh.edges[-1]), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh.edges[:, np.newaxis]),
        )

        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(combined_submesh.nodes[:, np.newaxis]),
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
