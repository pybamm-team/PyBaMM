import pytest
import numpy as np
import pybamm
from tests import get_mesh_for_testing_3d


class TestFiniteVolume3D:
    def test_node_to_edge_to_node_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        fin_vol = pybamm.FiniteVolume3D()
        fin_vol.build(mesh)

        submesh = mesh["negative electrode"]
        n_x, n_y, n_z = submesh.npts_x, submesh.npts_y, submesh.npts_z
        n_nodes = n_x * n_y * n_z

        y_test = np.ones(n_nodes) * 2
        c = pybamm.StateVector(slice(0, n_nodes), domain=["negative electrode"])

        for method in ["arithmetic", "harmonic"]:
            # Test X direction
            edge_x = fin_vol.node_to_edge(c, method=method, direction="x")
            node_x = fin_vol.edge_to_node(edge_x, method=method, direction="x")
            # Extrapolation means we won't get the exact same values back
            np.testing.assert_allclose(
                node_x.evaluate(None, y_test), y_test[:, np.newaxis], rtol=1e-1
            )

            # Test Y direction
            edge_y = fin_vol.node_to_edge(c, method=method, direction="y")
            node_y = fin_vol.edge_to_node(edge_y, method=method, direction="y")
            np.testing.assert_allclose(
                node_y.evaluate(None, y_test), y_test[:, np.newaxis], rtol=1e-1
            )

            # Test Z direction
            edge_z = fin_vol.node_to_edge(c, method=method, direction="z")
            node_z = fin_vol.edge_to_node(edge_z, method=method, direction="z")
            np.testing.assert_allclose(
                node_z.evaluate(None, y_test), y_test[:, np.newaxis], rtol=1e-1
            )

        with pytest.raises(ValueError, match="shift key"):
            fin_vol.shift(c, "bad shift key", "arithmetic", "x")
        with pytest.raises(ValueError, match="method"):
            fin_vol.shift(c, "node to edge", "bad method", "x")
        with pytest.raises(ValueError, match="direction"):
            fin_vol.shift(c, "node to edge", "arithmetic", "bad direction")

    def test_concatenation_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        fin_vol = pybamm.FiniteVolume3D()
        fin_vol.build(mesh)

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        edges = [pybamm.Vector(mesh[dom].edges_x, domain=dom) for dom in whole_cell]
        v_disc = fin_vol.concatenation(edges)
        np.testing.assert_array_equal(
            v_disc.evaluate()[:, 0],
            mesh[whole_cell].nodes,
        )

        bad_edges = [
            pybamm.Vector(np.ones(mesh[dom].npts + 2), domain=dom) for dom in whole_cell
        ]
        with pytest.raises(pybamm.ShapeError, match="child must have size n_nodes"):
            fin_vol.concatenation(bad_edges)

    def test_discretise_diffusivity_times_spatial_operator_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        submesh = mesh[whole_cell]

        var = pybamm.Variable("var", domain=whole_cell)
        disc.set_variable_slices([var])

        y_test = np.ones_like(submesh.nodes[:, np.newaxis])

        exprs = [
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
        ]

        bc_combinations = [
            {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            },
            {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
            },
            {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Neumann"),
            },
            {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            },
        ]

        for eqn in exprs:
            for bc in bc_combinations:
                disc.bcs = {var: bc}
                eqn_disc = disc.process_symbol(eqn)
                eqn_disc.evaluate(None, y_test)

    def test_discretise_spatial_variable_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume3D(),
            "negative particle": pybamm.FiniteVolume3D(),
            "positive particle": pybamm.FiniteVolume3D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        x1 = pybamm.SpatialVariable("x", ["negative electrode"])
        x1_disc = disc.process_symbol(x1)
        assert isinstance(x1_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            x1_disc.evaluate(), disc.mesh["negative electrode"].nodes[:, np.newaxis]
        )

        x2 = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        x2_disc = disc.process_symbol(x2)
        assert isinstance(x2_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            x2_disc.evaluate(),
            disc.mesh[("negative electrode", "separator")].nodes[:, np.newaxis],
        )

        r = 3 * pybamm.SpatialVariable("r", ["negative particle"])
        r_disc = disc.process_symbol(r)
        assert isinstance(r_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            r_disc.evaluate(),
            3 * disc.mesh["negative particle"].nodes[:, np.newaxis],
        )

    def test_mass_matrix_shape_3d(self):
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

        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]
        disc.process_model(model)

        mass = np.eye(submesh.npts)
        np.testing.assert_array_equal(mass, model.mass_matrix.entries.toarray())

    def test_jacobian_3d(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]
        spatial_method = pybamm.FiniteVolume3D()
        spatial_method.build(mesh)

        # Setup a concatenated variable var on the whole cell
        var = pybamm.Variable("var", domain=whole_cell)
        disc.set_variable_slices([var])
        y = pybamm.StateVector(slice(0, submesh.npts), domain=whole_cell)
        y_test = np.ones_like(submesh.nodes[:, np.newaxis])

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

        eqn = var * pybamm.grad(var)
        eqn_disc = disc.process_symbol(eqn)
        eqn_jac = eqn_disc.jac(y)

        eqn_jac.evaluate(y=y_test)

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

    def test_delta_function_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var = pybamm.Variable("var")
        delta_fn_left = pybamm.DeltaFunction(var, "left", "negative electrode")
        delta_fn_right = pybamm.DeltaFunction(var, "right", "negative electrode")
        disc.set_variable_slices([var])
        delta_fn_left_disc = disc.process_symbol(delta_fn_left)
        delta_fn_right_disc = disc.process_symbol(delta_fn_right)

        y = np.ones_like(mesh["negative electrode"].nodes[:, np.newaxis])

        assert delta_fn_left_disc.domains == delta_fn_left.domains
        assert isinstance(delta_fn_left_disc, pybamm.Multiplication)
        assert isinstance(delta_fn_left_disc.left, pybamm.Matrix)

        np.testing.assert_array_equal(delta_fn_left_disc.left.evaluate()[:, 1:], 0)
        assert delta_fn_left_disc.shape == y.shape

        assert delta_fn_right_disc.domains == delta_fn_right.domains
        assert isinstance(delta_fn_right_disc, pybamm.Multiplication)
        assert isinstance(delta_fn_right_disc.left, pybamm.Matrix)

        np.testing.assert_array_equal(delta_fn_right_disc.left.evaluate()[:, :-1], 0)
        assert delta_fn_right_disc.shape == y.shape

        var_disc = disc.process_symbol(var)
        x = pybamm.standard_spatial_vars.x_n  # stands for negative-electrode nodes
        delta_fn_int_disc = disc.process_symbol(pybamm.Integral(delta_fn_left, x))
        np.testing.assert_allclose(
            var_disc.evaluate(y=y) * mesh["negative electrode"].edges[-1],
            np.sum(delta_fn_int_disc.evaluate(y=y)),
        )

    def test_heaviside_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var = pybamm.Variable("var", domain="negative electrode")
        heav = var > 1

        disc.set_variable_slices([var])
        disc_heav = disc.process_symbol(heav * var)
        nodes = mesh["negative electrode"].nodes
        assert disc_heav.size == nodes.size

        np.testing.assert_array_equal(
            disc_heav.evaluate(y=2 * np.ones_like(nodes)),
            2 * np.ones((nodes.size, 1)),
        )
        np.testing.assert_array_equal(
            disc_heav.evaluate(y=-2 * np.ones_like(nodes)),
            np.zeros((nodes.size, 1)),
        )

    def test_upwind_downwind_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
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
        assert disc_upwind.size == nodes.size + 1
        assert disc_downwind.size == nodes.size + 1

        y_test = 2 * np.ones_like(nodes)
        np.testing.assert_array_equal(
            disc_upwind.evaluate(y=y_test),
            np.concatenate([np.array([8]), 2 * np.ones(n)])[:, np.newaxis],
        )
        np.testing.assert_array_equal(
            disc_downwind.evaluate(y=y_test),
            np.concatenate([2 * np.ones(n), np.array([4])])[:, np.newaxis],
        )

        disc.bcs = {}
        disc._discretised_symbols = {}
        with pytest.raises(pybamm.ModelError, match="Boundary conditions"):
            disc.process_symbol(upwind)

        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(5), "Neumann"),
                "right": (pybamm.Scalar(3), "Neumann"),
            }
        }
        with pytest.raises(pybamm.ModelError, match="Dirichlet boundary conditions"):
            disc.process_symbol(upwind)
        with pytest.raises(pybamm.ModelError, match="Dirichlet boundary conditions"):
            disc.process_symbol(downwind)

    def test_full_broadcast_domains_3d(self):
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

        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume3D(),
            "separator": pybamm.FiniteVolume3D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

    def test_inner_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume3D(),
            "negative particle": pybamm.FiniteVolume3D(),
        }

        var = pybamm.Variable("var", domain="negative particle")
        grad_var = pybamm.grad(var)
        inner = pybamm.inner(grad_var, grad_var)

        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.set_variable_slices([var])
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        inner_disc = disc.process_symbol(inner)

        assert isinstance(inner_disc, pybamm.Inner)
        assert isinstance(inner_disc.left, pybamm.MatrixMultiplication)
        assert isinstance(inner_disc.right, pybamm.MatrixMultiplication)

        n = mesh["negative particle"].npts
        y = np.ones(n)[:, np.newaxis]
        np.testing.assert_array_equal(inner_disc.evaluate(y=y), np.zeros((n, 1)))

        grad_var = pybamm.grad(pybamm.SecondaryBroadcast(var, "negative electrode"))
        inner = pybamm.inner(grad_var, grad_var)

        inner_disc = disc.process_symbol(inner)
        assert isinstance(inner_disc, pybamm.Inner)
        assert isinstance(inner_disc.left, pybamm.MatrixMultiplication)
        assert isinstance(inner_disc.right, pybamm.MatrixMultiplication)

        m = mesh["negative electrode"].npts
        np.testing.assert_array_equal(inner_disc.evaluate(y=y), np.zeros((n * m, 1)))

    def test_discrete_laplacian_linear_in_x_is_zero(self):
        nx, ny, nz = 6, 5, 4
        mesh = get_mesh_for_testing_3d(xpts=nx, ypts=ny, zpts=nz)

        spatial_method = pybamm.FiniteVolume3D()
        spatial_method.build(mesh)

        submesh = mesh["negative electrode"]
        n = submesh.npts

        x_nodes = submesh.nodes  # shape = (n,)

        u_exact = x_nodes[:, np.newaxis]

        var = pybamm.Variable("var", domain=["negative electrode"])
        lap_symbolic = pybamm.div(pybamm.grad(var))

        disc = pybamm.Discretisation(mesh, {"negative electrode": spatial_method})
        disc.set_variable_slices([var])

        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }

        lap_disc = disc.process_symbol(lap_symbolic)

        lap_u = lap_disc.evaluate(y=u_exact)

        np.testing.assert_allclose(lap_u, np.zeros((n, 1)), atol=0, rtol=0)

    def test_discrete_laplacian_quadratic_in_x_converges_to_two(self):
        nx = ny = nz = 20
        mesh = get_mesh_for_testing_3d(xpts=nx, ypts=ny, zpts=nz)

        spatial_method = pybamm.FiniteVolume3D()
        spatial_method.build(mesh)

        submesh = mesh["negative electrode"]
        n = submesh.npts

        x_nodes = submesh.nodes
        u_exact = (x_nodes**2)[:, np.newaxis]
        var = pybamm.Variable("var", domain=["negative electrode"])
        lap_symbolic = pybamm.div(pybamm.grad(var))

        disc = pybamm.Discretisation(mesh, {"negative electrode": spatial_method})
        disc.set_variable_slices([var])
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }

        lap_disc = disc.process_symbol(lap_symbolic)

        lap_u = lap_disc.evaluate(y=u_exact)

        tol = 5e-4
        two_vec = 2 * np.ones((n, 1))
        np.testing.assert_allclose(lap_u, two_vec, atol=tol, rtol=0)
