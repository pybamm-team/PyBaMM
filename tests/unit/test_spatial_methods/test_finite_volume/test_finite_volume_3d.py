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

            np.testing.assert_allclose(
                node_x.evaluate(None, y_test), y_test[:, np.newaxis]
            )

            # Test Y direction
            edge_y = fin_vol.node_to_edge(c, method=method, direction="y")
            node_y = fin_vol.edge_to_node(edge_y, method=method, direction="y")
            np.testing.assert_allclose(
                node_y.evaluate(None, y_test), y_test[:, np.newaxis]
            )

            # Test Z direction
            edge_z = fin_vol.node_to_edge(c, method=method, direction="z")
            node_z = fin_vol.edge_to_node(edge_z, method=method, direction="z")
            np.testing.assert_allclose(
                node_z.evaluate(None, y_test), y_test[:, np.newaxis]
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
        node_x_coords_per_domain = []
        for dom in whole_cell:
            x_nodes_domain = mesh[dom].nodes[:, 0]
            node_x_coords_per_domain.append(pybamm.Vector(x_nodes_domain, domain=dom))

        v_disc = fin_vol.concatenation(node_x_coords_per_domain)

        # The expected result of DomainConcatenation is the stacking of the evaluated children.
        # Each child_vector.evaluate() results in a column vector (shape N,1).
        # We want to concatenate the 1D arrays.
        expected_stacked_values_list = []
        for child_vector_symbol in node_x_coords_per_domain:
            evaluated_child_array = child_vector_symbol.evaluate()
            expected_stacked_values_list.append(evaluated_child_array[:, 0])

        expected_concatenated_x_nodes = np.concatenate(expected_stacked_values_list)

        np.testing.assert_array_equal(
            v_disc.evaluate()[:, 0],
            expected_concatenated_x_nodes,
        )

        bad_edges = [
            pybamm.Vector(np.ones(mesh[dom].npts + 2), domain=dom) for dom in whole_cell
        ]

        with pytest.raises(
            pybamm.ShapeError, match="expected child size to be n_nodes="
        ):
            fin_vol.concatenation(bad_edges)

    def test_discretise_diffusivity_times_spatial_operator_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        submesh = mesh[whole_cell]

        var = pybamm.Variable("var", domain=whole_cell)
        disc.set_variable_slices([var])

        y_test = np.ones(submesh.npts)[:, np.newaxis]

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
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = disc.mesh["negative electrode"]

        x1 = pybamm.SpatialVariable("x", ["negative electrode"], direction="x")
        x1_disc = disc.process_symbol(x1)
        X, Y, Z = np.meshgrid(
            submesh.nodes_x, submesh.nodes_y, submesh.nodes_z, indexing="ij"
        )
        expected_x = X.flatten(order="F")[:, np.newaxis]

        assert isinstance(x1_disc, pybamm.Vector)
        np.testing.assert_array_equal(x1_disc.evaluate(), expected_x)

        x2 = pybamm.SpatialVariable(
            "x", ["negative electrode", "separator"], direction="x"
        )
        x2_disc = disc.process_symbol(x2)
        assert isinstance(x2_disc, pybamm.Vector)

        combined_submesh = disc.mesh[("negative electrode", "separator")]
        X2, Y2, Z2 = np.meshgrid(
            combined_submesh.nodes_x,
            combined_submesh.nodes_y,
            combined_submesh.nodes_z,
            indexing="ij",
        )
        expected_x2 = X2.flatten(order="F")[:, np.newaxis]

        np.testing.assert_array_equal(x2_disc.evaluate(), expected_x2)

        r = 3 * pybamm.SpatialVariable("r", ["negative electrode"], direction="x")
        r_disc = disc.process_symbol(r)
        assert isinstance(r_disc, pybamm.Vector)

        expected_r = 3 * expected_x
        np.testing.assert_array_equal(r_disc.evaluate(), expected_r)

    def test_mass_matrix_shape_3d(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        N = pybamm.grad(c)
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N)}
        model.initial_conditions = {c: pybamm.Scalar(0)}
        model.boundary_conditions = {
            c: {
                "left": (0, "Dirichlet"),
                "right": (0, "Dirichlet"),
                "front": (0, "Neumann"),
                "back": (0, "Neumann"),
                "bottom": (0, "Neumann"),
                "top": (0, "Neumann"),
            }
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
        y_test = np.ones(submesh.npts)[:, np.newaxis]

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
            whole_cell, {"primary": whole_cell}, direction="x"
        ).entries

        # Fix: Handle different jacobian types
        if hasattr(jacobian, "toarray"):
            jacobian_array = jacobian.toarray()
        elif hasattr(jacobian, "entries"):
            jacobian_array = jacobian.entries.toarray()
        else:
            # For VectorResult or other types, try to convert to array
            jacobian_array = np.array(jacobian)

        np.testing.assert_allclose(jacobian_array[1:-1], grad_matrix.toarray())
        np.testing.assert_allclose(
            jacobian_array[0, 0], grad_matrix.toarray()[0][0] * -2
        )
        np.testing.assert_allclose(
            jacobian_array[-1, -1], grad_matrix.toarray()[-1][-1] * -2
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

        submesh = mesh["negative electrode"]
        n_x, n_y, n_z = submesh.npts_x, submesh.npts_y, submesh.npts_z
        n_nodes_total = submesh.npts

        y_eval_flat = np.arange(1, n_nodes_total + 1)
        y_eval_col_vec = y_eval_flat[:, np.newaxis]

        assert delta_fn_left_disc.domains == delta_fn_left.domains
        assert isinstance(delta_fn_left_disc, pybamm.Multiplication)
        assert isinstance(delta_fn_left_disc.left, pybamm.Matrix)

        vec_left_eval = delta_fn_left_disc.left.evaluate()
        if hasattr(vec_left_eval, "toarray"):
            vec_left_eval = vec_left_eval.toarray()

        expected_vec_left = np.zeros((n_nodes_total, 1))
        area_yz = (submesh.edges_y[-1] - submesh.edges_y[0]) * (
            submesh.edges_z[-1] - submesh.edges_z[0]
        )
        dx_left_face = submesh.d_edges_x[0]
        scale_left = area_yz / dx_left_face

        for k_idx in range(n_z):
            for j_idx in range(n_y):
                node_idx = 0 + j_idx * n_x + k_idx * n_x * n_y
                expected_vec_left[node_idx] = scale_left

        np.testing.assert_allclose(vec_left_eval, expected_vec_left, atol=1e-9)
        assert delta_fn_left_disc.shape == y_eval_col_vec.shape

        # Test right delta function discretization
        assert delta_fn_right_disc.domains == delta_fn_right.domains
        assert isinstance(delta_fn_right_disc, pybamm.Multiplication)
        assert isinstance(delta_fn_right_disc.left, pybamm.Matrix)

        vec_right_eval = delta_fn_right_disc.left.evaluate()
        if hasattr(vec_right_eval, "toarray"):
            vec_right_eval = vec_right_eval.toarray()

        expected_vec_right = np.zeros((n_nodes_total, 1))
        dx_right_face = submesh.d_edges_x[-1]
        scale_right = area_yz / dx_right_face

        for k_idx in range(n_z):
            for j_idx in range(n_y):
                node_idx = (n_x - 1) + j_idx * n_x + k_idx * n_x * n_y
                expected_vec_right[node_idx] = scale_right

        np.testing.assert_allclose(vec_right_eval, expected_vec_right, atol=1e-9)
        assert delta_fn_right_disc.shape == y_eval_col_vec.shape

        x_n = pybamm.SpatialVariable(
            "x_n",
            domain="negative electrode",
            coord_sys=submesh.coord_sys,
            direction="x",
        )
        delta_fn_left_of_var = pybamm.DeltaFunction(var, "left", "negative electrode")
        integral_of_delta_fn = pybamm.Integral(delta_fn_left_of_var, x_n)
        integral_disc = disc.process_symbol(integral_of_delta_fn)

        var_disc = disc.process_symbol(var)
        evaluated_integral = integral_disc.evaluate(y=y_eval_flat)
        if hasattr(evaluated_integral, "toarray"):
            evaluated_integral = evaluated_integral.toarray()

        var_disc_eval = var_disc.evaluate(y=y_eval_flat)
        if hasattr(var_disc_eval, "toarray"):
            var_disc_eval = var_disc_eval.toarray().flatten()

        total_expected = n_y * n_z * 0.5

        np.testing.assert_allclose(
            np.sum(evaluated_integral), total_expected, atol=1e-9
        )

    def test_heaviside_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var = pybamm.Variable("var", domain="negative electrode")
        heav = var > 1

        disc.set_variable_slices([var])
        disc_heav = disc.process_symbol(heav * var)
        submesh = mesh["negative electrode"]
        assert disc_heav.size == submesh.npts

        np.testing.assert_array_equal(
            disc_heav.evaluate(y=2 * np.ones_like(submesh.nodes[:, 0])),
            2 * np.ones((submesh.npts, 1)),
        )
        np.testing.assert_array_equal(
            disc_heav.evaluate(y=-2 * np.ones_like(submesh.nodes[:, 0])),
            np.zeros((submesh.npts, 1)),
        )

    def test_upwind_downwind_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        n = mesh["negative electrode"].npts
        var = pybamm.StateVector(slice(0, n), domain=["negative electrode"])
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

        submesh = mesh["negative electrode"]
        expected_size_x = submesh.npts + submesh.npts_y * submesh.npts_z
        expected_size_y = submesh.npts_x * (submesh.npts_y - 1) * submesh.npts_z
        expected_size_z = submesh.npts_x * submesh.npts_y * (submesh.npts_z - 1)
        expected_size = expected_size_x + expected_size_y + expected_size_z

        assert disc_upwind.size == expected_size
        assert disc_downwind.size == expected_size

        y_test = 2 * np.ones(submesh.npts)
        upwind_eval = disc_upwind.evaluate(y=y_test)
        downwind_eval = disc_downwind.evaluate(y=y_test)

        assert upwind_eval.shape[1] == 1
        assert downwind_eval.shape[1] == 1
        assert upwind_eval.shape[0] == expected_size
        assert downwind_eval.shape[0] == expected_size

        # Test error cases
        disc.bcs = {}
        disc._discretised_symbols = {}
        with pytest.raises(pybamm.ModelError, match="No boundary conditions"):
            disc.process_symbol(upwind)

        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(5), "Neumann"),
                "right": (pybamm.Scalar(3), "Neumann"),
            }
        }
        with pytest.raises(pybamm.ModelError, match="Dirichlet BC required"):
            disc.process_symbol(upwind)
        with pytest.raises(pybamm.ModelError, match="Dirichlet BC required"):
            disc.process_symbol(downwind)

    def test_inner_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume3D(),
        }

        var = pybamm.Variable("var", domain="negative electrode")
        grad_var = pybamm.grad(var)
        inner = pybamm.inner(grad_var, grad_var)

        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.set_variable_slices([var])
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
                "front": (pybamm.Scalar(0), "Neumann"),
                "back": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
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
                "front": (pybamm.Scalar(0), "Neumann"),
                "back": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
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
                "front": (pybamm.Scalar(0), "Neumann"),
                "back": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
            }
        }

        lap_disc = disc.process_symbol(lap_symbolic)

        lap_u = lap_disc.evaluate(y=u_exact)

        tol = 5e-4
        two_vec = 2 * np.ones((n, 1))
        np.testing.assert_allclose(lap_u, two_vec, atol=tol, rtol=0)
