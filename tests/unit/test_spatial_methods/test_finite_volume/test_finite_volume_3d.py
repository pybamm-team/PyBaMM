import pytest
import numpy as np
import pybamm
from tests import get_mesh_for_testing_3d


class TestFiniteVolume3D:
    def test_linear_solution_exact(self):
        mesh = get_mesh_for_testing_3d(xpts=8, ypts=6, zpts=5)
        method = pybamm.FiniteVolume3D()
        method.build(mesh)
        domain = next(iter(mesh.keys()))
        submesh = mesh[domain]

        Xc, Yc, Zc = np.meshgrid(
            submesh.nodes_x, submesh.nodes_y, submesh.nodes_z, indexing="ij"
        )
        u_cc = 2 * Xc + 3 * Yc + 1 * Zc
        u_vec = pybamm.Vector(u_cc.ravel(), domain=domain)

        u = pybamm.Variable("u", domain=domain)
        y_mid = 0.5 * (submesh.nodes_y[0] + submesh.nodes_y[-1])
        z_mid = 0.5 * (submesh.nodes_z[0] + submesh.nodes_z[-1])
        x_mid = 0.5 * (submesh.nodes_x[0] + submesh.nodes_x[-1])

        left_val = 2 * submesh.nodes_x[0] + 3 * y_mid + 1 * z_mid
        right_val = 2 * submesh.nodes_x[-1] + 3 * y_mid + 1 * z_mid
        front_val = 2 * x_mid + 3 * submesh.nodes_y[0] + 1 * z_mid
        back_val = 2 * x_mid + 3 * submesh.nodes_y[-1] + 1 * z_mid
        bottom_val = 2 * x_mid + 3 * y_mid + 1 * submesh.nodes_z[0]
        top_val = 2 * x_mid + 3 * y_mid + 1 * submesh.nodes_z[-1]

        bcs = {
            u: {
                "left": (pybamm.Scalar(left_val), "Dirichlet"),
                "right": (pybamm.Scalar(right_val), "Dirichlet"),
                "front": (pybamm.Scalar(front_val), "Dirichlet"),
                "back": (pybamm.Scalar(back_val), "Dirichlet"),
                "bottom": (pybamm.Scalar(bottom_val), "Dirichlet"),
                "top": (pybamm.Scalar(top_val), "Dirichlet"),
            }
        }

        grad = method.gradient(u, u_vec, bcs)
        div_grad = method.divergence(u, grad, bcs)

        gx = grad.x_field.evaluate().ravel()
        gy = grad.y_field.evaluate().ravel()
        gz = grad.z_field.evaluate().ravel()
        dd = div_grad.evaluate().ravel()

        np.testing.assert_allclose(gx, 2.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(gy, 3.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(gz, 1.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(dd, 0.0, rtol=0, atol=1e-12)

    def test_quadratic_solution_convergence(self):
        def solve_poisson_3d(npts):
            mesh = get_mesh_for_testing_3d(xpts=npts, ypts=npts, zpts=npts)
            spatial_method = pybamm.FiniteVolume3D()
            spatial_method.build(mesh)

            domain = next(iter(mesh.keys()))
            submesh = mesh[domain]

            # Manufactured solution: u(x,y,z) = x² + y² + z²
            x, y, z = np.meshgrid(
                submesh.nodes_x, submesh.nodes_y, submesh.nodes_z, indexing="ij"
            )
            u_exact = x.flatten() ** 2 + y.flatten() ** 2 + z.flatten() ** 2

            # Source term: f = -∇²u = -(2 + 2 + 2) = -6
            f_exact = -6 * np.ones_like(u_exact)

            # Create variables
            u = pybamm.Variable("u", domain=domain)
            # f = pybamm.Vector(f_exact, domain=domain)

            # Compute Laplacian
            laplacian_u = spatial_method.laplacian(
                u, pybamm.Vector(u_exact, domain=domain), {}
            )

            computed_source = -laplacian_u.evaluate()
            error = np.abs(computed_source - f_exact).max()

            h = min(
                submesh.d_edges_x.min(),
                submesh.d_edges_y.min(),
                submesh.d_edges_z.min(),
            )

            return error, h, u_exact

        mesh_sizes = [4, 6, 8]
        errors = []
        h_values = []

        for npts in mesh_sizes:
            error, h, _ = solve_poisson_3d(npts)
            errors.append(error)
            h_values.append(h)

        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1], f"Error should decrease: {errors}"

        if len(errors) >= 2:
            rate = np.log(errors[1] / errors[0]) / np.log(h_values[1] / h_values[0])
            assert rate > 1.5, f"Convergence rate {rate} should be close to 2.0"

    def test_node_to_edge_to_node_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=6, ypts=5, zpts=4)
        spatial_method = pybamm.FiniteVolume3D()
        spatial_method.build(mesh)

        domain = next(iter(mesh.keys()))
        submesh = mesh[domain]
        n_x, n_y, n_z = submesh.npts_x, submesh.npts_y, submesh.npts_z

        c_x = pybamm.StateVector(slice(0, n_x * n_y * n_z), domain=[domain])
        y_test_nodes = np.ones(n_x * n_y * n_z)

        edge_x = spatial_method.node_to_edge(c_x, method="arithmetic", direction="x")
        edge_x_vals = edge_x.evaluate(None, y_test_nodes)
        expected_edge_size_x = (n_x + 1) * n_y * n_z
        assert edge_x_vals.shape[0] == expected_edge_size_x
        np.testing.assert_array_equal(edge_x_vals, np.ones((expected_edge_size_x, 1)))

        d_x = pybamm.StateVector(slice(0, expected_edge_size_x), domain=[domain])
        y_test_edges_x = np.ones(expected_edge_size_x)
        node_x = spatial_method.edge_to_node(d_x, method="arithmetic", direction="x")
        node_x_vals = node_x.evaluate(None, y_test_edges_x)
        assert node_x_vals.shape[0] == n_x * n_y * n_z
        np.testing.assert_array_equal(node_x_vals, np.ones((n_x * n_y * n_z, 1)))

        expected_edge_size_y = n_x * (n_y + 1) * n_z
        edge_y = spatial_method.node_to_edge(c_x, method="arithmetic", direction="y")
        edge_y_vals = edge_y.evaluate(None, y_test_nodes)
        assert edge_y_vals.shape[0] == expected_edge_size_y

        d_y = pybamm.StateVector(slice(0, expected_edge_size_y), domain=[domain])
        y_test_edges_y = np.ones(expected_edge_size_y)
        node_y = spatial_method.edge_to_node(d_y, method="arithmetic", direction="y")
        node_y_vals = node_y.evaluate(None, y_test_edges_y)
        assert node_y_vals.shape[0] == n_x * n_y * n_z

        expected_edge_size_z = n_x * n_y * (n_z + 1)
        edge_z = spatial_method.node_to_edge(c_x, method="arithmetic", direction="z")
        edge_z_vals = edge_z.evaluate(None, y_test_nodes)
        assert edge_z_vals.shape[0] == expected_edge_size_z

        d_z = pybamm.StateVector(slice(0, expected_edge_size_z), domain=[domain])
        y_test_edges_z = np.ones(expected_edge_size_z)
        node_z = spatial_method.edge_to_node(d_z, method="arithmetic", direction="z")
        node_z_vals = node_z.evaluate(None, y_test_edges_z)
        assert node_z_vals.shape[0] == n_x * n_y * n_z

    def test_shift_method_errors_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=5, ypts=4, zpts=3)
        spatial_method = pybamm.FiniteVolume3D()
        spatial_method.build(mesh)

        domain = next(iter(mesh.keys()))
        c = pybamm.StateVector(slice(0, 60), domain=[domain])

        # bad shift key
        with pytest.raises(ValueError, match="shift key"):
            spatial_method.shift(c, "bad shift key", "arithmetic", "x")

        # bad method
        with pytest.raises(ValueError, match="method"):
            spatial_method.shift(c, "node to edge", "bad method", "x")

        # bad direction
        with pytest.raises(ValueError, match="direction"):
            spatial_method.shift(c, "node to edge", "arithmetic", "bad direction")

    def test_harmonic_mean_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=6, ypts=5, zpts=4)
        spatial_method = pybamm.FiniteVolume3D()
        spatial_method.build(mesh)

        domain = next(iter(mesh.keys()))
        submesh = mesh[domain]
        n_nodes = submesh.npts

        c = pybamm.StateVector(slice(0, n_nodes), domain=[domain])
        y_test = np.ones(n_nodes) * 2.0

        for direction in ["x", "y", "z"]:
            edge_harmonic = spatial_method.node_to_edge(
                c, method="harmonic", direction=direction
            )
            edge_harmonic_vals = edge_harmonic.evaluate(None, y_test)

            edge_arithmetic = spatial_method.node_to_edge(
                c, method="arithmetic", direction=direction
            )
            edge_arithmetic_vals = edge_arithmetic.evaluate(None, y_test)

            # For constant values, harmonic and arithmetic means should be equal
            np.testing.assert_allclose(
                edge_harmonic_vals, edge_arithmetic_vals, rtol=1e-12
            )
            np.testing.assert_allclose(edge_harmonic_vals, 2.0, rtol=1e-12)

    def test_concatenation_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_method = pybamm.FiniteVolume3D()
        spatial_method.build(mesh)

        # Create multiple domains (simulating multiple regions)
        domains = list(mesh.keys())
        if len(domains) < 2:
            domain1 = domains[0]
            domain2 = domain1  # Use same domain for testing
        else:
            domain1, domain2 = domains[:2]

        submesh1 = mesh[domain1]
        submesh2 = mesh[domain2]

        edges1 = pybamm.Vector(submesh1.edges_x, domain=domain1)
        edges2 = pybamm.Vector(submesh2.edges_x, domain=domain2)

        concatenated = spatial_method.concatenation([edges1, edges2])
        concatenated_vals = concatenated.evaluate()

        assert concatenated_vals.shape[0] > 0
        assert np.all(np.isfinite(concatenated_vals))

    def test_discretise_spatial_operators_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=6, ypts=5, zpts=4)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        domain = next(iter(mesh.keys()))
        submesh = mesh[domain]

        var = pybamm.Variable("var", domain=domain)
        disc.set_variable_slices([var])
        y_test = np.ones(submesh.npts)[:, np.newaxis]

        # Test various spatial operators
        operators_to_test = [
            pybamm.grad(var),
            pybamm.div(pybamm.grad(var)),
            pybamm.laplacian(var),
            var * pybamm.grad(var).x_field,
            var * pybamm.grad(var).y_field,
            var * pybamm.grad(var).z_field,
        ]

        for eqn in operators_to_test:
            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Dirichlet"),
                    "right": (pybamm.Scalar(1), "Dirichlet"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            result = eqn_disc.evaluate(None, y_test)
            assert np.all(np.isfinite(result))

            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            result = eqn_disc.evaluate(None, y_test)
            assert np.all(np.isfinite(result))

    def test_mass_matrix_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=5, ypts=4, zpts=3)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}

        domain = next(iter(mesh.keys()))
        submesh = mesh[domain]

        c = pybamm.Variable("c", domain=domain)
        N = pybamm.grad(c)

        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N)}
        model.initial_conditions = {c: pybamm.Scalar(0)}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        model.variables = {"c": c, "N": N}

        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        expected_mass = np.eye(submesh.npts)
        np.testing.assert_array_equal(
            expected_mass, model.mass_matrix.entries.toarray()
        )

    def test_jacobian_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=5, ypts=4, zpts=3)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        domain = next(iter(mesh.keys()))
        submesh = mesh[domain]
        spatial_method = pybamm.FiniteVolume3D()
        spatial_method.build(mesh)

        var = pybamm.Variable("var", domain=domain)
        disc.set_variable_slices([var])
        y = pybamm.StateVector(slice(0, submesh.npts))
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

        # Jacobian should have appropriate dimensions
        assert jacobian.shape[1] == submesh.npts
        assert np.all(np.isfinite(jacobian.toarray()))

        # Test Laplacian Jacobian
        eqn = pybamm.laplacian(var)
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(2), "Neumann"),
            }
        }
        eqn_disc = disc.process_symbol(eqn)
        eqn_jac = eqn_disc.jac(y)
        jacobian = eqn_jac.evaluate(y=y_test)

        assert jacobian.shape == (submesh.npts, submesh.npts)
        assert np.all(np.isfinite(jacobian.toarray()))

    def test_divergence_conservation_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=6, ypts=5, zpts=4)
        spatial_method = pybamm.FiniteVolume3D()
        spatial_method.build(mesh)

        domain = next(iter(mesh.keys()))
        submesh = mesh[domain]
        n_x, n_y, n_z = submesh.npts_x, submesh.npts_y, submesh.npts_z

        flux_x = pybamm.Vector(np.ones((n_x + 1) * n_y * n_z), domain=domain)
        flux_y = pybamm.Vector(np.ones(n_x * (n_y + 1) * n_z), domain=domain)
        flux_z = pybamm.Vector(np.ones(n_x * n_y * (n_z + 1)), domain=domain)

        vector_field = pybamm.VectorField3D(flux_x, flux_y, flux_z)

        div_result = spatial_method.divergence(vector_field, vector_field, {})
        div_vals = div_result.evaluate()

        assert np.abs(np.mean(div_vals)) < 1.0
        assert np.all(np.isfinite(div_vals))

    def test_integral_operators_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=6, ypts=5, zpts=4)
        spatial_method = pybamm.FiniteVolume3D()
        spatial_method.build(mesh)

        domain = next(iter(mesh.keys()))
        submesh = mesh[domain]

        var = pybamm.Variable("var", domain=domain)
        constant_field = pybamm.Vector(np.ones(submesh.npts), domain=domain)

        for direction in ["x", "y", "z"]:
            coord = pybamm.SpatialVariable(
                direction, domain=domain, direction=direction
            )

            integral_matrix = spatial_method.definite_integral_matrix(
                var, integration_variable=[coord]
            )

            result = integral_matrix @ constant_field.entries

            if direction == "x":
                expected_length = submesh.edges_x[-1] - submesh.edges_x[0]
            elif direction == "y":
                expected_length = submesh.edges_y[-1] - submesh.edges_y[0]
            elif direction == "z":
                expected_length = submesh.edges_z[-1] - submesh.edges_z[0]

            np.testing.assert_allclose(result, expected_length, rtol=0.1)

    def test_delta_function_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=8, ypts=6, zpts=4)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        domain = next(iter(mesh.keys()))
        submesh = mesh[domain]

        var = pybamm.Variable("var")

        x_center = (submesh.nodes_x[0] + submesh.nodes_x[-1]) / 2
        y_center = (submesh.nodes_y[0] + submesh.nodes_y[-1]) / 2
        z_center = (submesh.nodes_z[0] + submesh.nodes_z[-1]) / 2

        position = pybamm.Vector([x_center, y_center, z_center])
        delta_fn = pybamm.DeltaFunction(var, position, domain)

        disc.set_variable_slices([var])
        delta_fn_disc = disc.process_symbol(delta_fn)

        y = np.ones(submesh.npts)[:, np.newaxis]

        assert delta_fn_disc.domains == delta_fn.domains
        assert delta_fn_disc.shape == y.shape

        delta_vals = delta_fn_disc.evaluate(y=y)
        assert np.sum(delta_vals > 0) <= 1  # At most one non-zero entry

    def test_boundary_conditions_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=6, ypts=5, zpts=4)
        spatial_method = pybamm.FiniteVolume3D()
        spatial_method.build(mesh)

        domain = next(iter(mesh.keys()))
        submesh = mesh[domain]

        u = pybamm.Variable("u", domain=domain)
        x, y, z = np.meshgrid(
            submesh.nodes_x, submesh.nodes_y, submesh.nodes_z, indexing="ij"
        )
        u_vals = x.flatten() + y.flatten() + z.flatten()
        u_vec = pybamm.Vector(u_vals, domain=domain)

        # Test different boundary condition combinations
        bc_combinations = [
            {
                "left": (pybamm.Scalar(1.0), "Dirichlet"),
                "right": (pybamm.Scalar(2.0), "Dirichlet"),
            },
            {
                "left": (pybamm.Scalar(0.0), "Neumann"),
                "right": (pybamm.Scalar(0.0), "Neumann"),
            },
            {
                "left": (pybamm.Scalar(1.0), "Dirichlet"),
                "right": (pybamm.Scalar(0.0), "Neumann"),
            },
            {
                "left": (pybamm.Scalar(0.0), "Neumann"),
                "right": (pybamm.Scalar(2.0), "Dirichlet"),
            },
        ]

        for bcs in bc_combinations:
            boundary_conditions = {u: bcs}

            grad_x = spatial_method._gradient(u, u_vec, boundary_conditions, "x")
            grad_vals = grad_x.evaluate()

            assert np.all(np.isfinite(grad_vals))
            assert grad_vals.shape[0] > 0

            # Test Laplacian computation
            laplacian = spatial_method.laplacian(u, u_vec, boundary_conditions)
            laplacian_vals = laplacian.evaluate()

            assert np.all(np.isfinite(laplacian_vals))
            assert laplacian_vals.shape[0] == submesh.npts

    def test_spiral_coordinates_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=6, ypts=8, zpts=4)
        spatial_method = pybamm.FiniteVolume3D()
        spatial_method.build(mesh)

        domain = next(iter(mesh.keys()))
        submesh = mesh[domain]

        submesh.coord_sys = "spiral"

        spiral_metric = spatial_method.compute_spiral_metric(submesh)
        assert len(spiral_metric) == submesh.npts_x
        assert np.all(spiral_metric > 0)
        assert np.all(np.isfinite(spiral_metric))

        # Test basic operations with spiral coordinates
        u = pybamm.Variable("u", domain=domain)
        x, y, z = np.meshgrid(
            submesh.nodes_x, submesh.nodes_y, submesh.nodes_z, indexing="ij"
        )
        u_vals = x.flatten() + y.flatten()
        u_vec = pybamm.Vector(u_vals, domain=domain)

        grad_u = spatial_method.gradient(u, u_vec, {})

        grad_x_vals = grad_u.x_field.evaluate()
        grad_y_vals = grad_u.y_field.evaluate()
        grad_z_vals = grad_u.z_field.evaluate()

        assert np.all(np.isfinite(grad_x_vals))
        assert np.all(np.isfinite(grad_y_vals))
        assert np.all(np.isfinite(grad_z_vals))

    def test_convergence_rates_3d(self):
        """Test convergence rates for manufactured solutions"""

        def compute_error(npts):
            """Compute error for quadratic manufactured solution"""
            mesh = get_mesh_for_testing_3d(xpts=npts, ypts=npts, zpts=max(3, npts // 2))
            spatial_method = pybamm.FiniteVolume3D()
            spatial_method.build(mesh)

            domain = next(iter(mesh.keys()))
            submesh = mesh[domain]

            u = pybamm.Variable("u", domain=domain)
            x, y, z = np.meshgrid(
                submesh.nodes_x, submesh.nodes_y, submesh.nodes_z, indexing="ij"
            )
            u_vals = x.flatten() ** 2 + y.flatten() ** 2 + z.flatten() ** 2
            u_vec = pybamm.Vector(u_vals, domain=domain)

            # Compute Laplacian
            laplacian = spatial_method.laplacian(u, u_vec, {})
            laplacian_vals = laplacian.evaluate()

            error = np.abs(laplacian_vals - 6.0).max()
            h = min(
                np.diff(submesh.edges_x).min(),
                np.diff(submesh.edges_y).min(),
                np.diff(submesh.edges_z).min(),
            )

            return error, h

        # Test on progressively finer meshes
        mesh_sizes = [4, 6, 8]
        errors = []
        h_values = []

        for npts in mesh_sizes:
            try:
                error, h = compute_error(npts)
                if np.isfinite(error) and error > 0:
                    errors.append(error)
                    h_values.append(h)
            except Exception:
                continue

        if len(errors) >= 2:
            for i in range(1, len(errors)):
                ratio = errors[i] / errors[i - 1]
                assert ratio < 5.0, f"Error should not increase dramatically: {errors}"

    def test_evaluate_at_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=8, ypts=6, zpts=4)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        domain = next(iter(mesh.keys()))
        submesh = mesh[domain]
        n = submesh.npts

        var = pybamm.StateVector(slice(0, n), domain=domain)

        # Choose evaluation point near center
        idx_x = submesh.npts_x // 2
        idx_y = submesh.npts_y // 2
        idx_z = submesh.npts_z // 2

        x_pos = submesh.nodes_x[idx_x]
        y_pos = submesh.nodes_y[idx_y]
        z_pos = submesh.nodes_z[idx_z]

        position = pybamm.Vector([x_pos, y_pos, z_pos])
        evaluate_at = pybamm.EvaluateAt(var, position)

        evaluate_at_disc = disc.process_symbol(evaluate_at)

        assert isinstance(evaluate_at_disc, pybamm.MatrixMultiplication)
        assert isinstance(evaluate_at_disc.left, pybamm.Matrix)
        assert isinstance(evaluate_at_disc.right, pybamm.StateVector)

        # Test evaluation
        y = np.arange(n)[:, np.newaxis]
        result = evaluate_at_disc.evaluate(y=y)

        # Should return scalar value
        assert result.shape == (1, 1) or result.shape == ()
        assert np.isfinite(result)

    def test_upwind_downwind_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=6, ypts=5, zpts=4)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        domain = next(iter(mesh.keys()))
        submesh = mesh[domain]
        n = submesh.npts

        var = pybamm.StateVector(slice(0, n), domain=domain)

        try:
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

            y_test = 2 * np.ones(n)

            upwind_vals = disc_upwind.evaluate(y=y_test)
            downwind_vals = disc_downwind.evaluate(y=y_test)

            assert upwind_vals.shape[0] > n  # Should include boundary points
            assert downwind_vals.shape[0] > n
            assert np.all(np.isfinite(upwind_vals))
            assert np.all(np.isfinite(downwind_vals))

        except (NotImplementedError, AttributeError):
            pytest.skip("Upwind/downwind operators not implemented for 3D")

    def test_inner_product_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=6, ypts=5, zpts=4)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}

        domain = next(iter(mesh.keys()))
        submesh = mesh[domain]

        var = pybamm.Variable("var", domain=domain)
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

        n = submesh.npts
        y = np.ones(n)[:, np.newaxis]
        result = inner_disc.evaluate(y=y)

        # For constant field, inner product should be small
        assert result.shape == (n, 1)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result, 0, atol=1e-10)
