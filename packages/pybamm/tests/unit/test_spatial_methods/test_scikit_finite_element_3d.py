import numpy as np
import pytest

import pybamm
from tests import (
    get_3d_mesh_for_testing,
    get_unit_3d_mesh_for_testing,
)


class TestScikitFiniteElement3D:
    def test_not_implemented(self):
        mesh = get_3d_mesh_for_testing(include_particles=False)
        spatial_method = pybamm.ScikitFiniteElement3D()
        spatial_method.build(mesh)
        assert spatial_method.mesh == mesh
        with pytest.raises(NotImplementedError):
            spatial_method.indefinite_integral(None, None, None)

    def test_discretise_equations_pouch(self):
        mesh = get_3d_mesh_for_testing(geom_type="pouch", include_particles=False)
        spatial_methods = {
            "negative electrode": pybamm.ScikitFiniteElement3D(),
            "separator": pybamm.ScikitFiniteElement3D(),
            "positive electrode": pybamm.ScikitFiniteElement3D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="negative electrode")
        x = pybamm.SpatialVariable("x", ["negative electrode"])
        y = pybamm.SpatialVariable("y", ["negative electrode"])
        z = pybamm.SpatialVariable("z", ["negative electrode"])
        disc.set_variable_slices([var])
        y_test = np.ones(mesh["negative electrode"].npts)
        boundary_pairs = [("x_min", "x_max"), ("z_min", "z_max"), ("y_min", "y_max")]
        for side1, side2 in boundary_pairs:
            for eqn in [
                pybamm.laplacian(var),
                pybamm.grad_squared(var),
                pybamm.laplacian(var) + 2 * var,
                var * var,
                pybamm.Integral(var, [x, y, z]) - 1,
            ]:
                disc.bcs = {
                    var: {
                        side1: (pybamm.Scalar(0), "Dirichlet"),
                        side2: (pybamm.Scalar(1), "Dirichlet"),
                    }
                }
                eqn_disc = disc.process_symbol(eqn)
                result = eqn_disc.evaluate(None, y_test)
                assert result is not None
                disc.bcs = {
                    var: {
                        side1: (pybamm.Scalar(0), "Neumann"),
                        side2: (pybamm.Scalar(1), "Neumann"),
                    }
                }
                eqn_disc = disc.process_symbol(eqn)
                result = eqn_disc.evaluate(None, y_test)
                assert result is not None
                disc.bcs = {
                    var: {
                        side1: (pybamm.Scalar(0), "Neumann"),
                        side2: (pybamm.Scalar(1), "Dirichlet"),
                    }
                }
                eqn_disc = disc.process_symbol(eqn)
                result = eqn_disc.evaluate(None, y_test)
                assert result is not None

    def test_gradient_3d_manufactured(self):
        mesh = get_unit_3d_mesh_for_testing(h=0.3)
        disc = pybamm.Discretisation(
            mesh, {"current collector": pybamm.ScikitFiniteElement3D()}
        )
        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])

        grad_disc = disc.process_symbol(pybamm.grad(var))
        x, y, z = mesh["current collector"].nodes.T
        test_func = 2 * x + 3 * y + 4 * z

        grad_eval = grad_disc.evaluate(None, test_func)

        np.testing.assert_allclose(np.mean(grad_eval[:, 0]), 2, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(np.mean(grad_eval[:, 1]), 3, rtol=1e-7, atol=5e-6)
        np.testing.assert_allclose(np.mean(grad_eval[:, 2]), 4, rtol=1e-7, atol=5e-6)

    def test_laplacian_3d_manufactured_pouch(self):
        mesh = get_unit_3d_mesh_for_testing(h=0.4)
        disc = pybamm.Discretisation(
            mesh, {"current collector": pybamm.ScikitFiniteElement3D()}
        )
        var = pybamm.Variable("var", domain="current collector")

        eqn = pybamm.laplacian(var)

        processed_x = disc.process_symbol(
            pybamm.SpatialVariable("x", ["current collector"])
        )
        processed_y = disc.process_symbol(
            pybamm.SpatialVariable("y", ["current collector"])
        )
        processed_z = disc.process_symbol(
            pybamm.SpatialVariable("z", ["current collector"])
        )

        u_analytical_sym = processed_x * processed_y * processed_z

        all_boundaries = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        disc.bcs = {
            var: {name: (u_analytical_sym, "Dirichlet") for name in all_boundaries}
        }
        disc.set_variable_slices([var])
        eqn_disc = disc.process_symbol(eqn)

        x_num, y_num, z_num = mesh["current collector"].nodes.T
        u_analytical_num = x_num * y_num * z_num

        result = eqn_disc.evaluate(None, u_analytical_num)

        submesh = mesh["current collector"]
        boundary_dofs = np.unique(
            np.concatenate(
                [getattr(submesh, f"{name}_dofs") for name in all_boundaries]
            )
        )
        interior_mask = np.ones(submesh.npts, dtype=bool)
        interior_mask[boundary_dofs] = False
        l2_error = np.sqrt(np.mean(result[interior_mask] ** 2))
        assert l2_error < 1e-10

    def test_divergence_3d_manufactured(self):
        mesh = get_unit_3d_mesh_for_testing(h=0.3)
        disc = pybamm.Discretisation(
            mesh, {"current collector": pybamm.ScikitFiniteElement3D()}
        )

        u = pybamm.Variable("u", domain="current collector")
        disc.set_variable_slices([u])

        x, _y, _z = mesh["current collector"].nodes.T
        u_exact = 4.5 * x**2

        grad_u = pybamm.grad(u)
        div_grad_u = pybamm.div(grad_u)

        div_disc = disc.process_symbol(div_grad_u)
        div_result = div_disc.evaluate(None, u_exact)

        submesh = mesh["current collector"]

        all_boundary_names = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        boundary_dofs = np.unique(
            np.concatenate(
                [getattr(submesh, f"{name}_dofs") for name in all_boundary_names]
            )
        )

        interior_mask = np.ones(submesh.npts, dtype=bool)
        interior_mask[boundary_dofs] = False

        mean_on_interior = np.mean(div_result[interior_mask])

        np.testing.assert_allclose(mean_on_interior, 9, rtol=3e-2, atol=1e-2)

    def test_neumann_boundary_conditions(self):
        mesh = get_unit_3d_mesh_for_testing(h=0.8)
        spatial_method = pybamm.ScikitFiniteElement3D()
        spatial_method.build(mesh)

        var = pybamm.Variable("var", domain="current collector")
        bcs = {var: {"x_max": (pybamm.Scalar(5.0), "Neumann")}}

        boundary_load_disc = spatial_method.laplacian_boundary_load(var, bcs)
        boundary_load_eval = boundary_load_disc.evaluate()

        x_max_dofs = mesh["current collector"].x_max_dofs
        assert np.sum(boundary_load_eval) > 0
        assert np.count_nonzero(boundary_load_eval[x_max_dofs]) > 0

        non_boundary_load = np.delete(boundary_load_eval, x_max_dofs)
        assert np.count_nonzero(np.abs(non_boundary_load) > 1e-12) == 0

    def test_3d_thermal_equation_analytical(self):
        L_x = 2.0
        mesh = get_unit_3d_mesh_for_testing(
            h=0.4, geom_type="pouch", x_max=L_x, y_max=1.0, z_max=1.0
        )
        disc = pybamm.Discretisation(
            mesh, {"current collector": pybamm.ScikitFiniteElement3D()}
        )
        T = pybamm.Variable("T", domain="current collector")

        model = pybamm.BaseModel()
        # Equation: ∇²T + 1 = 0
        model.algebraic = {T: pybamm.laplacian(T) + pybamm.source(1, T)}

        model.boundary_conditions = {
            T: {
                "x_min": (pybamm.Scalar(0), "Dirichlet"),
                "x_max": (pybamm.Scalar(0), "Dirichlet"),
                "y_min": (pybamm.Scalar(0), "Neumann"),
                "y_max": (pybamm.Scalar(0), "Neumann"),
                "z_min": (pybamm.Scalar(0), "Neumann"),
                "z_max": (pybamm.Scalar(0), "Neumann"),
            }
        }
        model.initial_conditions = {T: pybamm.Scalar(0)}
        model.variables = {"T": T}

        disc.process_model(model)
        solver = pybamm.AlgebraicSolver()
        solution = solver.solve(model)
        T_numerical = solution.y.flatten()

        # Define the analytical solution: T(x) = 0.5 * x * (L - x)
        x_coords = mesh["current collector"].nodes[:, 0]
        T_analytical = 0.5 * x_coords * (L_x - x_coords)

        np.testing.assert_allclose(T_numerical, T_analytical, atol=1e-2)

    def test_gradient_squared_3d(self):
        mesh = get_unit_3d_mesh_for_testing(
            xpts=6, ypts=6, zpts=6, geom_type="pouch", include_particles=False
        )
        spatial_methods = {
            "current collector": pybamm.ScikitFiniteElement3D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])
        x = mesh["current collector"].nodes[:, 0]
        y = mesh["current collector"].nodes[:, 1]
        z = mesh["current collector"].nodes[:, 2]
        eqn = pybamm.grad_squared(var)
        eqn_disc = disc.process_symbol(eqn)
        test_function = x**2 + y**2 + z**2
        ans = eqn_disc.evaluate(None, test_function)
        np.testing.assert_array_less(0, ans)

    def test_manufactured_solution_3d(self):
        mesh = get_unit_3d_mesh_for_testing(h=0.4)
        disc = pybamm.Discretisation(
            mesh, {"current collector": pybamm.ScikitFiniteElement3D()}
        )
        submesh = mesh["current collector"]
        x_vertices, y_vertices, z_vertices = submesh.nodes.T

        Lx = np.max(x_vertices) - np.min(x_vertices)
        Ly = np.max(y_vertices) - np.min(y_vertices)
        Lz = np.max(z_vertices) - np.min(z_vertices)

        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])
        var_disc = disc.process_symbol(var)
        np.testing.assert_allclose(
            var_disc.evaluate(None, z_vertices),
            z_vertices[:, np.newaxis],
            rtol=1e-7,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            var_disc.evaluate(None, 6 * y_vertices),
            6 * y_vertices[:, np.newaxis],
            rtol=1e-7,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            var_disc.evaluate(None, 3 * x_vertices),
            3 * x_vertices[:, np.newaxis],
            rtol=1e-7,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            var_disc.evaluate(None, x_vertices * y_vertices * z_vertices),
            (x_vertices * y_vertices * z_vertices)[:, np.newaxis],
            rtol=1e-7,
            atol=1e-6,
        )

        var1 = pybamm.Variable("var1", domain="current collector")
        eqn1 = pybamm.laplacian(var1)
        disc.bcs = {
            var1: {
                "z_min": (pybamm.Scalar(0), "Dirichlet"),
                "z_max": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        disc.set_variable_slices([var1])
        eqn1_disc = disc.process_symbol(eqn1)
        u1 = np.sin(np.pi * z_vertices / Lz)  # Use Lz
        mass1 = pybamm.Mass(var1)
        mass1_disc = disc.process_symbol(mass1)
        soln1 = -((np.pi / Lz) ** 2) * u1  # Use Lz

        np.testing.assert_allclose(
            eqn1_disc.evaluate(None, u1).flatten(),
            mass1_disc.entries @ soln1,
            rtol=1e-2,  # Looser tolerance for coarse mesh
            atol=1e-2,
        )

        var2 = pybamm.Variable("var2", domain="current collector")
        eqn2 = pybamm.laplacian(var2)
        disc.bcs = {
            var2: {
                "y_min": (pybamm.Scalar(0), "Dirichlet"),
                "y_max": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        disc.set_variable_slices([var2])
        eqn2_disc = disc.process_symbol(eqn2)
        u2 = (
            np.cos(np.pi * x_vertices / Lx)
            * np.sin(np.pi * y_vertices / Ly)
            * np.cos(np.pi * z_vertices / Lz)
        )
        mass2 = pybamm.Mass(var2)
        mass2_disc = disc.process_symbol(mass2)
        soln2 = -((np.pi / Lx) ** 2 + (np.pi / Ly) ** 2 + (np.pi / Lz) ** 2) * u2

        np.testing.assert_allclose(
            eqn2_disc.evaluate(None, u2).flatten(),
            mass2_disc.entries @ soln2,
            rtol=1e-2,
            atol=1e-1,
        )

        var3 = pybamm.Variable("var3", domain="current collector")
        eqn3 = pybamm.laplacian(var3)
        all_boundaries = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        disc.bcs = {
            var3: {name: (pybamm.Scalar(0), "Dirichlet") for name in all_boundaries}
        }
        disc.set_variable_slices([var3])
        eqn3_disc = disc.process_symbol(eqn3)
        u3 = (
            np.sin(np.pi * x_vertices / Lx)
            * np.sin(np.pi * y_vertices / Ly)
            * np.sin(np.pi * z_vertices / Lz)
        )
        mass3 = pybamm.Mass(var3)
        mass3_disc = disc.process_symbol(mass3)
        soln3 = -((np.pi / Lx) ** 2 + (np.pi / Ly) ** 2 + (np.pi / Lz) ** 2) * u3
        np.testing.assert_allclose(
            eqn3_disc.evaluate(None, u3).flatten(),
            mass3_disc.entries @ soln3,
            rtol=1e-2,
            atol=1e-1,
        )

    def test_definite_integral_3d(self):
        mesh = get_3d_mesh_for_testing(
            xpts=6, ypts=6, zpts=6, geom_type="pouch", include_particles=False
        )
        spatial_methods = {
            "negative electrode": pybamm.ScikitFiniteElement3D(),
            "separator": pybamm.ScikitFiniteElement3D(),
            "positive electrode": pybamm.ScikitFiniteElement3D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="negative electrode")
        x = pybamm.SpatialVariable("x", ["negative electrode"])
        y = pybamm.SpatialVariable("y", ["negative electrode"])
        z = pybamm.SpatialVariable("z", ["negative electrode"])
        integral_eqn = pybamm.Integral(var, [x, y, z])
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)
        constant_val = 5
        y_test = constant_val * np.ones(mesh["negative electrode"].npts)
        fem_mesh = mesh["negative electrode"]
        lx = fem_mesh.nodes[:, 0].max() - fem_mesh.nodes[:, 0].min()
        ly = fem_mesh.nodes[:, 1].max() - fem_mesh.nodes[:, 1].min()
        lz = fem_mesh.nodes[:, 2].max() - fem_mesh.nodes[:, 2].min()
        expected_volume = lx * ly * lz
        result = integral_eqn_disc.evaluate(None, y_test)
        expected = constant_val * expected_volume
        np.testing.assert_allclose(result, expected, rtol=1e-1, atol=1e-1)

    def test_boundary_value_3d(self):
        mesh = get_3d_mesh_for_testing(
            xpts=6, ypts=6, zpts=6, geom_type="pouch", include_particles=False
        )
        spatial_methods = {
            "negative electrode": pybamm.ScikitFiniteElement3D(),
            "separator": pybamm.ScikitFiniteElement3D(),
            "positive electrode": pybamm.ScikitFiniteElement3D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="negative electrode")
        disc.set_variable_slices([var])
        boundary_faces = ["x_min", "x_max", "z_min", "z_max", "y_min", "y_max"]
        for face in boundary_faces:
            boundary_val = pybamm.BoundaryValue(var, face)
            boundary_val_disc = disc.process_symbol(boundary_val)
            constant_y = np.ones(mesh["negative electrode"].npts)[:, np.newaxis]
            result = boundary_val_disc.evaluate(None, constant_y)
            np.testing.assert_allclose(result, 1, rtol=1e-6, atol=1e-6)

    def test_boundary_integral_3d(self):
        mesh = get_3d_mesh_for_testing(geom_type="pouch")
        disc = pybamm.Discretisation(
            mesh, {"negative electrode": pybamm.ScikitFiniteElement3D()}
        )

        var = pybamm.Variable("var", domain="negative electrode")
        disc.set_variable_slices([var])

        const_value = 5
        var_with_const_value = pybamm.Vector(
            np.full(mesh["negative electrode"].npts, const_value)
        )

        full_boundary_integral = pybamm.BoundaryIntegral(var, region="entire")
        integral_disc = disc.process_symbol(full_boundary_integral)

        numerical_result = integral_disc.evaluate(y=var_with_const_value.entries)

        submesh = mesh["negative electrode"]
        lx = submesh.nodes[:, 0].max() - submesh.nodes[:, 0].min()
        ly = submesh.nodes[:, 1].max() - submesh.nodes[:, 1].min()
        lz = submesh.nodes[:, 2].max() - submesh.nodes[:, 2].min()
        analytical_area = 2 * (lx * ly + lx * lz + ly * lz)

        expected_result = const_value * analytical_area
        np.testing.assert_allclose(numerical_result, expected_result, rtol=1e-2)

    def test_disc_spatial_var_3d(self):
        mesh = get_unit_3d_mesh_for_testing(
            xpts=4, ypts=4, zpts=4, geom_type="pouch", include_particles=False
        )
        spatial_methods = {
            "current collector": pybamm.ScikitFiniteElement3D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        x = pybamm.SpatialVariable("x", ["current collector"])
        y = pybamm.SpatialVariable("y", ["current collector"])
        z = pybamm.SpatialVariable("z", ["current collector"])
        x_disc = disc.process_symbol(x)
        y_disc = disc.process_symbol(y)
        z_disc = disc.process_symbol(z)
        x_actual = mesh["current collector"].nodes[:, 0][:, np.newaxis]
        y_actual = mesh["current collector"].nodes[:, 1][:, np.newaxis]
        z_actual = mesh["current collector"].nodes[:, 2][:, np.newaxis]
        np.testing.assert_array_equal(x_disc.evaluate(), x_actual)
        np.testing.assert_array_equal(y_disc.evaluate(), y_actual)
        np.testing.assert_array_equal(z_disc.evaluate(), z_actual)

    def test_pure_neumann_poisson_3d(self):
        # Solves grad^2 u = 1, with du/dz = Lz at z=Lz, and du/dn=0 elsewhere.
        # The solution is constrained to have zero average value.
        # Exact solution for this case: u(z) = z^2 / 2 - Lz^2 / 6

        u = pybamm.Variable("u", domain="current collector")
        c = pybamm.Variable("c")

        x = pybamm.SpatialVariable("x", ["current collector"])
        y = pybamm.SpatialVariable("y", ["current collector"])
        z = pybamm.SpatialVariable("z", ["current collector"])

        model = pybamm.BaseModel()
        model.algebraic = {
            u: pybamm.laplacian(u)
            - pybamm.source(1, u)
            + c * pybamm.DefiniteIntegralVector(u, vector_type="column"),
            c: pybamm.Integral(u, [x, y, z]) + pybamm.Multiplication(0, c),
        }
        model.initial_conditions = {u: pybamm.Scalar(0), c: pybamm.Scalar(0)}
        Lz = 3.0
        model.boundary_conditions = {
            u: {
                "z_min": (pybamm.Scalar(0), "Neumann"),
                "z_max": (pybamm.Scalar(Lz), "Neumann"),
                "x_min": (pybamm.Scalar(0), "Neumann"),
                "x_max": (pybamm.Scalar(0), "Neumann"),
                "y_min": (pybamm.Scalar(0), "Neumann"),
                "y_max": (pybamm.Scalar(0), "Neumann"),
            }
        }
        model.variables = {"c": c, "u": u}

        mesh = get_unit_3d_mesh_for_testing(h=0.4, z_max=Lz, x_max=1.0, y_max=2.0)

        spatial_methods = {"current collector": pybamm.ScikitFiniteElement3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        solver = pybamm.AlgebraicSolver()
        solution = solver.solve(model)
        u_numerical = solution.y[:-1]

        z_coords = mesh["current collector"].nodes[:, 2]

        u_exact_at_nodes = (z_coords**2 / 2) - (Lz**2 / 6)

        np.testing.assert_allclose(
            u_numerical.flatten(), u_exact_at_nodes.flatten(), rtol=1e-2, atol=5e-2
        )

    def test_dirichlet_bcs_3d(self):
        # Manufactured solution u(x) = a*x^2 + b*x + k
        # Equation: -laplacian(u) + source(2*a, u) = 0  => laplacian(u) = 2*a
        a_val = 3
        b_val = 4
        k_val = 5

        u = pybamm.Variable("u", domain="current collector")
        model = pybamm.BaseModel()

        model.algebraic = {u: -pybamm.laplacian(u) + pybamm.source(2 * a_val, u)}

        model.boundary_conditions = {
            u: {
                "x_min": (pybamm.Scalar(k_val), "Dirichlet"),
                "x_max": (pybamm.Scalar(a_val + b_val + k_val), "Dirichlet"),
                "z_min": (pybamm.Scalar(0), "Neumann"),
                "z_max": (pybamm.Scalar(0), "Neumann"),
                "y_max": (pybamm.Scalar(0), "Neumann"),
                "y_min": (pybamm.Scalar(0), "Neumann"),
            }
        }
        model.initial_conditions = {u: pybamm.Scalar(1)}
        model.variables = {"u": u}

        mesh = get_unit_3d_mesh_for_testing(h=0.4)
        spatial_methods = {"current collector": pybamm.ScikitFiniteElement3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        solver = pybamm.AlgebraicSolver()
        solution = solver.solve(model)

        x_coords = mesh["current collector"].nodes[:, 0]
        u_exact_at_nodes = a_val * x_coords**2 + b_val * x_coords + k_val

        np.testing.assert_allclose(
            solution.y.flatten(), u_exact_at_nodes.flatten(), rtol=1e-2, atol=1e-2
        )

    def test_scalar_field_discretization(self):
        mesh = get_unit_3d_mesh_for_testing(h=0.2)
        disc = pybamm.Discretisation(
            mesh, {"current collector": pybamm.ScikitFiniteElement3D()}
        )

        phi = pybamm.Variable("phi", domain="current collector")

        disc.set_variable_slices([phi])
        phi_disc = disc.process_symbol(phi)

        nodes = mesh["current collector"].nodes
        x_vals, y_vals, z_vals = nodes.T

        constant_field = np.full_like(x_vals, 5.0)
        result_1 = phi_disc.evaluate(None, constant_field)
        np.testing.assert_allclose(
            result_1.flatten(),
            constant_field,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Constant scalar field should be preserved exactly",
        )

        linear_field = 2 * x_vals + 3 * y_vals + 4 * z_vals
        result_2 = phi_disc.evaluate(None, linear_field)
        np.testing.assert_allclose(
            result_2.flatten(),
            linear_field,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Linear scalar field should be preserved exactly",
        )

        quadratic_field = x_vals**2 + y_vals**2 + z_vals**2
        result_3 = phi_disc.evaluate(None, quadratic_field)
        np.testing.assert_allclose(
            result_3.flatten(),
            quadratic_field,
            rtol=1e-8,
            atol=1e-8,
            err_msg="Quadratic scalar field should be well-preserved",
        )

        mixed_field = x_vals * y_vals + y_vals * z_vals + x_vals * z_vals
        result_4 = phi_disc.evaluate(None, mixed_field)
        np.testing.assert_allclose(
            result_4.flatten(),
            mixed_field,
            rtol=1e-8,
            atol=1e-8,
            err_msg="Mixed polynomial scalar field should be well-preserved",
        )

        assert np.abs(np.max(result_3) - np.max(quadratic_field)) < 1e-6
        assert np.abs(np.min(result_3) - np.min(quadratic_field)) < 1e-6

        mid_idx = len(x_vals) // 2
        x_mid, y_mid, z_mid = x_vals[mid_idx], y_vals[mid_idx], z_vals[mid_idx]
        expected_at_mid = x_mid**2 + y_mid**2 + z_mid**2
        actual_at_mid = result_3[mid_idx, 0]

        np.testing.assert_allclose(
            actual_at_mid,
            expected_at_mid,
            rtol=1e-8,
            atol=1e-8,
            err_msg="Scalar field should be accurate at specific points",
        )

    def test_gradient_cylinder_manufactured(self):
        mesh = get_unit_3d_mesh_for_testing(
            geom_type="cylinder", r_inner=0.1, radius=0.5, height=1.0, h=0.15
        )
        disc = pybamm.Discretisation(
            mesh, {"current collector": pybamm.ScikitFiniteElement3D()}
        )
        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])
        grad_disc = disc.process_symbol(pybamm.grad(var))
        nodes = mesh["current collector"].nodes
        x, y, z = nodes.T
        r = np.sqrt(x**2 + y**2)
        u_exact_num = r**2 * z
        grad_u_exact_r = 2 * r * z
        grad_u_exact_theta = np.zeros_like(r)
        grad_u_exact_z = r**2
        grad_eval = grad_disc.evaluate(None, u_exact_num)
        l2_error_r = np.sqrt(np.mean((grad_eval[:, 0] - grad_u_exact_r) ** 2))
        l2_error_theta = np.sqrt(np.mean((grad_eval[:, 1] - grad_u_exact_theta) ** 2))
        l2_error_z = np.sqrt(np.mean((grad_eval[:, 2] - grad_u_exact_z) ** 2))
        assert l2_error_r < 0.1
        assert l2_error_theta < 5e-2
        assert l2_error_z < 0.1

    def test_laplacian_cylinder_manufactured(self):
        mesh = get_unit_3d_mesh_for_testing(
            geom_type="cylinder", r_inner=0.2, radius=0.5, height=1.0, h=0.4
        )
        model = pybamm.BaseModel()
        u = pybamm.Variable("u", "current collector")
        r = pybamm.SpatialVariable("r", "current collector")
        z = pybamm.SpatialVariable("z", "current collector")
        u_exact = r**2 * z
        f = -4 * z
        model.algebraic = {u: pybamm.laplacian(u) - pybamm.source(f, u)}
        model.boundary_conditions = {
            u: {
                "z_min": (u_exact, "Dirichlet"),
                "z_max": (u_exact, "Dirichlet"),
                "r_min": (u_exact, "Dirichlet"),
                "r_max": (u_exact, "Dirichlet"),
            }
        }
        model.initial_conditions = {u: pybamm.Scalar(0)}
        model.variables = {"u": u}
        spatial_methods = {"current collector": pybamm.ScikitFiniteElement3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        solver = pybamm.AlgebraicSolver()
        solution = solver.solve(model)
        u_numerical = solution.y.flatten()
        nodes = mesh["current collector"].nodes
        x_nodes, y_nodes, z_nodes = nodes.T
        r_nodes = np.sqrt(x_nodes**2 + y_nodes**2)
        u_analytical = r_nodes**2 * z_nodes
        l2_error = np.sqrt(np.mean((u_numerical - u_analytical) ** 2))
        assert l2_error < 0.08

    def test_divergence_cylinder_manufactured(self):
        mesh = get_unit_3d_mesh_for_testing(
            geom_type="cylinder", r_inner=0.1, radius=0.5, height=1.0, h=0.2
        )
        disc = pybamm.Discretisation(
            mesh, {"current collector": pybamm.ScikitFiniteElement3D()}
        )
        u = pybamm.Variable("u", domain="current collector")
        disc.set_variable_slices([u])

        nodes = mesh["current collector"].nodes
        x, y, z = nodes.T
        r = np.sqrt(x**2 + y**2)

        # Manufactured solution u = r^3 * z
        # div(grad(u)) = lap(u) = 9*r*z
        u_exact_num = r**3 * z
        div_grad_u_exact_num = 9 * r * z

        div_grad_u_sym = pybamm.div(pybamm.grad(u))
        div_grad_u_disc = disc.process_symbol(div_grad_u_sym)
        div_grad_u_eval = div_grad_u_disc.evaluate(None, u_exact_num).flatten()

        l2_error = np.sqrt(np.mean((div_grad_u_eval - div_grad_u_exact_num) ** 2))
        assert l2_error < 1.5

    def test_cylinder_mesh_properties(self):
        mesh = get_unit_3d_mesh_for_testing(
            geom_type="cylinder", radius=0.4, height=0.8
        )
        nodes = mesh["current collector"].nodes
        radii = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
        assert radii.max() < 0.4 + 1e-7
        assert radii.min() > 0.0 - 1e-7
        assert nodes[:, 2].min() > 0.0 - 1e-7
        assert nodes[:, 2].max() < 0.8 + 1e-7

    def test_boundary_integral_3d_pouch(self):
        x_max, y_max, z_max = 1.0, 2.0, 3.0
        mesh = get_unit_3d_mesh_for_testing(
            geom_type="pouch", h=0.3, x_max=x_max, y_max=y_max, z_max=z_max
        )
        domain = "current collector"
        disc = pybamm.Discretisation(mesh, {domain: pybamm.ScikitFiniteElement3D()})

        var = pybamm.Variable("var", domain=domain)
        disc.set_variable_slices([var])

        y_test = np.ones(mesh[domain].npts)

        regions_and_areas = {
            "x_min": y_max * z_max,
            "x_max": y_max * z_max,
            "y_min": x_max * z_max,
            "y_max": x_max * z_max,
            "z_min": x_max * y_max,
            "z_max": x_max * y_max,
        }

        for region, analytical_area in regions_and_areas.items():
            integral_op = pybamm.BoundaryIntegral(var, region=region)
            integral_disc = disc.process_symbol(integral_op)
            numerical_area = integral_disc.evaluate(y=y_test)
            np.testing.assert_allclose(
                numerical_area,
                analytical_area,
                rtol=5e-2,
                err_msg=f"Failed for region: '{region}'",
            )

    def test_discretise_expression_cylinder(self):
        mesh = get_unit_3d_mesh_for_testing(
            geom_type="cylinder", r_inner=0.2, radius=0.5, height=1.0, h=0.1
        )
        disc = pybamm.Discretisation(
            mesh, {"current collector": pybamm.ScikitFiniteElement3D()}
        )

        r_sym = pybamm.SpatialVariable("r", ["current collector"])
        z_sym = pybamm.SpatialVariable("z", ["current collector"])
        expression_sym = r_sym**2 + z_sym

        expression_disc = disc.process_symbol(expression_sym)
        result_numerical = expression_disc.evaluate().flatten()

        nodes = mesh["current collector"].nodes
        x, y, z = nodes.T
        r_analytical = np.sqrt(x**2 + y**2)
        result_analytical = r_analytical**2 + z

        np.testing.assert_allclose(result_numerical, result_analytical, atol=1e-12)

    def test_scalar_times_gradient_3d(self):
        mesh = get_unit_3d_mesh_for_testing(h=0.3, x_max=1.0, y_max=1.0, z_max=1.0)
        disc = pybamm.Discretisation(
            mesh, {"current collector": pybamm.ScikitFiniteElement3D()}
        )
        T = pybamm.Variable("T", domain="current collector")
        disc.set_variable_slices([T])

        expression_sym = T * pybamm.grad(T)
        expression_disc = disc.process_symbol(expression_sym)

        nodes = mesh["current collector"].nodes
        x, y, z = nodes.T
        T_analytical_func = 2 * x + 3 * y + 4 * z

        numerical_result = expression_disc.evaluate(None, T_analytical_func)

        grad_T_analytical = np.array([2.0, 3.0, 4.0])
        analytical_result = (
            T_analytical_func[:, np.newaxis] * grad_T_analytical[np.newaxis, :]
        )

        np.testing.assert_allclose(numerical_result, analytical_result)
