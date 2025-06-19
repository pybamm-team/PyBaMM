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

    def test_discretise_equations_box(self):
        mesh = get_3d_mesh_for_testing(geom_type="box", include_particles=False)
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
        boundary_pairs = [("left", "right"), ("front", "back"), ("bottom", "top")]
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

    def test_laplacian_3d_manufactured_box(self):
        mesh = get_unit_3d_mesh_for_testing(h=0.1)
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

        all_boundaries = ["left", "right", "bottom", "top", "front", "back"]
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

    def test_discretise_equations_cylinder(self):
        try:
            mesh = get_unit_3d_mesh_for_testing(
                xpts=5,
                ypts=5,
                zpts=5,
                geom_type="cylinder",
                radius=0.5,
                height=1.0,
                h=0.1,
            )
        except pybamm.DiscretisationError as e:
            pytest.skip(f"Cylinder mesh generation failed: {e}")

        spatial_methods = {"current collector": pybamm.ScikitFiniteElement3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="current collector")
        source_term = pybamm.source(4, var)
        eqn = pybamm.laplacian(var) - source_term

        x_sym = pybamm.SpatialVariable("x", "current collector")
        y_sym = pybamm.SpatialVariable("y", "current collector")
        processed_x = disc.process_symbol(x_sym)
        processed_y = disc.process_symbol(y_sym)
        u_analytical_sym = processed_x**2 + processed_y**2

        all_boundaries = ["side wall", "top cap", "bottom cap"]
        disc.bcs = {
            var: {name: (u_analytical_sym, "Dirichlet") for name in all_boundaries}
        }

        disc.set_variable_slices([var])
        eqn_disc = disc.process_symbol(eqn)
        submesh = mesh["current collector"]
        x_num, y_num, z_num = submesh.nodes.T
        u_analytical_num = x_num**2 + y_num**2
        result = eqn_disc.evaluate(None, u_analytical_num)

        boundary_dofs = np.unique(
            np.concatenate(
                [getattr(submesh, f"{name}_dofs") for name in all_boundaries]
            )
        )
        interior_mask = np.ones(submesh.npts, dtype=bool)
        interior_mask[boundary_dofs] = False

        l2_error = np.sqrt(np.mean(result[interior_mask] ** 2))

        assert l2_error < 1e-2

    def test_divergence_3d(self):
        mesh = get_unit_3d_mesh_for_testing(h=0.1)
        disc = pybamm.Discretisation(
            mesh, {"current collector": pybamm.ScikitFiniteElement3D()}
        )

        u = pybamm.Variable("u", domain="current collector")
        disc.set_variable_slices([u])

        x_nodes, y_nodes, z_nodes = mesh["current collector"].nodes.T
        u_exact = x_nodes**2 + y_nodes**2 + z_nodes**2

        disc.bcs = {
            u: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (
                    pybamm.Scalar(1 + y_nodes[0] ** 2 + z_nodes[0] ** 2),
                    "Dirichlet",
                ),
                "front": (pybamm.Scalar(0), "Dirichlet"),
                "back": (
                    pybamm.Scalar(x_nodes[0] ** 2 + 1 + z_nodes[0] ** 2),
                    "Dirichlet",
                ),
                "bottom": (pybamm.Scalar(0), "Dirichlet"),
                "top": (
                    pybamm.Scalar(x_nodes[0] ** 2 + y_nodes[0] ** 2 + 1),
                    "Dirichlet",
                ),
            }
        }

        grad_u = pybamm.grad(u)
        div_grad_u = pybamm.div(grad_u)

        div_disc = disc.process_symbol(div_grad_u)
        div_result = div_disc.evaluate(None, u_exact)

        np.testing.assert_allclose(
            np.mean(div_result),
            6,
            rtol=0.2,
            atol=0.5,
            err_msg="Divergence of gradient should be 6 with Dirichlet BCs",
        )

    def test_neumann_boundary_conditions(self):
        mesh = get_unit_3d_mesh_for_testing(h=0.5)
        spatial_method = pybamm.ScikitFiniteElement3D()
        spatial_method.build(mesh)

        var = pybamm.Variable("var", domain="current collector")
        bcs = {var: {"right": (pybamm.Scalar(5.0), "Neumann")}}

        boundary_load_disc = spatial_method.laplacian_boundary_load(var, bcs)
        boundary_load_eval = boundary_load_disc.evaluate()

        right_dofs = mesh["current collector"].right_dofs
        assert np.sum(boundary_load_eval) > 0
        assert np.count_nonzero(boundary_load_eval[right_dofs]) > 0

        non_boundary_load = np.delete(boundary_load_eval, right_dofs)
        assert np.count_nonzero(np.abs(non_boundary_load) > 1e-12) == 0

    def test_3d_thermal_equation(self):
        mesh = get_unit_3d_mesh_for_testing(h=0.3)  # Slightly coarser mesh
        disc = pybamm.Discretisation(
            mesh, {"current collector": pybamm.ScikitFiniteElement3D()}
        )
        T = pybamm.Variable("T", domain="current collector")

        model = pybamm.BaseModel()

        model.algebraic = {T: pybamm.laplacian(T) + pybamm.source(1, T)}

        model.boundary_conditions = {
            T: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Dirichlet"),
                "front": (pybamm.Scalar(0), "Neumann"),
                "back": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
            }
        }

        model.initial_conditions = {T: pybamm.Scalar(0)}
        model.variables = {"T": T}

        disc.process_model(model)

        solver = pybamm.AlgebraicSolver()
        solution = solver.solve(model)

        T_solution = solution.y.flatten()

        assert T_solution.shape[0] == mesh["current collector"].npts
        assert np.mean(T_solution) > 0, (
            "Solution should be positive due to heat source ∇²T = -1"
        )
        assert np.max(np.abs(T_solution)) > 1e-6, "Solution should not be all zeros"
        assert np.std(T_solution) > 0, "Solution should vary spatially"

        left_dofs = mesh["current collector"].left_dofs
        right_dofs = mesh["current collector"].right_dofs

        np.testing.assert_allclose(
            T_solution[left_dofs], 0, atol=1e-6, err_msg="Left boundary should be zero"
        )
        np.testing.assert_allclose(
            T_solution[right_dofs],
            0,
            atol=1e-6,
            err_msg="Right boundary should be zero",
        )

    def test_gradient_squared_3d(self):
        mesh = get_unit_3d_mesh_for_testing(
            xpts=6, ypts=6, zpts=6, geom_type="box", include_particles=False
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
        mesh = get_unit_3d_mesh_for_testing(h=0.1)
        disc = pybamm.Discretisation(
            mesh, {"current collector": pybamm.ScikitFiniteElement3D()}
        )

        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])
        var_disc = disc.process_symbol(var)
        x_vertices, y_vertices, z_vertices = mesh["current collector"].nodes.T
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
                "front": (pybamm.Scalar(0), "Dirichlet"),
                "back": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        disc.set_variable_slices([var1])
        eqn1_disc = disc.process_symbol(eqn1)
        u1 = np.sin(np.pi * z_vertices)
        mass1 = pybamm.Mass(var1)
        mass1_disc = disc.process_symbol(mass1)
        soln1 = -(np.pi**2) * u1

        np.testing.assert_allclose(
            eqn1_disc.evaluate(None, u1).flatten(),
            mass1_disc.entries @ soln1,
            rtol=1e-4,
            atol=1e-3,
        )

        var2 = pybamm.Variable("var2", domain="current collector")
        eqn2 = pybamm.laplacian(var2)
        disc.bcs = {
            var2: {
                "bottom": (pybamm.Scalar(0), "Dirichlet"),
                "top": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        disc.set_variable_slices([var2])
        eqn2_disc = disc.process_symbol(eqn2)
        u2 = (
            np.cos(np.pi * x_vertices)
            * np.sin(np.pi * y_vertices)
            * np.cos(np.pi * z_vertices)
        )
        mass2 = pybamm.Mass(var2)
        mass2_disc = disc.process_symbol(mass2)
        soln2 = -3 * (np.pi**2) * u2
        np.testing.assert_allclose(
            eqn2_disc.evaluate(None, u2).flatten(),
            mass2_disc.entries @ soln2,
            rtol=1e-2,
            atol=1e-1,
        )

        var3 = pybamm.Variable("var3", domain="current collector")
        eqn3 = pybamm.laplacian(var3)
        all_boundaries = ["left", "right", "bottom", "top", "front", "back"]
        disc.bcs = {
            var3: {name: (pybamm.Scalar(0), "Dirichlet") for name in all_boundaries}
        }
        disc.set_variable_slices([var3])
        eqn3_disc = disc.process_symbol(eqn3)
        u3 = (
            np.sin(np.pi * x_vertices)
            * np.sin(np.pi * y_vertices)
            * np.sin(np.pi * z_vertices)
        )
        mass3 = pybamm.Mass(var3)
        mass3_disc = disc.process_symbol(mass3)
        soln3 = -3 * (np.pi**2) * u3
        np.testing.assert_allclose(
            eqn3_disc.evaluate(None, u3).flatten(),
            mass3_disc.entries @ soln3,
            rtol=1e-2,
            atol=1e-1,
        )

    def test_definite_integral_3d(self):
        mesh = get_3d_mesh_for_testing(
            xpts=6, ypts=6, zpts=6, geom_type="box", include_particles=False
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
            xpts=6, ypts=6, zpts=6, geom_type="box", include_particles=False
        )
        spatial_methods = {
            "negative electrode": pybamm.ScikitFiniteElement3D(),
            "separator": pybamm.ScikitFiniteElement3D(),
            "positive electrode": pybamm.ScikitFiniteElement3D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="negative electrode")
        disc.set_variable_slices([var])
        boundary_faces = ["left", "right", "front", "back", "bottom", "top"]
        for face in boundary_faces:
            boundary_val = pybamm.BoundaryValue(var, face)
            boundary_val_disc = disc.process_symbol(boundary_val)
            constant_y = np.ones(mesh["negative electrode"].npts)[:, np.newaxis]
            result = boundary_val_disc.evaluate(None, constant_y)
            np.testing.assert_allclose(result, 1, rtol=1e-6, atol=1e-6)

    def test_boundary_integral_3d(self):
        mesh = get_3d_mesh_for_testing(geom_type="box")
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
        np.testing.assert_allclose(numerical_result, expected_result, rtol=5e-2)

    def test_cylinder_mesh_properties(self):
        mesh = get_3d_mesh_for_testing(
            xpts=6,
            ypts=6,
            zpts=6,
            geom_type="cylinder",
            include_particles=False,
            radius=0.4,
            height=0.8,
        )
        nodes = mesh["negative electrode"].nodes
        radii = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
        assert radii.max() <= 0.42
        assert nodes[:, 2].min() >= -0.1
        assert nodes[:, 2].max() <= 0.9

    def test_disc_spatial_var_3d(self):
        mesh = get_unit_3d_mesh_for_testing(
            xpts=4, ypts=4, zpts=4, geom_type="box", include_particles=False
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
        # Solves grad^2 u = 1, with du/dz = 1 at z=1 (back), du/dz = 0 at z=0 (front),
        # and du/dn = 0 on other faces (x-sides, y-sides).
        # The solution is constrained to have zero average value.
        # Exact solution: u(z) = z^2 / 2 - 1/6

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
        model.boundary_conditions = {
            u: {
                "front": (pybamm.Scalar(0), "Neumann"),  # du/dz = 0 at z=0
                "back": (pybamm.Scalar(1), "Neumann"),  # du/dz = 1 at z=1
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
            }
        }
        model.variables = {"c": c, "u": u}

        mesh = get_unit_3d_mesh_for_testing(h=0.2)
        spatial_methods = {"current collector": pybamm.ScikitFiniteElement3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        solver = pybamm.AlgebraicSolver()
        solution = solver.solve(model)

        u_numerical = solution.y[:-1]

        z_coords = mesh["current collector"].nodes[:, 2]
        u_exact_at_nodes = (z_coords**2 / 2) - (1 / 6)

        np.testing.assert_allclose(
            u_numerical.flatten(), u_exact_at_nodes.flatten(), rtol=1e-2, atol=1e-1
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
                "left": (pybamm.Scalar(k_val), "Dirichlet"),
                "right": (pybamm.Scalar(a_val + b_val + k_val), "Dirichlet"),
                "front": (pybamm.Scalar(0), "Neumann"),
                "back": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        model.initial_conditions = {u: pybamm.Scalar(1)}
        model.variables = {"u": u}

        mesh = get_unit_3d_mesh_for_testing(h=0.15)
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

    def test_laplacian_manufactured_solution_on_cylinder_polynomial(self):
        radius = 0.5
        height = 1.0
        mesh = get_unit_3d_mesh_for_testing(
            geom_type="cylinder", radius=radius, height=height, h=0.2
        )

        model = pybamm.BaseModel()
        u = pybamm.Variable("u", "current collector")

        x = pybamm.SpatialVariable("x", "current collector")
        y = pybamm.SpatialVariable("y", "current collector")
        z = pybamm.SpatialVariable("z", "current collector")

        u_exact = x * y * z * (1 - z)

        f = -(-2 * x * y)

        model.algebraic = {u: pybamm.laplacian(u) - pybamm.source(f, u)}

        model.boundary_conditions = {
            u: {
                "bottom cap": (pybamm.Scalar(0), "Dirichlet"),
                "top cap": (pybamm.Scalar(0), "Dirichlet"),
                "side wall": (u_exact, "Dirichlet"),
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
        x_nodes = nodes[:, 0]
        y_nodes = nodes[:, 1]
        z_nodes = nodes[:, 2]

        u_analytical = x_nodes * y_nodes * z_nodes * (1 - z_nodes)

        np.testing.assert_allclose(u_numerical, u_analytical, rtol=1e-2, atol=1e-2)

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
