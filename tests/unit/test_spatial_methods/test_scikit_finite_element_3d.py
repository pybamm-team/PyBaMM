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

    def test_discretise_equations_cylinder(self):
        try:
            mesh = get_3d_mesh_for_testing(
                geom_type="cylinder",
                include_particles=False,
                radius=0.5,
                height=1.0,
                h=0.15,  # Use a reasonably fine mesh for accuracy
            )
        except pybamm.DiscretisationError as e:
            pytest.skip(f"Cylinder mesh generation failed: {e}")

        spatial_methods = {"negative electrode": pybamm.ScikitFiniteElement3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="negative electrode")

        source_term = pybamm.Scalar(4)
        eqn = pybamm.laplacian(var) - source_term

        x_sym = pybamm.SpatialVariable("x", "negative electrode")
        y_sym = pybamm.SpatialVariable("y", "negative electrode")
        u_analytical_sym = x_sym**2 + y_sym**2

        disc.bcs = {
            var: {
                name: (u_analytical_sym, "Dirichlet")
                for name in ["side wall", "top cap", "bottom cap"]
            }
        }

        disc.set_variable_slices([var])
        eqn_disc = disc.process_symbol(eqn)

        x_num, y_num, _ = mesh["negative electrode"].nodes.T
        u_analytical_num = x_num**2 + y_num**2

        result = eqn_disc.evaluate(None, u_analytical_num)

        l2_error = np.sqrt(np.mean(result**2))
        assert l2_error < 1e-2

    def test_discretise_equations_spiral(self):
        try:
            mesh = get_3d_mesh_for_testing(
                geom_type="spiral",
                include_particles=False,
                inner_radius=0.1,
                outer_radius=0.4,
                height=1.0,
                turns=1.0,  # Fewer turns
                h=0.15,  # Coarser mesh
            )
        except pybamm.DiscretisationError as e:
            pytest.skip(f"Spiral mesh generation failed: {e}")

        spatial_methods = {"negative electrode": pybamm.ScikitFiniteElement3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="negative electrode")

        eqn = pybamm.laplacian(var)
        disc.bcs = {}
        disc.set_variable_slices([var])

        eqn_disc = disc.process_symbol(eqn)
        test_func = np.ones(mesh["negative electrode"].npts)
        result = eqn_disc.evaluate(None, test_func)

        assert result is not None
        assert not np.all(np.isnan(result))

    def test_gradient_3d_manufactured(self):
        mesh = get_unit_3d_mesh_for_testing(h=0.3)
        disc = pybamm.Discretisation(
            mesh, {"negative electrode": pybamm.ScikitFiniteElement3D()}
        )
        var = pybamm.Variable("var", domain="negative electrode")
        disc.set_variable_slices([var])

        grad_disc = disc.process_symbol(pybamm.grad(var))
        x, y, z = mesh["negative electrode"].nodes.T
        test_func = 2 * x + 3 * y + 4 * z

        grad_eval = grad_disc.evaluate(None, test_func)

        np.testing.assert_allclose(np.mean(grad_eval[:, 0]), 2, rtol=5e-2, atol=5e-2)
        np.testing.assert_allclose(np.mean(grad_eval[:, 1]), 3, rtol=5e-2, atol=5e-2)
        np.testing.assert_allclose(np.mean(grad_eval[:, 2]), 4, rtol=5e-2, atol=5e-2)

    def test_laplacian_3d_manufactured_box(self):
        mesh = get_unit_3d_mesh_for_testing(h=0.4)
        disc = pybamm.Discretisation(
            mesh, {"negative electrode": pybamm.ScikitFiniteElement3D()}
        )
        var = pybamm.Variable("var", domain="negative electrode")

        eqn = pybamm.laplacian(var)

        x_sym = pybamm.SpatialVariable("x", "negative electrode")
        y_sym = pybamm.SpatialVariable("y", "negative electrode")
        z_sym = pybamm.SpatialVariable("z", "negative electrode")
        u_analytical_sym = x_sym * y_sym * z_sym

        all_boundaries = ["left", "right", "bottom", "top", "front", "back"]
        disc.bcs = {
            var: {name: (u_analytical_sym, "Dirichlet") for name in all_boundaries}
        }
        disc.set_variable_slices([var])
        eqn_disc = disc.process_symbol(eqn)

        x_num, y_num, z_num = mesh["negative electrode"].nodes.T
        u_analytical_num = x_num * y_num * z_num

        result = eqn_disc.evaluate(None, u_analytical_num)

        submesh = mesh["negative electrode"]
        boundary_dofs = np.unique(
            np.concatenate(
                [getattr(submesh, f"{name}_dofs") for name in all_boundaries]
            )
        )
        interior_mask = np.ones(submesh.npts, dtype=bool)
        interior_mask[boundary_dofs] = False

        l2_error = np.sqrt(np.mean(result[interior_mask] ** 2))
        assert l2_error < 1e-10

    def test_laplacian_3d_manufactured_cylinder(self):
        try:
            mesh = get_unit_3d_mesh_for_testing(
                geom_type="cylinder", radius=0.5, height=1, h=0.2
            )
        except pybamm.DiscretisationError as e:
            pytest.skip(f"Cylinder mesh generation failed: {e}")

        disc = pybamm.Discretisation(
            mesh, {"negative electrode": pybamm.ScikitFiniteElement3D()}
        )
        var = pybamm.Variable("var", domain="negative electrode")

        source_term = pybamm.Scalar(4)
        eqn = pybamm.laplacian(var) - source_term

        x_sym = pybamm.SpatialVariable("x", "negative electrode")
        y_sym = pybamm.SpatialVariable("y", "negative electrode")
        u_analytical_sym = x_sym**2 + y_sym**2

        all_boundaries = ["side wall", "top cap", "bottom cap"]
        disc.bcs = {
            var: {name: (u_analytical_sym, "Dirichlet") for name in all_boundaries}
        }
        disc.set_variable_slices([var])
        eqn_disc = disc.process_symbol(eqn)

        x_num, y_num, _ = mesh["negative electrode"].nodes.T
        u_analytical_num = x_num**2 + y_num**2

        result = eqn_disc.evaluate(None, u_analytical_num)

        submesh = mesh["negative electrode"]
        boundary_dofs = np.unique(
            np.concatenate(
                [getattr(submesh, f"{name}_dofs") for name in all_boundaries]
            )
        )
        interior_mask = np.ones(submesh.npts, dtype=bool)
        interior_mask[boundary_dofs] = False

        l2_error = np.sqrt(np.mean(result[interior_mask] ** 2))
        assert l2_error < 1e-1

    def test_divergence_3d_manufactured(self):
        mesh = get_unit_3d_mesh_for_testing(h=0.4)
        disc = pybamm.Discretisation(
            mesh, {"negative electrode": pybamm.ScikitFiniteElement3D()}
        )

        u = pybamm.Variable("u", domain="negative electrode")
        eqn = pybamm.div(pybamm.grad(u))

        x_sym = pybamm.SpatialVariable("x", "negative electrode")
        u_analytical_sym = 4.5 * x_sym**2

        disc.bcs = {
            u: {
                "left": (u_analytical_sym, "Dirichlet"),
                "right": (u_analytical_sym, "Dirichlet"),
            }
        }
        disc.set_variable_slices([u])
        div_disc = disc.process_symbol(eqn)

        x, _, _ = mesh["negative electrode"].nodes.T
        test_func = 4.5 * x**2

        result = div_disc.evaluate(None, test_func)

        np.testing.assert_allclose(np.mean(result), 9, rtol=1e-1, atol=1e-1)

    def test_neumann_boundary_conditions(self):
        mesh = get_unit_3d_mesh_for_testing(h=0.5)
        spatial_method = pybamm.ScikitFiniteElement3D()
        spatial_method.build(mesh)

        var = pybamm.Variable("var", domain="negative electrode")
        bcs = {var: {"right": (pybamm.Scalar(5.0), "Neumann")}}

        boundary_load_disc = spatial_method.laplacian_boundary_load(var, bcs)
        boundary_load_eval = boundary_load_disc.evaluate()

        right_dofs = mesh["negative electrode"].right_dofs
        assert np.sum(boundary_load_eval) > 0
        assert np.count_nonzero(boundary_load_eval[right_dofs]) > 0

        non_boundary_load = np.delete(boundary_load_eval, right_dofs)
        assert np.count_nonzero(np.abs(non_boundary_load) > 1e-12) == 0

    def test_3d_thermal_equation(self):
        mesh = get_unit_3d_mesh_for_testing(h=0.5)
        disc = pybamm.Discretisation(
            mesh, {"negative electrode": pybamm.ScikitFiniteElement3D()}
        )
        T = pybamm.Variable("T", domain="negative electrode")

        model = pybamm.BaseModel()
        model.rhs = {T: pybamm.laplacian(T) + 1}
        model.boundary_conditions = {
            T: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        model.initial_conditions = {T: pybamm.Scalar(0)}
        model.variables = {"T": T}

        disc.process_model(model)
        solver = pybamm.ScipySolver()
        solution = solver.solve(model, t_eval=np.linspace(0, 0.1, 10))

        assert solution.t.size == 10
        assert solution.y.shape[0] == mesh["negative electrode"].npts
        assert solution.y.shape[1] == 10

        T_final_time = solution.y[:, -1]

        assert T_final_time.shape[0] == mesh["negative electrode"].npts

        assert np.mean(T_final_time) > 0, (
            "Solution should be positive due to source term"
        )
        assert np.max(np.abs(T_final_time)) > 1e-6, "Solution should not be all zeros"
        assert np.std(T_final_time) > 0, "Solution should vary spatially"

    def test_gradient_squared_3d(self):
        mesh = get_unit_3d_mesh_for_testing(
            xpts=6, ypts=6, zpts=6, geom_type="box", include_particles=False
        )
        spatial_methods = {
            "negative electrode": pybamm.ScikitFiniteElement3D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="negative electrode")
        disc.set_variable_slices([var])
        x = mesh["negative electrode"].nodes[:, 0]
        y = mesh["negative electrode"].nodes[:, 1]
        z = mesh["negative electrode"].nodes[:, 2]
        eqn = pybamm.grad_squared(var)
        eqn_disc = disc.process_symbol(eqn)
        test_function = x**2 + y**2 + z**2
        ans = eqn_disc.evaluate(None, test_function)
        np.testing.assert_array_less(0, ans)

    def test_manufactured_solution_3d(self):
        mesh = get_unit_3d_mesh_for_testing(
            xpts=8, ypts=8, zpts=8, geom_type="box", include_particles=False
        )
        spatial_methods = {
            "negative electrode": pybamm.ScikitFiniteElement3D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="negative electrode")
        disc.set_variable_slices([var])
        var_disc = disc.process_symbol(var)
        x_nodes = mesh["negative electrode"].nodes[:, 0]
        y_nodes = mesh["negative electrode"].nodes[:, 1]
        z_nodes = mesh["negative electrode"].nodes[:, 2]
        np.testing.assert_allclose(
            var_disc.evaluate(None, x_nodes),
            x_nodes[:, np.newaxis],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            var_disc.evaluate(None, 2 * y_nodes),
            2 * y_nodes[:, np.newaxis],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            var_disc.evaluate(None, 3 * z_nodes),
            3 * z_nodes[:, np.newaxis],
            rtol=1e-6,
            atol=1e-6,
        )
        mixed_func = x_nodes * y_nodes * z_nodes
        np.testing.assert_allclose(
            var_disc.evaluate(None, mixed_func),
            mixed_func[:, np.newaxis],
            rtol=1e-6,
            atol=1e-6,
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

    def test_spiral_mesh_properties(self):
        mesh = get_3d_mesh_for_testing(
            xpts=6,
            ypts=6,
            zpts=6,
            geom_type="spiral",
            include_particles=False,
            inner_radius=0.1,
            outer_radius=0.4,
            height=0.8,
            turns=2,
        )
        nodes = mesh["negative electrode"].nodes
        radii = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
        assert radii.min() >= 0.08
        assert radii.max() <= 0.42
        assert nodes[:, 2].min() >= -0.1
        assert nodes[:, 2].max() <= 0.9

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
            "negative electrode": pybamm.ScikitFiniteElement3D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        x = pybamm.SpatialVariable("x", ["negative electrode"])
        y = pybamm.SpatialVariable("y", ["negative electrode"])
        z = pybamm.SpatialVariable("z", ["negative electrode"])
        x_disc = disc.process_symbol(x)
        y_disc = disc.process_symbol(y)
        z_disc = disc.process_symbol(z)
        x_actual = mesh["negative electrode"].nodes[:, 0][:, np.newaxis]
        y_actual = mesh["negative electrode"].nodes[:, 1][:, np.newaxis]
        z_actual = mesh["negative electrode"].nodes[:, 2][:, np.newaxis]
        np.testing.assert_array_equal(x_disc.evaluate(), x_actual)
        np.testing.assert_array_equal(y_disc.evaluate(), y_actual)
        np.testing.assert_array_equal(z_disc.evaluate(), z_actual)
