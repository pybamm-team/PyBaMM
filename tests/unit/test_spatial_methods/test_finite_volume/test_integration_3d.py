from tests import get_mesh_for_testing_3d
import pybamm
import numpy as np


class TestIntegration3D:
    def test_definite_integral_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume3D(),
            "separator": pybamm.FiniteVolume3D(),
            "positive electrode": pybamm.FiniteVolume3D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        Lx_n = mesh["negative electrode"].edges_x[-1]
        Lx_s = mesh["separator"].edges_x[-1] - Lx_n
        Lx_p = mesh["positive electrode"].edges_x[-1] - (Lx_n + Lx_s)
        total_Lx = Lx_n + Lx_s + Lx_p

        Ly = mesh["negative electrode"].edges_y[-1]

        Lz = mesh["negative electrode"].edges_z[-1]

        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable(
            "x", ["negative electrode", "separator"], direction="x"
        )
        integral_eqn = pybamm.Integral(var, x)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        submesh = mesh[("negative electrode", "separator")]

        n_tot = submesh.npts_x * submesh.npts_y * submesh.npts_z
        constant_ones = np.ones(n_tot)

        result = integral_eqn_disc.evaluate(None, constant_ones).flatten()
        expected = Lx_n + Lx_s
        assert (result == expected).all()

        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        y = pybamm.SpatialVariable(
            "y", ["negative electrode", "separator"], direction="y"
        )
        integral_eqn = pybamm.Integral(var, y)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        # Reuse the same submesh and constant vector
        result = integral_eqn_disc.evaluate(None, constant_ones).flatten()
        expected = Ly
        assert (result == expected).all()

        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        z = pybamm.SpatialVariable(
            "z", ["negative electrode", "separator"], direction="z"
        )
        integral_eqn = pybamm.Integral(var, z)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        result = integral_eqn_disc.evaluate(None, constant_ones).flatten()
        # Integral over z = Lz, length = n_x * n_y
        expected = Lz
        assert (result == expected).all()

        var = pybamm.Variable(
            "var",
            domain=[
                "negative electrode",
                "separator",
                "positive electrode",
            ],
        )
        x = pybamm.SpatialVariable(
            "x",
            ["negative electrode", "separator", "positive electrode"],
            direction="x",
        )
        integral_eqn = pybamm.Integral(var, x)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        submesh_full = mesh[("negative electrode", "separator", "positive electrode")]
        n_x = submesh_full.npts_x
        n_y = submesh_full.npts_y
        n_z = submesh_full.npts_z

        x_nodes_1d = submesh_full.nodes_x

        x_coords = np.zeros(n_x * n_y * n_z)

        for k in range(n_z):
            for j in range(n_y):
                for i in range(n_x):
                    idx = k * n_x * n_y + j * n_x + i
                    x_coords[idx] = x_nodes_1d[i]  # Use the actual x-coordinate

        manual_result = []
        for k in range(submesh_full.npts_z):
            for j in range(submesh_full.npts_y):
                start_idx = (
                    k * submesh_full.npts_x * submesh_full.npts_y
                    + j * submesh_full.npts_x
                )
                end_idx = start_idx + submesh_full.npts_x
                x_slice = x_coords[start_idx:end_idx]
                dx_slice = submesh_full.d_edges_x

                integral_value = np.sum(x_slice * dx_slice)
                manual_result.append(integral_value)

        result = integral_eqn_disc.evaluate(None, x_coords).flatten()
        n_y = submesh_full.npts_y
        n_z = submesh_full.npts_z
        expected = np.ones(n_y * n_z) * (total_Lx**2 / 2)
        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-6)

    def test_indefinite_integral_3d(self):
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume3D(),
            "separator": pybamm.FiniteVolume3D(),
            "positive electrode": pybamm.FiniteVolume3D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        submesh = mesh["negative electrode"]
        n_x = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        var = pybamm.Variable("var", domain=["negative electrode"])
        x = pybamm.SpatialVariable("x", ["negative electrode"], direction="x")

        indefinite_integral_eqn = pybamm.IndefiniteIntegral(var, x)
        disc.set_variable_slices([var])
        indefinite_integral_disc = disc.process_symbol(indefinite_integral_eqn)

        constant_ones = np.ones(n_x * n_y * n_z)
        result = indefinite_integral_disc.evaluate(None, constant_ones)

        expected_size = (n_x + 1) * n_y * n_z
        assert len(result) == expected_size, (
            f"Expected output size {expected_size}, got {len(result)}"
        )

        result_3d = result.reshape((n_z, n_y, n_x + 1))

        for k in range(n_z):
            for j in range(n_y):
                x_slice = result_3d[k, j, :]
                if n_x > 1:  # Only check if we have multiple x points
                    x_diffs = np.diff(x_slice)
                    assert np.all(x_diffs >= 0), (
                        f"Indefinite integral should increase along x, got diffs: {x_diffs}"
                    )

        var = pybamm.Variable("var", domain=["negative electrode"])
        x = pybamm.SpatialVariable("x", ["negative electrode"], direction="x")
        indefinite_integral_eqn = pybamm.IndefiniteIntegral(var, x)
        disc.set_variable_slices([var])
        indefinite_integral_disc = disc.process_symbol(indefinite_integral_eqn)

        x_coords = np.zeros(n_x * n_y * n_z)
        x_nodes_1d = submesh.nodes_x

        for k in range(n_z):
            for j in range(n_y):
                for i in range(n_x):
                    idx = k * n_x * n_y + j * n_x + i
                    x_coords[idx] = x_nodes_1d[i]

        result = indefinite_integral_disc.evaluate(None, x_coords)
        result_3d = result.reshape((n_z, n_y, n_x + 1))

        if n_x > 1:
            for k in range(n_z):
                for j in range(n_y):
                    integral_slice = result_3d[k, j, :]
                    x_edges = np.zeros(n_x + 1)
                    x_edges[0] = submesh.edges_x[0]  # Left boundary
                    x_edges[1:] = submesh.edges_x[1:]  # Right boundaries

                    if len(integral_slice) > 2:
                        first_diffs = np.diff(integral_slice)
                        assert np.all(first_diffs[1:] >= first_diffs[:-1]), (
                            "Integral of x should be increasing at increasing rate (x^2/2)"
                        )

        var = pybamm.Variable("var", domain=["negative electrode"])
        x = pybamm.SpatialVariable("x", ["negative electrode"], direction="x")

        indefinite_integral_eqn = pybamm.IndefiniteIntegral(var, x)
        disc.set_variable_slices([var])
        indefinite_integral_disc = disc.process_symbol(indefinite_integral_eqn)

        x_squared = np.zeros(n_x * n_y * n_z)
        for k in range(n_z):
            for j in range(n_y):
                for i in range(n_x):
                    idx = k * n_x * n_y + j * n_x + i
                    x_squared[idx] = x_nodes_1d[i] ** 2

        result = indefinite_integral_disc.evaluate(None, x_squared)
        result_3d = result.reshape((n_z, n_y, n_x + 1))

        x_coords_3d = x_coords.reshape((n_z, n_y, n_x))

        integral_slice = result_3d[0, 0, :]
        x_slice = x_coords_3d[0, 0, :]

        edge_positions = submesh.edges_x
        expected_integral_at_edges = np.zeros(n_x + 1)
        for i in range(1, n_x + 1):
            x_left = edge_positions[0]
            x_right = edge_positions[i]
            expected_integral_at_edges[i] = (x_right**3 - x_left**3) / 3

        if len(integral_slice) > 1:
            numerical_derivative = np.diff(integral_slice) / np.diff(edge_positions)
            midpoints = (edge_positions[1:] + edge_positions[:-1]) / 2
            expected_derivative = midpoints**2

            print(f"Numerical derivative: {numerical_derivative}")
            print(f"Expected derivative (x² at midpoints): {expected_derivative}")
            print(f"Difference: {numerical_derivative - expected_derivative}")

            np.testing.assert_allclose(
                numerical_derivative,
                expected_derivative,
                rtol=1e-7,
                atol=1e-6,
                err_msg="Derivative of ∫x² dx should approximate x²",
            )
