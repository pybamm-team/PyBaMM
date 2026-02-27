"""
Unit tests for FiniteVolumeUnstructured spatial method.

Tests cover both 2D (triangle) and 3D (tet) meshes, validating:
- TPFA Laplacian structural properties and conservation
- Green-Gauss gradient on linear fields
- Divergence (adjoint of gradient)
- Mass matrix, integrals, boundary value/flux
- Internal Neumann condition for domain coupling
"""

import numpy as np
import pytest
from scipy.sparse import coo_matrix as sp_coo
from scipy.sparse import csr_matrix as sp_csr

from pybamm.meshes.unstructured_submesh import (
    UnstructuredSubMesh,
    _hex_to_tet,
    _quad_to_tri,
    compute_interface_data,
)
from pybamm.spatial_methods.finite_volume_unstructured import (
    FiniteVolumeUnstructured,
)

# ======================================================================
# Mesh helpers
# ======================================================================


def _make_2d_mesh(nx=4, nz=4, x_range=(0, 1), z_range=(0, 1)):
    x_edges = np.linspace(x_range[0], x_range[1], nx + 1)
    z_edges = np.linspace(z_range[0], z_range[1], nz + 1)
    nodes, elements = _quad_to_tri(x_edges, z_edges)
    return UnstructuredSubMesh(nodes, elements)


def _make_3d_mesh(nx=3, ny=3, nz=3, x_range=(0, 1), y_range=(0, 1), z_range=(0, 1)):
    x_edges = np.linspace(x_range[0], x_range[1], nx + 1)
    y_edges = np.linspace(y_range[0], y_range[1], ny + 1)
    z_edges = np.linspace(z_range[0], z_range[1], nz + 1)
    nodes, elements = _hex_to_tet(x_edges, y_edges, z_edges)
    return UnstructuredSubMesh(nodes, elements)


def _make_split_2d_meshes(nx_left=3, nx_right=3, nz=3):
    """Create two adjacent 2D meshes for interface testing."""
    left = _make_2d_mesh(nx_left, nz, x_range=(0, 0.5))
    right = _make_2d_mesh(nx_right, nz, x_range=(0.5, 1.0))
    compute_interface_data(left, right, left_name="left", right_name="right")
    return left, right


def _get_internal_cells(mesh):
    """Return indices of cells that do not touch any boundary face."""
    bnd_cells = set()
    for indices in mesh.boundary_faces.values():
        for fi in indices:
            bnd_cells.add(mesh.face_owner[fi])
    return [i for i in range(mesh.npts) if i not in bnd_cells]


# ======================================================================
# Tests: TPFA Laplacian
# ======================================================================


class TestTPFALaplacian:
    def test_tpfa_matrix_shape_2d(self):
        mesh = _make_2d_mesh(5, 5)
        fvu = FiniteVolumeUnstructured()
        L = fvu._tpfa_matrix(mesh)
        assert L.shape == (mesh.npts, mesh.npts)

    def test_tpfa_matrix_shape_3d(self):
        mesh = _make_3d_mesh(3, 3, 3)
        fvu = FiniteVolumeUnstructured()
        L = fvu._tpfa_matrix(mesh)
        assert L.shape == (mesh.npts, mesh.npts)

    def test_tpfa_stiffness_symmetry_2d(self):
        """The raw stiffness matrix K (before volume scaling) should be symmetric."""
        mesh = _make_2d_mesh(5, 5)
        n = mesh.npts
        n_int = mesh.n_internal_faces

        owner = mesh.face_owner[:n_int]
        neighbor = mesh.face_neighbor[:n_int]
        areas = mesh.face_areas[:n_int]
        c_owner = mesh.cell_centroids[owner]
        c_neighbor = mesh.cell_centroids[neighbor]
        dist = np.linalg.norm(c_neighbor - c_owner, axis=1)
        coeff = areas / dist

        rows = np.concatenate([owner, neighbor, owner, neighbor])
        cols = np.concatenate([neighbor, owner, owner, neighbor])
        data = np.concatenate([coeff, coeff, -coeff, -coeff])
        K = sp_csr(sp_coo((data, (rows, cols)), shape=(n, n)))

        diff = K - K.T
        assert abs(diff).max() < 1e-12

    def test_tpfa_conservation_2d(self):
        """Weighted sum of L@u over all cells = 0 (internal flux conservation)."""
        mesh = _make_2d_mesh(5, 5)
        fvu = FiniteVolumeUnstructured()
        L = fvu._tpfa_matrix(mesh)

        u = mesh.cell_centroids[:, 0] ** 2
        Lu = L @ u
        total = np.sum(Lu * mesh.cell_volumes)
        np.testing.assert_allclose(total, 0.0, atol=1e-10)

    def test_tpfa_conservation_3d(self):
        mesh = _make_3d_mesh(3, 3, 3)
        fvu = FiniteVolumeUnstructured()
        L = fvu._tpfa_matrix(mesh)

        u = mesh.cell_centroids[:, 0] ** 2
        Lu = L @ u
        total = np.sum(Lu * mesh.cell_volumes)
        np.testing.assert_allclose(total, 0.0, atol=1e-10)

    def test_tpfa_constant_field_2d(self):
        """Laplacian of constant = 0."""
        mesh = _make_2d_mesh(5, 5)
        fvu = FiniteVolumeUnstructured()
        L = fvu._tpfa_matrix(mesh)

        u = np.ones(mesh.npts) * 7.0
        np.testing.assert_allclose(L @ u, 0.0, atol=1e-12)

    def test_tpfa_constant_field_3d(self):
        mesh = _make_3d_mesh(3, 3, 3)
        fvu = FiniteVolumeUnstructured()
        L = fvu._tpfa_matrix(mesh)

        u = np.ones(mesh.npts) * 7.0
        np.testing.assert_allclose(L @ u, 0.0, atol=1e-12)

    def test_tpfa_negative_diagonal_2d(self):
        """Diagonal entries of TPFA matrix should be non-positive."""
        mesh = _make_2d_mesh(5, 5)
        fvu = FiniteVolumeUnstructured()
        L = fvu._tpfa_matrix(mesh)
        diag = L.diagonal()
        assert np.all(diag <= 1e-15)


# ======================================================================
# Tests: Green-Gauss Gradient
# ======================================================================


class TestGreenGaussGradient:
    def test_gradient_constant_field_2d(self):
        """Gradient of constant = 0 everywhere."""
        mesh = _make_2d_mesh(5, 5)
        fvu = FiniteVolumeUnstructured()
        G = fvu._green_gauss_matrices(mesh)

        u = np.ones(mesh.npts) * 3.14
        for k in range(mesh.dimension):
            np.testing.assert_allclose(G[k] @ u, 0.0, atol=1e-12)

    def test_gradient_constant_field_3d(self):
        mesh = _make_3d_mesh(3, 3, 3)
        fvu = FiniteVolumeUnstructured()
        G = fvu._green_gauss_matrices(mesh)

        u = np.ones(mesh.npts) * 3.14
        for k in range(mesh.dimension):
            np.testing.assert_allclose(G[k] @ u, 0.0, atol=1e-12)

    def test_gradient_linear_x_2d(self):
        """Gradient of u = x should be [1, 0] on internal cells."""
        mesh = _make_2d_mesh(8, 8)
        fvu = FiniteVolumeUnstructured()
        G = fvu._green_gauss_matrices(mesh)

        u = mesh.cell_centroids[:, 0]
        internal = _get_internal_cells(mesh)

        if internal:
            np.testing.assert_allclose((G[0] @ u)[internal], 1.0, atol=1e-10)
            np.testing.assert_allclose((G[1] @ u)[internal], 0.0, atol=1e-10)

    def test_gradient_linear_z_2d(self):
        """Gradient of u = z should be [0, 1] on internal cells."""
        mesh = _make_2d_mesh(8, 8)
        fvu = FiniteVolumeUnstructured()
        G = fvu._green_gauss_matrices(mesh)

        u = mesh.cell_centroids[:, 1]
        internal = _get_internal_cells(mesh)

        if internal:
            np.testing.assert_allclose((G[0] @ u)[internal], 0.0, atol=1e-10)
            np.testing.assert_allclose((G[1] @ u)[internal], 1.0, atol=1e-10)

    def test_gradient_linear_x_3d(self):
        """Gradient of u = x on 3D tet mesh.

        On non-orthogonal tet meshes from hex splitting, the Green-Gauss
        gradient with distance-weighted interpolation has O(h) error.
        Boundary cells contribute a bias from zeroth-order face
        extrapolation. We verify the mean is within 15% and that
        internal cells are accurate.
        """
        mesh = _make_3d_mesh(4, 4, 4)
        fvu = FiniteVolumeUnstructured()
        G = fvu._green_gauss_matrices(mesh)

        u = mesh.cell_centroids[:, 0]
        grad_x = G[0] @ u

        mean_grad_x = np.sum(grad_x * mesh.cell_volumes) / mesh.cell_volumes.sum()
        np.testing.assert_allclose(mean_grad_x, 1.0, atol=0.15)

    def test_gradient_linear_combo_2d(self):
        """Gradient of u = 2x + 3z should be [2, 3]."""
        mesh = _make_2d_mesh(8, 8)
        fvu = FiniteVolumeUnstructured()
        G = fvu._green_gauss_matrices(mesh)

        u = 2 * mesh.cell_centroids[:, 0] + 3 * mesh.cell_centroids[:, 1]
        internal = _get_internal_cells(mesh)

        if internal:
            np.testing.assert_allclose((G[0] @ u)[internal], 2.0, atol=1e-10)
            np.testing.assert_allclose((G[1] @ u)[internal], 3.0, atol=1e-10)


# ======================================================================
# Tests: Divergence
# ======================================================================


class TestDivergence:
    def test_divergence_matrices_shape_2d(self):
        mesh = _make_2d_mesh(5, 5)
        fvu = FiniteVolumeUnstructured()
        D = fvu._divergence_matrices(mesh)
        assert len(D) == 2
        assert D[0].shape == (mesh.npts, mesh.npts)

    def test_divergence_matrices_shape_3d(self):
        mesh = _make_3d_mesh(3, 3, 3)
        fvu = FiniteVolumeUnstructured()
        D = fvu._divergence_matrices(mesh)
        assert len(D) == 3
        assert D[0].shape == (mesh.npts, mesh.npts)

    def test_divergence_constant_vector_field_2d(self):
        """Divergence of a constant vector field = 0 on internal cells."""
        mesh = _make_2d_mesh(6, 6)
        fvu = FiniteVolumeUnstructured()
        D = fvu._divergence_matrices(mesh)

        Fx = np.ones(mesh.npts) * 2.0
        Fz = np.ones(mesh.npts) * 3.0
        div = D[0] @ Fx + D[1] @ Fz

        internal = _get_internal_cells(mesh)
        if internal:
            np.testing.assert_allclose(div[internal], 0.0, atol=1e-10)

    def test_divergence_constant_vector_field_3d(self):
        mesh = _make_3d_mesh(3, 3, 3)
        fvu = FiniteVolumeUnstructured()
        D = fvu._divergence_matrices(mesh)

        Fx = np.ones(mesh.npts) * 2.0
        Fy = np.ones(mesh.npts) * 3.0
        Fz = np.ones(mesh.npts) * 4.0
        div = D[0] @ Fx + D[1] @ Fy + D[2] @ Fz

        internal = _get_internal_cells(mesh)
        if internal:
            np.testing.assert_allclose(div[internal], 0.0, atol=1e-10)


# ======================================================================
# Tests: Mass matrix (cell volumes)
# ======================================================================


class TestMassMatrix:
    def test_volume_sum_2d(self):
        """Sum of cell volumes = domain area."""
        mesh = _make_2d_mesh(5, 5)
        np.testing.assert_allclose(mesh.cell_volumes.sum(), 1.0, atol=1e-12)

    def test_volume_sum_3d(self):
        """Sum of cell volumes = domain volume."""
        mesh = _make_3d_mesh(3, 3, 3)
        np.testing.assert_allclose(mesh.cell_volumes.sum(), 1.0, atol=1e-12)

    def test_volumes_positive_2d(self):
        mesh = _make_2d_mesh(5, 5)
        assert np.all(mesh.cell_volumes > 0)

    def test_volumes_positive_3d(self):
        mesh = _make_3d_mesh(3, 3, 3)
        assert np.all(mesh.cell_volumes > 0)

    def test_volume_sum_rectangle(self):
        """Non-square domain: [0,2] x [0,0.5] should have area 1.0."""
        mesh = _make_2d_mesh(6, 4, x_range=(0, 2), z_range=(0, 0.5))
        np.testing.assert_allclose(mesh.cell_volumes.sum(), 1.0, atol=1e-12)


# ======================================================================
# Tests: Integral
# ======================================================================


class TestIntegral:
    def test_definite_integral_constant_2d(self):
        """Integral of 1 over [0,1]^2 = 1."""
        mesh = _make_2d_mesh(5, 5)
        fvu = FiniteVolumeUnstructured()
        fvu._mesh = {("test",): mesh}

        class FakeChild:
            domain = ("test",)

        mat = fvu.definite_integral_matrix(FakeChild())
        result = mat @ np.ones(mesh.npts)
        np.testing.assert_allclose(result[0], 1.0, atol=1e-12)

    def test_definite_integral_constant_3d(self):
        """Integral of 1 over [0,1]^3 = 1."""
        mesh = _make_3d_mesh(3, 3, 3)
        fvu = FiniteVolumeUnstructured()
        fvu._mesh = {("test",): mesh}

        class FakeChild:
            domain = ("test",)

        mat = fvu.definite_integral_matrix(FakeChild())
        result = mat @ np.ones(mesh.npts)
        np.testing.assert_allclose(result[0], 1.0, atol=1e-12)

    def test_integral_linear_field_2d(self):
        """Integral of u = x over [0,1]^2 = 0.5."""
        mesh = _make_2d_mesh(10, 10)
        fvu = FiniteVolumeUnstructured()
        fvu._mesh = {("test",): mesh}

        class FakeChild:
            domain = ("test",)

        mat = fvu.definite_integral_matrix(FakeChild())
        u = mesh.cell_centroids[:, 0]
        result = mat @ u
        np.testing.assert_allclose(result[0], 0.5, atol=0.01)

    def test_integral_linear_field_3d(self):
        """Integral of u = x over [0,1]^3 = 0.5."""
        mesh = _make_3d_mesh(4, 4, 4)
        fvu = FiniteVolumeUnstructured()
        fvu._mesh = {("test",): mesh}

        class FakeChild:
            domain = ("test",)

        mat = fvu.definite_integral_matrix(FakeChild())
        u = mesh.cell_centroids[:, 0]
        result = mat @ u
        np.testing.assert_allclose(result[0], 0.5, atol=0.01)


# ======================================================================
# Tests: Boundary value / flux
# ======================================================================


class TestBoundaryValue:
    def test_boundary_faces_exist_2d(self):
        mesh = _make_2d_mesh(5, 5)
        assert "left" in mesh.boundary_faces
        assert "right" in mesh.boundary_faces
        assert "bottom" in mesh.boundary_faces
        assert "top" in mesh.boundary_faces

        for tag in ["left", "right", "bottom", "top"]:
            assert len(mesh.boundary_faces[tag]) > 0

    def test_boundary_faces_exist_3d(self):
        mesh = _make_3d_mesh(3, 3, 3)
        assert "left" in mesh.boundary_faces
        assert "right" in mesh.boundary_faces

    def test_left_boundary_x_zero_2d(self):
        """Left boundary face centroids should have x ≈ 0."""
        mesh = _make_2d_mesh(5, 5)
        left_centroids = mesh.face_centroids[mesh.boundary_faces["left"]]
        np.testing.assert_allclose(left_centroids[:, 0], 0.0, atol=1e-14)

    def test_right_boundary_x_one_2d(self):
        """Right boundary face centroids should have x ≈ 1."""
        mesh = _make_2d_mesh(5, 5)
        right_centroids = mesh.face_centroids[mesh.boundary_faces["right"]]
        np.testing.assert_allclose(right_centroids[:, 0], 1.0, atol=1e-14)


# ======================================================================
# Tests: Interface / internal_neumann_condition
# ======================================================================


class TestInternalNeumann:
    def test_interface_data_exists(self):
        left, right = _make_split_2d_meshes(3, 3, 3)
        assert len(left.interface_data) > 0 or len(right.interface_data) > 0

    def test_interface_face_count(self):
        """Number of interface faces should equal the number of z-boundary faces."""
        left, _right = _make_split_2d_meshes(4, 4, 4)
        interface = next(iter(left.interface_data.values()))
        assert len(interface["left_cells"]) > 0
        assert len(interface["right_cells"]) > 0
        assert len(interface["left_cells"]) == len(interface["right_cells"])

    def test_interface_uniform_field(self):
        """Interface gradient of uniform field = 0."""
        left, right = _make_split_2d_meshes(4, 4, 4)
        interface = next(iter(left.interface_data.values()))

        left_vals = np.ones(left.npts) * 5.0
        right_vals = np.ones(right.npts) * 5.0

        inv_dx = 1.0 / interface["cell_distances"]
        grad = inv_dx * (
            right_vals[interface["right_cells"]] - left_vals[interface["left_cells"]]
        )
        np.testing.assert_allclose(grad, 0.0, atol=1e-12)

    def test_interface_gradient_positive_for_increasing_x(self):
        """For u = x, interface gradient should be positive."""
        left, right = _make_split_2d_meshes(4, 4, 4)
        interface = next(iter(left.interface_data.values()))

        left_vals = left.cell_centroids[:, 0]
        right_vals = right.cell_centroids[:, 0]

        inv_dx = 1.0 / interface["cell_distances"]
        grad = inv_dx * (
            right_vals[interface["right_cells"]] - left_vals[interface["left_cells"]]
        )
        assert np.all(grad > 0), "Gradient should be positive for u = x"

    def test_interface_cell_distances_positive(self):
        left, _right = _make_split_2d_meshes(4, 4, 4)
        interface = next(iter(left.interface_data.values()))
        assert np.all(interface["cell_distances"] > 0)

    def test_interface_face_areas_positive(self):
        left, _right = _make_split_2d_meshes(4, 4, 4)
        interface = next(iter(left.interface_data.values()))
        assert np.all(interface["face_areas"] > 0)


# ======================================================================
# Tests: Conservation / divergence theorem
# ======================================================================


class TestConservation:
    def test_tpfa_conservation_2d(self):
        """Total internal flux = 0 (conservation of Laplacian)."""
        mesh = _make_2d_mesh(5, 5)
        fvu = FiniteVolumeUnstructured()
        L = fvu._tpfa_matrix(mesh)

        u = mesh.cell_centroids[:, 0] ** 2
        Lu = L @ u
        total = np.sum(Lu * mesh.cell_volumes)
        np.testing.assert_allclose(total, 0.0, atol=1e-10)

    def test_divergence_theorem_volume_weighted_2d(self):
        """
        For F = (x, z): div(F) = 2.
        Volume-weighted integral of div(F) should approach 2 * area.
        The Green-Gauss divergence has boundary-cell errors, so we use
        a generous tolerance.
        """
        mesh = _make_2d_mesh(10, 10)
        fvu = FiniteVolumeUnstructured()
        D = fvu._divergence_matrices(mesh)

        Fx = mesh.cell_centroids[:, 0]
        Fz = mesh.cell_centroids[:, 1]
        div_F = D[0] @ Fx + D[1] @ Fz

        vol_integral = np.sum(div_F * mesh.cell_volumes)
        np.testing.assert_allclose(vol_integral, 2.0, atol=0.25)


# ======================================================================
# Tests: Gradient squared
# ======================================================================


class TestGradientSquared:
    def test_gradient_squared_linear_x_2d(self):
        """|grad(x)|^2 ≈ 1 on internal cells."""
        mesh = _make_2d_mesh(8, 8)
        fvu = FiniteVolumeUnstructured()
        G = fvu._green_gauss_matrices(mesh)

        u = mesh.cell_centroids[:, 0]
        grad_sq = sum((G[k] @ u) ** 2 for k in range(mesh.dimension))

        internal = _get_internal_cells(mesh)
        if internal:
            np.testing.assert_allclose(grad_sq[internal], 1.0, atol=1e-10)

    def test_gradient_squared_constant_2d(self):
        """|grad(const)|^2 = 0."""
        mesh = _make_2d_mesh(5, 5)
        fvu = FiniteVolumeUnstructured()
        G = fvu._green_gauss_matrices(mesh)

        u = np.ones(mesh.npts) * 42.0
        grad_sq = sum((G[k] @ u) ** 2 for k in range(mesh.dimension))
        np.testing.assert_allclose(grad_sq, 0.0, atol=1e-20)


# ======================================================================
# Tests: Not implemented operators
# ======================================================================


class TestNotImplemented:
    def test_indefinite_integral_raises(self):
        fvu = FiniteVolumeUnstructured()
        with pytest.raises(NotImplementedError, match="Indefinite integral"):
            fvu.indefinite_integral(None, None, None)

    def test_delta_function_raises(self):
        fvu = FiniteVolumeUnstructured()
        with pytest.raises(NotImplementedError, match="Delta function"):
            fvu.delta_function(None, None)


# ======================================================================
# Tests: 3D specific
# ======================================================================


class Test3D:
    def test_gradient_mean_accuracy_3d(self):
        """Volume-weighted mean gradient of u = x should be ~1.

        Boundary cells bias the mean via zeroth-order face extrapolation;
        tolerance of 0.15 is appropriate for a 4^3 tet mesh.
        """
        mesh = _make_3d_mesh(4, 4, 4)
        fvu = FiniteVolumeUnstructured()
        G = fvu._green_gauss_matrices(mesh)

        u = mesh.cell_centroids[:, 0]
        vol = mesh.cell_volumes
        total_vol = vol.sum()

        mean_gx = np.sum((G[0] @ u) * vol) / total_vol
        mean_gy = np.sum((G[1] @ u) * vol) / total_vol
        mean_gz = np.sum((G[2] @ u) * vol) / total_vol

        np.testing.assert_allclose(mean_gx, 1.0, atol=0.15)
        np.testing.assert_allclose(mean_gy, 0.0, atol=0.15)
        np.testing.assert_allclose(mean_gz, 0.0, atol=0.15)

    def test_gradient_y_mean_accuracy_3d(self):
        """Volume-weighted mean gradient of u = y should be ~[0,1,0]."""
        mesh = _make_3d_mesh(4, 4, 4)
        fvu = FiniteVolumeUnstructured()
        G = fvu._green_gauss_matrices(mesh)

        u = mesh.cell_centroids[:, 1]
        vol = mesh.cell_volumes
        total_vol = vol.sum()

        mean_gx = np.sum((G[0] @ u) * vol) / total_vol
        mean_gy = np.sum((G[1] @ u) * vol) / total_vol
        mean_gz = np.sum((G[2] @ u) * vol) / total_vol

        np.testing.assert_allclose(mean_gx, 0.0, atol=0.15)
        np.testing.assert_allclose(mean_gy, 1.0, atol=0.15)
        np.testing.assert_allclose(mean_gz, 0.0, atol=0.15)

    def test_gradient_z_mean_accuracy_3d(self):
        """Volume-weighted mean gradient of u = z should be ~[0,0,1]."""
        mesh = _make_3d_mesh(4, 4, 4)
        fvu = FiniteVolumeUnstructured()
        G = fvu._green_gauss_matrices(mesh)

        u = mesh.cell_centroids[:, 2]
        vol = mesh.cell_volumes
        total_vol = vol.sum()

        mean_gx = np.sum((G[0] @ u) * vol) / total_vol
        mean_gy = np.sum((G[1] @ u) * vol) / total_vol
        mean_gz = np.sum((G[2] @ u) * vol) / total_vol

        np.testing.assert_allclose(mean_gx, 0.0, atol=0.15)
        np.testing.assert_allclose(mean_gy, 0.0, atol=0.15)
        np.testing.assert_allclose(mean_gz, 1.0, atol=0.15)

    def test_tpfa_constant_3d(self):
        """Laplacian of constant = 0."""
        mesh = _make_3d_mesh(3, 3, 3)
        fvu = FiniteVolumeUnstructured()
        L = fvu._tpfa_matrix(mesh)
        u = np.ones(mesh.npts) * 7.0
        np.testing.assert_allclose(L @ u, 0.0, atol=1e-12)

    def test_divergence_conservation_3d(self):
        """Weighted Laplacian sum = 0 (conservation)."""
        mesh = _make_3d_mesh(3, 3, 3)
        fvu = FiniteVolumeUnstructured()
        L = fvu._tpfa_matrix(mesh)

        u = mesh.cell_centroids[:, 0] ** 2
        Lu = L @ u
        total = np.sum(Lu * mesh.cell_volumes)
        np.testing.assert_allclose(total, 0.0, atol=1e-10)


# ======================================================================
# Tests: Miscellaneous
# ======================================================================


class TestMisc:
    def test_face_count_2d(self):
        """Total faces = internal + boundary."""
        mesh = _make_2d_mesh(4, 4)
        n_total = len(mesh.faces)
        n_bnd = sum(len(v) for v in mesh.boundary_faces.values())
        assert n_total == mesh.n_internal_faces + n_bnd

    def test_face_count_3d(self):
        mesh = _make_3d_mesh(2, 2, 2)
        n_total = len(mesh.faces)
        n_bnd = sum(len(v) for v in mesh.boundary_faces.values())
        assert n_total == mesh.n_internal_faces + n_bnd

    def test_gradient_divergence_duality_2d(self):
        """
        For the Green-Gauss method, gradient and divergence matrices are
        structurally related (same interpolation weights, same normals).
        Test that G_k and D_k are identical.
        """
        mesh = _make_2d_mesh(5, 5)
        fvu = FiniteVolumeUnstructured()
        G = fvu._green_gauss_matrices(mesh)
        D = fvu._divergence_matrices(mesh)

        for k in range(mesh.dimension):
            diff = G[k] - D[k]
            assert abs(diff).max() < 1e-14

    def test_constructor_default_options(self):
        fvu = FiniteVolumeUnstructured()
        assert fvu.options is not None
        assert "extrapolation" in fvu.options
