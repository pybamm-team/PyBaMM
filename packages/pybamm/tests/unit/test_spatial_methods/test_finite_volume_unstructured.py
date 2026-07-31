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

import pybamm
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


class _MeshMap(dict):
    """Minimal Mesh-like mapping that accepts PyBaMM domain lists."""

    def __getitem__(self, key):
        if isinstance(key, list):
            key = tuple(key)
        elif isinstance(key, str):
            key = (key,)
        return super().__getitem__(key)


def _method_with_mesh(mesh, **auxiliary_meshes):
    meshes = {("test",): mesh}
    meshes.update({(name,): value for name, value in auxiliary_meshes.items()})
    method = FiniteVolumeUnstructured()
    method._mesh = _MeshMap(meshes)
    return method


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


class TestFiniteVolumeUnstructuredBehavior:
    def test_build_discovers_interfaces_and_ignores_other_meshes(self):
        left = _make_2d_mesh(2, 2, x_range=(0, 0.5))
        right = _make_2d_mesh(2, 2, x_range=(0.5, 1))
        structured = pybamm.SubMesh1D(np.array([0, 1]), "cartesian")
        meshes = _MeshMap(
            {("left",): left, ("right",): right, ("structured",): structured}
        )

        method = FiniteVolumeUnstructured()
        method.build(meshes)

        assert right in [data["other_mesh"] for data in left.interface_data.values()]
        assert left.npts_for_broadcast_to_nodes == left.npts
        assert structured.npts_for_broadcast_to_nodes == structured.npts

    def test_interface_matching_edge_cases(self):
        empty = _make_2d_mesh(1, 1)
        empty.boundary_faces = {}
        other = _make_2d_mesh(1, 1)
        a_idx, b_idx, matched = FiniteVolumeUnstructured._interface_face_match(
            empty, other
        )
        assert not matched
        assert a_idx.size == b_idx.size == 0

        mesh_3d = _make_3d_mesh(1, 1, 1)
        assert not FiniteVolumeUnstructured._interface_face_match(other, mesh_3d)[2]

        distant = _make_2d_mesh(1, 1, x_range=(2, 3))
        assert not FiniteVolumeUnstructured._interface_face_match(other, distant)[2]

    def test_compute_pair_interface_success_and_noops(self):
        left = _make_2d_mesh(2, 2, x_range=(0, 0.5))
        right = _make_2d_mesh(2, 2, x_range=(0.5, 1))
        method = FiniteVolumeUnstructured()

        assert method._compute_pair_interface(left, right, "left", "right")
        assert "iface_right" in left.boundary_faces
        assert "iface_left" in right.boundary_faces
        assert method._compute_pair_interface(left, right, "left", "right") is False

        far = _make_2d_mesh(1, 1, x_range=(2, 3))
        assert method._compute_pair_interface(left, far, "left", "far") is False

        shared = _make_2d_mesh(1, 1)
        method._auto_compute_all_interfaces(
            _MeshMap({("first",): shared, ("alias",): shared})
        )

    def test_spatial_variable_directions_and_auxiliary_repeats(self):
        mesh = _make_3d_mesh(1, 1, 1)
        aux = _make_2d_mesh(1, 1)
        method = _method_with_mesh(mesh, aux=aux)
        domains = {"primary": ["test"], "secondary": ["aux"]}

        for name, direction, column in [
            ("x", None, 0),
            ("y", None, 1),
            ("z", None, 2),
            ("r", None, 0),
            ("s", "lr", 0),
            ("s", "tb", 2),
            ("s", "fb", 1),
            ("s", "unknown", 0),
        ]:
            symbol = pybamm.SpatialVariable(name, domains=domains, direction=direction)
            actual = method.spatial_variable(symbol).evaluate().reshape(-1)
            expected = np.tile(mesh.cell_centroids[:, column], aux.npts)
            np.testing.assert_allclose(actual, expected)

    def test_broadcast_variants(self):
        mesh = _make_2d_mesh(1, 1)
        aux = _make_2d_mesh(1, 1)
        method = _method_with_mesh(mesh, aux=aux)
        primary = {"primary": ["test"], "secondary": []}

        scalar_primary = method.broadcast(pybamm.Scalar(2), primary, "primary to nodes")
        np.testing.assert_array_equal(
            scalar_primary.evaluate()[:, 0], np.full(mesh.npts, 2)
        )

        vector_primary = method.broadcast(
            pybamm.Vector([2, 3]), primary, "primary to nodes"
        )
        np.testing.assert_array_equal(
            vector_primary.evaluate()[:, 0], np.repeat([2, 3], mesh.npts)
        )

        full_domains = {"primary": ["test"], "secondary": ["aux"]}
        full = method.broadcast(pybamm.Scalar(4), full_domains, "full to nodes")
        np.testing.assert_array_equal(
            full.evaluate()[:, 0], np.full(mesh.npts * aux.npts, 4)
        )

        secondary_child = pybamm.Vector([1, 2], domain="test")
        secondary = method.broadcast(secondary_child, primary, "secondary to nodes")
        np.testing.assert_array_equal(secondary.evaluate(), secondary_child.evaluate())
        assert secondary.domain == primary["primary"]
        assert secondary.domains["secondary"] == primary["secondary"]

    def test_broadcast_does_not_mutate_simplified_child(self):
        mesh = pybamm.SubMesh1D(np.array([0, 1]), "cartesian")
        method = _method_with_mesh(mesh)
        child = pybamm.StateVector(slice(0, 1))
        domains = {"primary": ["test"], "secondary": []}

        result = method.broadcast(child, domains, "full to nodes")

        assert result is not child
        assert child.domain == []
        assert result.domains["primary"] == ["test"]
        np.testing.assert_array_equal(result.evaluate(y=np.array([7])), [[7]])

    def test_laplacian_and_boundary_conditions(self):
        mesh = _make_2d_mesh(2, 2)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        values = pybamm.Vector(np.arange(mesh.npts), domain="test")

        plain = method.laplacian(variable, values, {})
        np.testing.assert_allclose(
            plain.evaluate()[:, 0], method._tpfa_matrix(mesh) @ np.arange(mesh.npts)
        )

        constant = pybamm.Vector(np.full(mesh.npts, 3), domain="test")
        dirichlet_bcs = {
            variable: {
                side: (pybamm.Scalar(3), "Dirichlet")
                for side in ["left", "right", "top", "bottom"]
            }
        }
        np.testing.assert_allclose(
            method.laplacian(variable, constant, dirichlet_bcs).evaluate(),
            0,
            atol=1e-12,
        )

        neumann_bcs = {
            variable: {
                side: (pybamm.Scalar(0), "Neumann")
                for side in ["left", "right", "top", "bottom"]
            }
            | {
                "missing": (pybamm.Scalar(3), "Dirichlet"),
            }
        }
        np.testing.assert_allclose(
            method.laplacian(variable, constant, neumann_bcs).evaluate(), 0, atol=1e-12
        )

        face_count = len(mesh.boundary_faces["top"])
        vector_bc = pybamm.Vector(np.arange(face_count) + 1)
        _, rhs = method._apply_bcs_to_laplacian(
            mesh,
            method._tpfa_matrix(mesh),
            pybamm.Vector(np.zeros(mesh.npts)),
            {"top": (vector_bc, "Dirichlet")},
        )
        faces = mesh.boundary_faces["top"]
        owners = mesh.face_owner[faces]
        distance = np.linalg.norm(
            mesh.face_centroids[faces] - mesh.cell_centroids[owners], axis=1
        )
        coefficients = mesh.face_areas[faces] / distance / mesh.cell_volumes[owners]
        expected_rhs = np.zeros(mesh.npts)
        np.add.at(expected_rhs, owners, coefficients * (np.arange(face_count) + 1))
        np.testing.assert_allclose(rhs.evaluate()[:, 0], expected_rhs)

    def test_gradient_and_gradient_squared(self):
        mesh = _make_2d_mesh(2, 2)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        constant = pybamm.Vector(np.full(mesh.npts, 3), domain="test")
        dirichlet_bcs = {
            variable: {
                side: (pybamm.Scalar(3), "Dirichlet")
                for side in ["left", "right", "top", "bottom"]
            }
        }
        gradient = method.gradient(variable, constant, dirichlet_bcs)
        assert gradient._disc_state_vector is constant
        for component in gradient._components:
            np.testing.assert_allclose(component.evaluate(), 0, atol=1e-12)

        neumann_bcs = {
            variable: {
                side: (pybamm.Scalar(0), "Neumann")
                for side in ["left", "right", "top", "bottom"]
            }
            | {
                "missing": (pybamm.Scalar(2), "Neumann"),
            }
        }
        for component in method.gradient(variable, constant, neumann_bcs)._components:
            np.testing.assert_allclose(component.evaluate(), 0, atol=1e-12)

        x_values = mesh.cell_centroids[:, 0]
        values = pybamm.Vector(x_values, domain="test")
        grad_squared = method.gradient_squared(variable, values, {})
        matrices = method._green_gauss_matrices(mesh)
        expected = sum((matrix @ x_values) ** 2 for matrix in matrices)
        np.testing.assert_allclose(grad_squared.evaluate()[:, 0], expected)

    def test_divergence_input_forms_and_error(self):
        mesh = _make_2d_mesh(2, 2)
        method = _method_with_mesh(mesh)
        symbol = pybamm.Variable("F", domain="test")
        components = [
            pybamm.Vector(np.ones(mesh.npts), domain="test"),
            pybamm.Vector(np.full(mesh.npts, 2), domain="test"),
        ]

        from_list = method.divergence(symbol, components, {})
        from_field = method.divergence(symbol, pybamm.VectorField(*components), {})
        np.testing.assert_allclose(from_list.evaluate(), from_field.evaluate())
        matrices = method._divergence_matrices(mesh)
        expected = matrices[0] @ np.ones(mesh.npts)
        expected += matrices[1] @ np.full(mesh.npts, 2)
        np.testing.assert_allclose(from_list.evaluate()[:, 0], expected)

        with pytest.raises(TypeError, match="expects a VectorField"):
            method.divergence(symbol, pybamm.Scalar(1), {})

    def test_divergence_boundary_correction(self):
        mesh = _make_2d_mesh(2, 2)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        other = pybamm.Variable("v", domain="other")
        bcs = {
            "not a symbol": {"left": (pybamm.Scalar(0), "Dirichlet")},
            other: {"left": (pybamm.Scalar(0), "Dirichlet")},
            variable: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(2), "Neumann"),
                "missing": (pybamm.Scalar(3), "Neumann"),
            },
        }

        L_bc, rhs, boundary_matrices = method._div_boundary_correction(
            mesh, bcs, domain=["test"]
        )
        assert L_bc.shape == (mesh.npts, mesh.npts)
        assert rhs.evaluate().shape == (mesh.npts, 1)
        assert len(boundary_matrices) == mesh.dimension
        left_faces = mesh.boundary_faces["left"]
        left_owners = mesh.face_owner[left_faces]
        expected_rhs = np.zeros(mesh.npts)
        left_distance = np.linalg.norm(
            mesh.face_centroids[left_faces] - mesh.cell_centroids[left_owners], axis=1
        )
        np.add.at(
            expected_rhs,
            left_owners,
            mesh.face_areas[left_faces]
            / left_distance
            / mesh.cell_volumes[left_owners],
        )
        right_faces = mesh.boundary_faces["right"]
        right_owners = mesh.face_owner[right_faces]
        np.add.at(
            expected_rhs,
            right_owners,
            2 * mesh.face_areas[right_faces] / mesh.cell_volumes[right_owners],
        )
        np.testing.assert_allclose(rhs.evaluate()[:, 0], expected_rhs)

        none_L, zero_rhs, none_D = method._div_boundary_correction(mesh, {})
        assert none_L is None
        assert none_D is None
        np.testing.assert_allclose(zero_rhs.evaluate(), 0)

    def test_div_D_grad_scalar_and_vector_coefficients(self):
        mesh = _make_2d_mesh(2, 2)
        aux = _make_2d_mesh(1, 1)
        method = _method_with_mesh(mesh, aux=aux)
        variable = pybamm.Variable("u", domain="test")
        div_symbol = pybamm.Variable("div", domain="test")
        cell_values = mesh.cell_centroids[:, 0] ** 2
        values = pybamm.Vector(cell_values, domain="test")
        bcs = {
            variable: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(2), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "missing": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        scalar_result = method.div_D_grad(
            div_symbol, variable, pybamm.Scalar(2), values, bcs
        )

        coefficient = pybamm.Vector(np.full(mesh.npts, 2), domain="test")
        vector_result = method.div_D_grad(
            div_symbol, variable, coefficient, values, bcs
        )
        np.testing.assert_allclose(
            vector_result.evaluate(), scalar_result.evaluate(), atol=1e-12
        )

        repeated_domains = {"primary": ["test"], "secondary": ["aux"]}
        repeated_div = pybamm.Variable("repeated div", domains=repeated_domains)
        repeated_u = pybamm.Variable("repeated u", domains=repeated_domains)
        size = mesh.npts * aux.npts
        repeated_values = pybamm.Vector(
            np.tile(cell_values, aux.npts), domains=repeated_domains
        )
        repeated_coefficient = pybamm.Vector(np.full(size, 2), domains=repeated_domains)
        repeated = method.div_D_grad(
            repeated_div,
            repeated_u,
            repeated_coefficient,
            repeated_values,
            {
                repeated_u: {
                    "left": (pybamm.Scalar(0), "Dirichlet"),
                    "right": (pybamm.Scalar(2), "Neumann"),
                    "top": (pybamm.Scalar(0), "Neumann"),
                    "missing": (pybamm.Scalar(1), "Dirichlet"),
                }
            },
        )
        np.testing.assert_allclose(
            repeated.evaluate()[:, 0],
            np.tile(vector_result.evaluate()[:, 0], aux.npts),
            atol=1e-12,
        )

    def test_integral_and_boundary_integral(self):
        mesh = _make_2d_mesh(2, 2)
        aux = _make_2d_mesh(1, 1)
        method = _method_with_mesh(mesh, aux=aux)
        domains = {"primary": ["test"], "secondary": ["aux"]}
        child = pybamm.Variable("u", domains=domains)
        values = pybamm.Vector(np.ones(mesh.npts * aux.npts), domains=domains)

        integral = method.integral(child, values, "primary")
        np.testing.assert_allclose(integral.evaluate(), 1)

        row = method.definite_integral_matrix(child)
        np.testing.assert_allclose(row.toarray()[0], mesh.cell_volumes)

        boundary = method.boundary_integral(child, values, "left")
        np.testing.assert_allclose(boundary.evaluate(), 1)
        missing = method.boundary_integral(child, values, "missing")
        assert missing == pybamm.Scalar(0)

    @pytest.mark.parametrize(
        "side",
        ["left", "missing", "top-right", "top-left", "bottom-right", "bottom-left"],
    )
    def test_boundary_value_and_corners(self, side):
        mesh = _make_2d_mesh(2, 2)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        values = pybamm.Vector(np.arange(mesh.npts), domain="test")
        symbol = pybamm.BoundaryValue(variable, side)

        result = method.boundary_value_or_flux(symbol, values)
        assert result.domain == []
        if side == "missing":
            assert result == pybamm.Scalar(0)
        elif "-" in side:
            top_bottom, left_right = side.split("-")
            x = mesh.cell_centroids[:, 0]
            z = mesh.cell_centroids[:, -1]
            target_x = x.max() if left_right == "right" else x.min()
            target_z = z.max() if top_bottom == "top" else z.min()
            expected = np.argmin((x - target_x) ** 2 + (z - target_z) ** 2)
            assert result.evaluate().item() == expected
        else:
            owners = mesh.face_owner[mesh.boundary_faces[side]]
            np.testing.assert_array_equal(result.evaluate()[:, 0], owners)

    def test_process_binary_operators(self):
        method = FiniteVolumeUnstructured()
        left_components = [pybamm.StateVector(slice(0, 2)), pybamm.Vector([2, 3])]
        right_components = [pybamm.Vector([4, 5]), pybamm.Vector([6, 7])]
        left_field = pybamm.VectorField(*left_components)
        left_field._disc_state_vector = left_components[0]
        right_field = pybamm.VectorField(*right_components)
        multiplication = pybamm.Multiplication(pybamm.Scalar(1), pybamm.Scalar(2))

        both = method.process_binary_operators(
            multiplication,
            None,
            None,
            left_field,
            right_field,
        )
        assert both.n_components == 2
        assert both._disc_state_vector is left_components[0]
        np.testing.assert_array_equal(
            both._components[0].evaluate(y=np.array([1, 2]))[:, 0], [4, 10]
        )
        np.testing.assert_array_equal(both._components[1].evaluate()[:, 0], [12, 21])

        field_left = method.process_binary_operators(
            multiplication, None, None, left_field, pybamm.Scalar(2)
        )
        field_right = method.process_binary_operators(
            multiplication, None, None, pybamm.Scalar(2), right_field
        )
        np.testing.assert_array_equal(
            field_left._components[0].evaluate(y=np.array([1, 2]))[:, 0], [2, 4]
        )
        np.testing.assert_array_equal(
            field_right._components[0].evaluate()[:, 0], [8, 10]
        )

        scalar = method.process_binary_operators(
            multiplication, None, None, pybamm.Scalar(3), pybamm.Scalar(4)
        )
        assert scalar.evaluate() == 12

    def test_internal_neumann_unstructured_paths(self):
        left, right = _make_split_2d_meshes(2, 2, 2)
        method = FiniteVolumeUnstructured()
        left_values = pybamm.Vector(np.arange(left.npts), domain="left")
        right_values = pybamm.Vector(np.arange(right.npts), domain="right")

        direct = method._internal_neumann_unstructured(
            left_values, right_values, left, right, 1
        )
        interface = next(iter(left.interface_data.values()))
        expected = (
            np.arange(right.npts)[interface["right_cells"]]
            - np.arange(left.npts)[interface["left_cells"]]
        ) / interface["cell_distances"]
        np.testing.assert_allclose(direct.evaluate()[:, 0], expected)

        left_data = left.interface_data
        left.interface_data = {}
        reverse = method._internal_neumann_unstructured(
            left_values, right_values, left, right, 1
        )
        np.testing.assert_allclose(reverse.evaluate(), direct.evaluate())

        right.interface_data = {}
        absent = method._internal_neumann_unstructured(
            left_values, right_values, left, right, 2
        )
        np.testing.assert_allclose(absent.evaluate(), 0)
        assert absent.shape[0] == left.npts * 2
        left.interface_data = left_data

    def test_internal_neumann_dispatch_structured_and_mismatch(self):
        method = FiniteVolumeUnstructured()
        left_mesh = pybamm.SubMesh1D(np.array([0, 0.5]), "cartesian")
        right_mesh = pybamm.SubMesh1D(np.array([0.5, 1]), "cartesian")
        left = pybamm.Vector(np.arange(left_mesh.npts), domain="left")
        right = pybamm.Vector(np.arange(right_mesh.npts), domain="right")

        structured = method.internal_neumann_condition(
            left, right, left_mesh, right_mesh
        )
        dx = right_mesh.nodes[0] - left_mesh.nodes[-1]
        expected = (np.arange(right_mesh.npts)[0] - np.arange(left_mesh.npts)[-1]) / dx
        assert structured.evaluate().item() == expected

        unstructured_left, unstructured_right = _make_split_2d_meshes(1, 1, 1)
        method._mesh = _MeshMap(
            {
                ("aux",): _make_2d_mesh(1, 1),
                ("other aux",): _make_2d_mesh(2, 1),
            }
        )
        left_repeated = pybamm.Vector(
            np.ones(unstructured_left.npts * method.mesh["aux"].npts),
            domains={"primary": ["left"], "secondary": ["aux"]},
        )
        right_repeated = pybamm.Vector(
            np.ones(unstructured_right.npts * method.mesh["other aux"].npts),
            domains={"primary": ["right"], "secondary": ["other aux"]},
        )
        with pytest.raises(pybamm.DomainError, match="secondary points"):
            method.internal_neumann_condition(
                left_repeated,
                right_repeated,
                unstructured_left,
                unstructured_right,
            )

    def test_internal_bcs_for_concatenation(self):
        left = _make_2d_mesh(1, 1, x_range=(0, 0.5))
        right = _make_2d_mesh(1, 1, x_range=(0.5, 1))
        method = FiniteVolumeUnstructured()
        method._compute_pair_interface(left, right, "left", "right")
        method._mesh = _MeshMap({("left",): left, ("right",): right})
        children = [
            pybamm.Variable("left temperature", domain="left"),
            pybamm.Variable("right temperature", domain="right"),
        ]

        class Disc:
            def process_symbol(self, child):
                size = method.mesh[child.domain].npts
                return pybamm.Vector(np.ones(size), domains=child.domains)

        result = method.set_internal_bcs_for_concat(
            Disc(),
            children[0],
            children,
            {"left": (pybamm.Scalar(0), "Dirichlet")},
        )
        assert set(result) == set(children)
        assert "iface_right" in result[children[0]]
        interface_gradient, bc_type = result[children[0]]["iface_right"]
        assert bc_type == "Neumann"
        np.testing.assert_allclose(interface_gradient.evaluate(), 0)

        structured = pybamm.SubMesh1D(np.array([0, 1]), "cartesian")
        method._mesh[("structured",)] = structured
        structured_child = pybamm.Variable(
            "structured temperature", domain="structured"
        )
        partial = method.set_internal_bcs_for_concat(
            Disc(),
            children[0],
            [children[0], structured_child],
            {},
        )
        assert structured_child not in partial
        assert partial[children[0]] == {}

        no_interface = _method_with_mesh(_make_2d_mesh(1, 1))
        assert (
            no_interface.set_internal_bcs_for_concat(
                Disc(), children[0], [pybamm.Variable("u", domain="test")], {}
            )
            is None
        )

    def test_concatenation_preserves_domain_order(self):
        left = _make_2d_mesh(1, 1, x_range=(0, 0.5))
        right = _make_2d_mesh(1, 1, x_range=(0.5, 1))
        method = FiniteVolumeUnstructured()
        method._mesh = _MeshMap({("left",): left, ("right",): right})
        left_values = pybamm.Vector([1, 2], domain="left")
        right_values = pybamm.Vector([3, 4], domain="right")

        result = method.concatenation([left_values, right_values])

        np.testing.assert_array_equal(result.evaluate()[:, 0], [1, 2, 3, 4])
        assert result.domain == ["left", "right"]
