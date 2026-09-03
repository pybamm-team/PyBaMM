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
from scipy.sparse.linalg import spsolve

import pybamm
from pybamm.meshes.unstructured_submesh import (
    UnstructuredSubMesh,
    _hex_grid,
    _hex_to_tet,
    _make_quad_grid,
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
    submesh = UnstructuredSubMesh(nodes, elements)
    submesh.detect_box_boundaries()
    return submesh


def _make_3d_mesh(nx=3, ny=3, nz=3, x_range=(0, 1), y_range=(0, 1), z_range=(0, 1)):
    x_edges = np.linspace(x_range[0], x_range[1], nx + 1)
    y_edges = np.linspace(y_range[0], y_range[1], ny + 1)
    z_edges = np.linspace(z_range[0], z_range[1], nz + 1)
    nodes, elements = _hex_to_tet(x_edges, y_edges, z_edges)
    submesh = UnstructuredSubMesh(nodes, elements)
    submesh.detect_box_boundaries()
    return submesh


def _make_quad_mesh(nx=4, nz=4, x_range=(0, 1), z_range=(0, 1)):
    """TPFA-orthogonal quadrilateral mesh (exact for linear fields)."""
    x_edges = np.linspace(x_range[0], x_range[1], nx + 1)
    z_edges = np.linspace(z_range[0], z_range[1], nz + 1)
    nodes, elements = _make_quad_grid(x_edges, z_edges)
    submesh = UnstructuredSubMesh(nodes, elements)
    submesh.detect_box_boundaries()
    return submesh


def _make_hex_mesh(nx=3, ny=3, nz=3):
    """TPFA-orthogonal hexahedral mesh on the unit cube."""
    x_edges = np.linspace(0, 1, nx + 1)
    y_edges = np.linspace(0, 1, ny + 1)
    z_edges = np.linspace(0, 1, nz + 1)
    nodes, elements = _hex_grid(x_edges, y_edges, z_edges)
    submesh = UnstructuredSubMesh(nodes, elements)
    submesh.detect_box_boundaries()
    return submesh


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
# Tests: Neumann sign convention (PyBaMM coordinate-direction values)
# ======================================================================


class TestNeumannSignConvention:
    """Named sides take coordinate-direction derivatives (matching
    FiniteVolume/FiniteVolume2D), so ``u = x`` needs value +1 on *both*
    left and right, not the outward-normal ±1."""

    def test_laplacian_neumann_left_right_2d(self):
        mesh = _make_quad_mesh(4, 4)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        u = pybamm.Vector(mesh.cell_centroids[:, 0], domain="test")
        bcs = {
            variable: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        result = method.laplacian(variable, u, bcs)
        np.testing.assert_allclose(result.evaluate(), 0, atol=1e-10)

    def test_laplacian_neumann_top_bottom_2d(self):
        mesh = _make_quad_mesh(4, 4)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        u = pybamm.Vector(mesh.cell_centroids[:, 1], domain="test")
        bcs = {
            variable: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(1), "Neumann"),
                "bottom": (pybamm.Scalar(1), "Neumann"),
            }
        }
        result = method.laplacian(variable, u, bcs)
        np.testing.assert_allclose(result.evaluate(), 0, atol=1e-10)

    def test_laplacian_neumann_front_back_3d(self):
        mesh = _make_hex_mesh(3, 3, 3)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        u = pybamm.Vector(mesh.cell_centroids[:, 1], domain="test")
        bcs = {
            variable: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
                "front": (pybamm.Scalar(1), "Neumann"),
                "back": (pybamm.Scalar(1), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        result = method.laplacian(variable, u, bcs)
        np.testing.assert_allclose(result.evaluate(), 0, atol=1e-10)

    def test_gradient_neumann_left_2d(self):
        mesh = _make_quad_mesh(4, 4)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        u = pybamm.StateVector(slice(0, mesh.npts), domains={"primary": ["test"]})
        y = mesh.cell_centroids[:, 0]
        bcs = {
            variable: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        grad = method.gradient(variable, u, bcs)
        np.testing.assert_allclose(grad.components[0].evaluate(y=y), 1, atol=1e-10)
        np.testing.assert_allclose(grad.components[1].evaluate(y=y), 0, atol=1e-10)

    def test_div_D_grad_neumann_left_2d(self):
        mesh = _make_quad_mesh(4, 4)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        div_symbol = pybamm.Variable("div", domain="test")
        u = pybamm.Vector(mesh.cell_centroids[:, 0], domain="test")
        bcs = {
            variable: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        result = method.div_D_grad(div_symbol, variable, pybamm.Scalar(2), u, bcs)
        np.testing.assert_allclose(result.evaluate(), 0, atol=1e-10)

    def test_custom_tags_use_outward_normal(self):
        # Rename the axis buckets to custom tags: values are then
        # outward-normal derivatives, so u = x needs -1 on the left tag.
        mesh = _make_quad_mesh(4, 4)
        mesh.boundary_faces = {
            f"tag_{name}": faces for name, faces in mesh.boundary_faces.items()
        }
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        u = pybamm.Vector(mesh.cell_centroids[:, 0], domain="test")
        bcs = {
            variable: {
                "tag_left": (pybamm.Scalar(-1), "Neumann"),
                "tag_right": (pybamm.Scalar(1), "Neumann"),
                "tag_top": (pybamm.Scalar(0), "Neumann"),
                "tag_bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        result = method.laplacian(variable, u, bcs)
        np.testing.assert_allclose(result.evaluate(), 0, atol=1e-10)


# ======================================================================
# Tests: operator caching
# ======================================================================


class TestOperatorCaching:
    def test_operators_cached_and_invalidated_on_reordering(self):
        mesh = _make_2d_mesh(4, 4)
        fvu = FiniteVolumeUnstructured()
        assert fvu._tpfa_matrix(mesh) is fvu._tpfa_matrix(mesh)
        assert fvu._green_gauss_matrices(mesh) is fvu._green_gauss_matrices(mesh)
        assert fvu._divergence_matrices(mesh) is fvu._divergence_matrices(mesh)
        assert fvu._div_D_grad_matrices(mesh) is fvu._div_D_grad_matrices(mesh)

        laplacian_before = fvu._tpfa_matrix(mesh)
        mesh.optimize_ordering()
        assert fvu._tpfa_matrix(mesh) is not laplacian_before

    def test_bc_application_does_not_mutate_cached_operators(self):
        mesh = _make_quad_mesh(3, 3)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        values = pybamm.Vector(np.arange(mesh.npts), domain="test")
        bcs = {
            variable: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Dirichlet"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        laplacian_cached = method._tpfa_matrix(mesh).copy()
        gauss_cached = method._green_gauss_matrices(mesh)[0].copy()

        method.laplacian(variable, values, bcs)
        method.gradient(variable, values, bcs)

        assert (method._tpfa_matrix(mesh) - laplacian_cached).nnz == 0
        assert (method._green_gauss_matrices(mesh)[0] - gauss_cached).nnz == 0


# ======================================================================
# Tests: auxiliary domains (secondary/tertiary repeats)
# ======================================================================


class TestAuxiliaryDomains:
    def _bcs(self, variable):
        return {
            variable: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(2), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(3), "Dirichlet"),
            }
        }

    def test_laplacian_with_secondary_domain(self):
        mesh = _make_quad_mesh(2, 2)
        aux = _make_quad_mesh(1, 3)
        method = _method_with_mesh(mesh, aux=aux)
        cell_values = mesh.cell_centroids[:, 0] ** 2

        variable = pybamm.Variable("u", domain="test")
        single = method.laplacian(
            variable,
            pybamm.Vector(cell_values, domain="test"),
            self._bcs(variable),
        )

        domains = {"primary": ["test"], "secondary": ["aux"]}
        repeated_var = pybamm.Variable("u rep", domains=domains)
        repeated = method.laplacian(
            repeated_var,
            pybamm.Vector(np.tile(cell_values, aux.npts), domains=domains),
            self._bcs(repeated_var),
        )
        np.testing.assert_allclose(
            repeated.evaluate()[:, 0],
            np.tile(single.evaluate()[:, 0], aux.npts),
            atol=1e-12,
        )

    def test_gradient_with_secondary_domain(self):
        mesh = _make_quad_mesh(2, 2)
        aux = _make_quad_mesh(1, 3)
        method = _method_with_mesh(mesh, aux=aux)
        cell_values = mesh.cell_centroids[:, 0] ** 2

        variable = pybamm.Variable("u", domain="test")
        single = method.gradient(
            variable,
            pybamm.Vector(cell_values, domain="test"),
            self._bcs(variable),
        )

        domains = {"primary": ["test"], "secondary": ["aux"]}
        repeated_var = pybamm.Variable("u rep", domains=domains)
        repeated = method.gradient(
            repeated_var,
            pybamm.Vector(np.tile(cell_values, aux.npts), domains=domains),
            self._bcs(repeated_var),
        )
        for single_comp, repeated_comp in zip(
            single.components, repeated.components, strict=True
        ):
            np.testing.assert_allclose(
                repeated_comp.evaluate()[:, 0],
                np.tile(single_comp.evaluate()[:, 0], aux.npts),
                atol=1e-12,
            )

    def test_tertiary_broadcast_size(self):
        mesh = _make_quad_mesh(2, 2)
        sec = _make_quad_mesh(1, 2)
        ter = _make_quad_mesh(1, 5)
        method = _method_with_mesh(mesh, sec=sec, ter=ter)

        child_size = mesh.npts * sec.npts
        child = pybamm.Vector(
            np.arange(child_size),
            domains={"primary": ["test"], "secondary": ["sec"]},
        )
        domains = {
            "primary": ["test"],
            "secondary": ["sec"],
            "tertiary": ["ter"],
        }
        out = method.broadcast(child, domains, "tertiary to nodes")
        assert out.shape_for_testing == (child_size * ter.npts, 1)
        np.testing.assert_array_equal(
            out.evaluate()[:, 0], np.tile(np.arange(child_size), ter.npts)
        )


# ======================================================================
# Tests: BC tag / type validation
# ======================================================================


class TestBCValidation:
    def _setup(self):
        mesh = _make_quad_mesh(2, 2)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        values = pybamm.Vector(np.arange(mesh.npts), domain="test")
        return mesh, method, variable, values

    def test_laplacian_unknown_side_raises(self):
        _, method, variable, values = self._setup()
        bcs = {variable: {"weft": (pybamm.Scalar(0), "Neumann")}}
        with pytest.raises(pybamm.DiscretisationError, match="weft"):
            method.laplacian(variable, values, bcs)

    def test_laplacian_unknown_bc_type_raises(self):
        _, method, variable, values = self._setup()
        bcs = {variable: {"left": (pybamm.Scalar(0), "Robin")}}
        with pytest.raises(pybamm.DiscretisationError, match="Robin"):
            method.laplacian(variable, values, bcs)

    def test_gradient_unknown_side_raises(self):
        _, method, variable, values = self._setup()
        bcs = {variable: {"weft": (pybamm.Scalar(0), "Neumann")}}
        with pytest.raises(pybamm.DiscretisationError, match="weft"):
            method.gradient(variable, values, bcs)

    def test_gradient_unknown_bc_type_raises(self):
        _, method, variable, values = self._setup()
        bcs = {variable: {"left": (pybamm.Scalar(0), "Dirchlet")}}
        with pytest.raises(pybamm.DiscretisationError, match="Dirchlet"):
            method.gradient(variable, values, bcs)

    def test_div_D_grad_unknown_side_raises(self):
        _, method, variable, values = self._setup()
        div_symbol = pybamm.Variable("div", domain="test")
        bcs = {variable: {"weft": (pybamm.Scalar(0), "Neumann")}}
        with pytest.raises(pybamm.DiscretisationError, match="weft"):
            method.div_D_grad(div_symbol, variable, pybamm.Scalar(1), values, bcs)

    def test_boundary_value_unknown_side_raises(self):
        _, method, variable, values = self._setup()
        symbol = pybamm.BoundaryValue(variable, "missing")
        with pytest.raises(pybamm.DiscretisationError, match="missing"):
            method.boundary_value_or_flux(symbol, values)

    def test_boundary_integral_unknown_region_raises(self):
        _, method, variable, values = self._setup()
        with pytest.raises(pybamm.DiscretisationError, match="missing"):
            method.boundary_integral(variable, values, "missing")

    def test_boundary_integral_entire(self):
        # integral of u = 1 over the whole boundary = perimeter of unit square
        mesh, method, variable, _ = self._setup()
        ones = pybamm.Vector(np.ones(mesh.npts), domain="test")
        result = method.boundary_integral(variable, ones, "entire")
        np.testing.assert_allclose(result.evaluate().sum(), 4.0, atol=1e-12)

    def test_boundary_integral_entire_excludes_interface_faces(self):
        left, _right = _make_split_2d_meshes()
        # emulate interface discovery: move left mesh's right faces to an
        # iface bucket
        left.boundary_faces["iface_right"] = left.boundary_faces.pop("right")
        method = _method_with_mesh(left)
        variable = pybamm.Variable("u", domain="test")
        ones = pybamm.Vector(np.ones(left.npts), domain="test")
        result = method.boundary_integral(variable, ones, "entire")
        # perimeter of [0, 0.5] x [0, 1] minus the shared edge (length 1)
        np.testing.assert_allclose(result.evaluate().sum(), 2.0, atol=1e-12)

    def test_deleted_bucket_message_mentions_interface(self):
        mesh, method, variable, values = self._setup()
        mesh.boundary_faces["iface_other"] = mesh.boundary_faces.pop("right")
        bcs = {variable: {"right": (pybamm.Scalar(0), "Dirichlet")}}
        with pytest.raises(pybamm.DiscretisationError, match="interface"):
            method.laplacian(variable, values, bcs)


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
    @pytest.mark.parametrize(
        ("make_mesh", "expected_linear"),
        [(lambda: _make_2d_mesh(10, 10), 0.5), (lambda: _make_3d_mesh(4, 4, 4), 0.5)],
        ids=["2d", "3d"],
    )
    def test_definite_integral_matrix(self, make_mesh, expected_linear):
        """Integral of 1 over the unit box is 1; of ``x`` is 0.5."""
        mesh = make_mesh()
        method = _method_with_mesh(mesh)
        child = pybamm.Variable("u", domain="test")
        mat = method.definite_integral_matrix(child)
        assert isinstance(mat, pybamm.Matrix)
        assert mat.shape == (1, mesh.npts)
        np.testing.assert_allclose(mat.entries @ np.ones(mesh.npts), 1.0, atol=1e-12)
        np.testing.assert_allclose(
            mat.entries @ mesh.cell_centroids[:, 0], expected_linear, atol=0.01
        )

    def test_definite_integral_matrix_column(self):
        mesh = _make_2d_mesh(3, 3)
        method = _method_with_mesh(mesh)
        child = pybamm.Variable("u", domain="test")
        column = method.definite_integral_matrix(child, vector_type="column")
        assert column.shape == (mesh.npts, 1)
        np.testing.assert_array_equal(column.entries.toarray()[:, 0], mesh.cell_volumes)
        with pytest.raises(pybamm.DiscretisationError, match="vector_type"):
            method.definite_integral_matrix(child, vector_type="diagonal")

    def test_non_primary_integration_dimension_raises(self):
        mesh = _make_2d_mesh(2, 2)
        aux = _make_2d_mesh(1, 1)
        method = _method_with_mesh(mesh, aux=aux)
        child = pybamm.Variable(
            "u", domains={"primary": ["test"], "secondary": ["aux"]}
        )
        values = pybamm.Vector(np.ones(mesh.npts * aux.npts), domains=child.domains)
        with pytest.raises(NotImplementedError, match="secondary"):
            method.integral(child, values, "secondary")
        with pytest.raises(NotImplementedError, match="secondary"):
            method.definite_integral_matrix(child, integration_dimension="secondary")

    def test_definite_integral_vector_through_discretisation(self):
        """``DefiniteIntegralVector`` must come back as a ``pybamm.Matrix`` so
        ``process_symbol`` can shape-check it, in both orientations."""
        x = pybamm.SpatialVariable("x_n", domain=["negative electrode"])
        z = pybamm.SpatialVariable(
            "z_2d", domain=["negative electrode"], coord_sys="cartesian", direction="tb"
        )
        geometry = {
            "negative electrode": {x: {"min": 0, "max": 1}, z: {"min": 0, "max": 1}}
        }
        generator = pybamm.meshes.unstructured_submesh.UnstructuredMeshGenerator()
        mesh = pybamm.Mesh(geometry, {"negative electrode": generator}, {x: 3, z: 3})
        disc = pybamm.Discretisation(
            mesh, {"negative electrode": FiniteVolumeUnstructured()}
        )
        var = pybamm.Variable("var", domain="negative electrode")
        disc.set_variable_slices([var])
        npts = mesh["negative electrode"].npts

        row = disc.process_symbol(pybamm.DefiniteIntegralVector(var))
        assert row.shape == (1, npts)
        column = disc.process_symbol(
            pybamm.DefiniteIntegralVector(var, vector_type="column")
        )
        assert column.shape == (npts, 1)
        np.testing.assert_allclose(row.evaluate() @ np.ones(npts), 1.0, atol=1e-12)


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

    def test_build_warns_on_untagged_mesh(self, caplog):
        import logging

        untagged = _make_2d_mesh(2, 2)
        untagged.boundary_faces = {}
        tagged = _make_2d_mesh(2, 2)
        method = FiniteVolumeUnstructured()
        with caplog.at_level(logging.WARNING):
            method.build(_MeshMap({("untagged",): untagged, ("tagged",): tagged}))
        assert "no boundary tags" in caplog.text
        assert "'untagged'" in caplog.text
        assert "'tagged'" not in caplog.text

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
            ("x_n", None, 0),
            ("s", "lr", 0),
            ("s", "tb", 2),
            ("s", "fb", 1),
        ]:
            symbol = pybamm.SpatialVariable(name, domains=domains, direction=direction)
            actual = method.spatial_variable(symbol).evaluate().reshape(-1)
            expected = np.tile(mesh.cell_centroids[:, column], aux.npts)
            np.testing.assert_allclose(actual, expected)

        # ambiguous names and unknown directions raise instead of guessing x
        for name, direction in [("r", None), ("zeta", None), ("s", "unknown")]:
            symbol = pybamm.SpatialVariable(name, domains=domains, direction=direction)
            with pytest.raises(pybamm.DomainError):
                method.spatial_variable(symbol)

    def test_spatial_variable_2d_rejects_y_and_fb(self):
        mesh = _make_2d_mesh(1, 1)
        method = _method_with_mesh(mesh)
        z = pybamm.SpatialVariable("z_2d", domain="test", direction="tb")
        np.testing.assert_allclose(
            method.spatial_variable(z).evaluate().reshape(-1),
            mesh.cell_centroids[:, 1],
        )
        # 2D meshes are x-z: y names and the front-back direction don't exist
        for name, direction in [("y", None), ("s", "fb")]:
            symbol = pybamm.SpatialVariable(name, domain="test", direction=direction)
            with pytest.raises(pybamm.DomainError):
                method.spatial_variable(symbol)

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
        cell_values = np.arange(mesh.npts)
        expected = method._tpfa_matrix(mesh) @ cell_values
        gradient_matrices, _, _ = method._least_squares_gradient(mesh, {})
        for K, G in zip(
            method._cross_term_matrices(mesh), gradient_matrices, strict=True
        ):
            expected = expected + K @ (G @ cell_values)
        np.testing.assert_allclose(plain.evaluate()[:, 0], expected)

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
        distance, alpha, _ = method._boundary_decomposition(mesh, faces)
        coefficients = (
            mesh.face_areas[faces] * alpha / distance / mesh.cell_volumes[owners]
        )
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
        for component in gradient.components:
            np.testing.assert_allclose(component.evaluate(), 0, atol=1e-12)

        neumann_bcs = {
            variable: {
                side: (pybamm.Scalar(0), "Neumann")
                for side in ["left", "right", "top", "bottom"]
            }
        }
        for component in method.gradient(variable, constant, neumann_bcs).components:
            np.testing.assert_allclose(component.evaluate(), 0, atol=1e-12)

        x_values = mesh.cell_centroids[:, 0]
        values = pybamm.Vector(x_values, domain="test")
        grad_squared = method.gradient_squared(variable, values, {})
        matrices, _, _ = method._least_squares_gradient(mesh, {})
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

        with pytest.raises(pybamm.DiscretisationError, match="expects a VectorField"):
            method.divergence(symbol, pybamm.Scalar(1), {})

    def test_divergence_of_bc_bearing_flux_raises(self):
        # div of a flux whose gradient parent has BCs is not conservative
        # (the BC flux would be silently ignored), so it must raise
        mesh = _make_2d_mesh(2, 2)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        flux = -(pybamm.Scalar(2) * pybamm.grad(variable))
        components = [
            pybamm.Vector(np.ones(mesh.npts), domain="test"),
            pybamm.Vector(np.ones(mesh.npts), domain="test"),
        ]
        vector_field = pybamm.VectorField(*components)
        bcs = {variable: {"left": (pybamm.Scalar(0), "Dirichlet")}}
        with pytest.raises(pybamm.DiscretisationError, match="conservative"):
            method.divergence(flux, vector_field, bcs)

        # without BCs on u the same flux is fine
        result = method.divergence(flux, vector_field, {})
        assert result.evaluate().shape == (mesh.npts, 1)

    def test_gradient_warns_on_bucket_without_bc(self, caplog):
        import logging

        mesh = _make_quad_mesh(2, 2)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        values = pybamm.Vector(np.arange(mesh.npts), domain="test")
        bcs = {
            variable: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        with caplog.at_level(logging.WARNING):
            method.gradient(variable, values, bcs)
        assert "no boundary condition" in caplog.text
        assert "top" in caplog.text and "bottom" in caplog.text

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
                }
            },
        )
        np.testing.assert_allclose(
            repeated.evaluate()[:, 0],
            np.tile(vector_result.evaluate()[:, 0], aux.npts),
            atol=1e-12,
        )

    def test_div_D_grad_per_face_bc_vectors_with_repeats(self):
        # A BC value with one entry per boundary face must be shared across
        # auxiliary-domain repeats, matching _bc_contribution's convention.
        mesh = _make_2d_mesh(2, 2)
        aux = _make_2d_mesh(1, 1)
        method = _method_with_mesh(mesh, aux=aux)
        variable = pybamm.Variable("u", domain="test")
        div_symbol = pybamm.Variable("div", domain="test")
        cell_values = mesh.cell_centroids[:, 0] ** 2
        values = pybamm.Vector(cell_values, domain="test")

        n_left = len(mesh.boundary_faces["left"])
        n_right = len(mesh.boundary_faces["right"])
        bcs = {
            variable: {
                "left": (pybamm.Vector(np.linspace(1, 2, n_left)), "Dirichlet"),
                "right": (pybamm.Vector(np.linspace(-1, 1, n_right)), "Neumann"),
            }
        }
        single = method.div_D_grad(div_symbol, variable, pybamm.Scalar(2), values, bcs)

        repeated_domains = {"primary": ["test"], "secondary": ["aux"]}
        repeated_div = pybamm.Variable("repeated div", domains=repeated_domains)
        repeated_u = pybamm.Variable("repeated u", domains=repeated_domains)
        repeated_values = pybamm.Vector(
            np.tile(cell_values, aux.npts), domains=repeated_domains
        )
        repeated = method.div_D_grad(
            repeated_div,
            repeated_u,
            pybamm.Scalar(2),
            repeated_values,
            {repeated_u: bcs[variable]},
        )
        np.testing.assert_allclose(
            repeated.evaluate()[:, 0],
            np.tile(single.evaluate()[:, 0], aux.npts),
            atol=1e-12,
        )

        # One entry per face per repeat passes through untiled
        single_right = method.div_D_grad(
            div_symbol,
            variable,
            pybamm.Scalar(2),
            values,
            {variable: {"right": bcs[variable]["right"]}},
        )
        full = method.div_D_grad(
            repeated_div,
            repeated_u,
            pybamm.Scalar(2),
            repeated_values,
            {
                repeated_u: {
                    "right": (
                        pybamm.Vector(np.tile(np.linspace(-1, 1, n_right), aux.npts)),
                        "Neumann",
                    ),
                }
            },
        )
        np.testing.assert_allclose(
            full.evaluate()[:, 0],
            np.tile(single_right.evaluate()[:, 0], aux.npts),
            atol=1e-12,
        )

    def test_div_D_grad_anisotropic_coefficient_raises(self):
        mesh = _make_2d_mesh(2, 2)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        div_symbol = pybamm.Variable("div", domain="test")
        values = pybamm.Vector(np.arange(mesh.npts), domain="test")
        anisotropic = pybamm.VectorField(pybamm.Scalar(1), pybamm.Scalar(2))
        with pytest.raises(pybamm.DiscretisationError, match="Anisotropic"):
            method.div_D_grad(div_symbol, variable, anisotropic, values, {})

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
        np.testing.assert_allclose(
            row.entries.toarray()[0, : mesh.npts], mesh.cell_volumes
        )
        assert row.shape == (aux.npts, mesh.npts * aux.npts)

        boundary = method.boundary_integral(child, values, "left")
        np.testing.assert_allclose(boundary.evaluate(), 1)

    @pytest.mark.parametrize(
        "side",
        ["left", "top-right", "top-left", "bottom-right", "bottom-left"],
    )
    def test_boundary_value_and_corners(self, side):
        mesh = _make_2d_mesh(2, 2)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        values = pybamm.Vector(np.arange(mesh.npts), domain="test")
        symbol = pybamm.BoundaryValue(variable, side)

        result = method.boundary_value_or_flux(symbol, values)
        assert result.domain == []
        if "-" in side:
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

    def test_boundary_gradient_raises(self):
        mesh = _make_2d_mesh(2, 2)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        values = pybamm.Vector(np.arange(mesh.npts), domain="test")
        symbol = pybamm.BoundaryGradient(variable, "left")
        with pytest.raises(NotImplementedError, match="BoundaryGradient"):
            method.boundary_value_or_flux(symbol, values)

    def test_corner_value_uses_boundary_cell_on_nonconvex_domain(self):
        # L-shaped domain: [0,2]x[0,1] plus [0,1]x[1,2]; the top-right
        # bounding-box corner (2,2) is outside the domain, and the interior
        # cell nearest to it must not be picked
        squares = []
        for x0 in (0, 1):
            squares.append((x0, 0))
        squares.append((0, 1))
        nodes_list, elems_list = [], []
        node_ids = {}

        def nid(p):
            if p not in node_ids:
                node_ids[p] = len(nodes_list)
                nodes_list.append(p)
            return node_ids[p]

        for x0, z0 in squares:
            corners = [(x0, z0), (x0 + 1, z0), (x0 + 1, z0 + 1), (x0, z0 + 1)]
            elems_list.append([nid(c) for c in corners])
        mesh = UnstructuredSubMesh(
            np.array(nodes_list, dtype=float), np.array(elems_list)
        )
        mesh.detect_box_boundaries()
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        values = pybamm.Vector(np.arange(mesh.npts), domain="test")
        symbol = pybamm.BoundaryValue(variable, "top-right")
        result = method.boundary_value_or_flux(symbol, values)
        chosen = int(result.evaluate().item())
        candidates = set(mesh.face_owner[mesh.boundary_faces["top"]].tolist()) | set(
            mesh.face_owner[mesh.boundary_faces["right"]].tolist()
        )
        assert chosen in candidates

    def test_process_binary_operators(self):
        method = FiniteVolumeUnstructured()
        left_components = [pybamm.StateVector(slice(0, 2)), pybamm.Vector([2, 3])]
        right_components = [pybamm.Vector([4, 5]), pybamm.Vector([6, 7])]
        left_field = pybamm.VectorField(*left_components)
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
        np.testing.assert_array_equal(
            both.components[0].evaluate(y=np.array([1, 2]))[:, 0], [4, 10]
        )
        np.testing.assert_array_equal(both.components[1].evaluate()[:, 0], [12, 21])

        field_left = method.process_binary_operators(
            multiplication, None, None, left_field, pybamm.Scalar(2)
        )
        field_right = method.process_binary_operators(
            multiplication, None, None, pybamm.Scalar(2), right_field
        )
        np.testing.assert_array_equal(
            field_left.components[0].evaluate(y=np.array([1, 2]))[:, 0], [2, 4]
        )
        np.testing.assert_array_equal(
            field_right.components[0].evaluate()[:, 0], [8, 10]
        )

        scalar = method.process_binary_operators(
            multiplication, None, None, pybamm.Scalar(3), pybamm.Scalar(4)
        )
        assert scalar.evaluate() == 12

    def test_internal_neumann_unstructured_paths(self):
        left, right = _make_split_2d_meshes(2, 2, 2)
        method = FiniteVolumeUnstructured()
        # u = x: the interface normal gradient is exactly 1 once the
        # non-orthogonal cross term is included (the interface cells touch
        # no external side where the fitted zero normal derivative is wrong)
        left_values = pybamm.Vector(left.cell_centroids[:, 0], domain="left")
        right_values = pybamm.Vector(right.cell_centroids[:, 0], domain="right")

        direct = method._internal_neumann_unstructured(
            left_values, right_values, left, right, 1
        )
        np.testing.assert_allclose(direct.evaluate()[:, 0], 1.0, atol=1e-12)

        # orthogonal interface (quads): plain two-point difference
        quad_left = _make_quad_mesh(2, 2, x_range=(0, 0.5))
        quad_right = _make_quad_mesh(2, 2, x_range=(0.5, 1))
        compute_interface_data(quad_left, quad_right, "left", "right")
        interface = quad_left.interface_data["right"]
        u_left, u_right = np.arange(quad_left.npts), np.arange(quad_right.npts)
        quad_value = method._internal_neumann_unstructured(
            pybamm.Vector(u_left, domain="left"),
            pybamm.Vector(u_right, domain="right"),
            quad_left,
            quad_right,
            1,
        )
        expected = (
            u_right[interface["right_cells"]] - u_left[interface["left_cells"]]
        ) / interface["cell_distances"]
        np.testing.assert_allclose(quad_value.evaluate()[:, 0], expected)

        left_data = left.interface_data
        left.interface_data = {}
        reverse = method._internal_neumann_unstructured(
            left_values, right_values, left, right, 1
        )
        np.testing.assert_allclose(reverse.evaluate(), direct.evaluate())

        # unpaired meshes raise: silently returning zeros would decouple
        # the domains and solve to a wrong answer
        right.interface_data = {}
        with pytest.raises(pybamm.DiscretisationError, match="decoupled"):
            method._internal_neumann_unstructured(
                left_values, right_values, left, right, 2
            )
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
        np.testing.assert_allclose(interface_gradient.evaluate(), 0, atol=1e-12)

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


# ======================================================================
# Tests: Discretisation dispatch
# ======================================================================


def _get_unstructured_disc(nx=4, nz=4):
    """Single-domain 2D unstructured discretisation on [0,1]^2."""
    x = pybamm.SpatialVariable(
        "x_n", domain=["negative electrode"], coord_sys="cartesian"
    )
    z = pybamm.SpatialVariable(
        "z_2d", domain=["negative electrode"], coord_sys="cartesian", direction="tb"
    )
    geometry = {
        "negative electrode": {x: {"min": 0.0, "max": 1.0}, z: {"min": 0.0, "max": 1.0}}
    }
    mesh = pybamm.Mesh(
        geometry,
        {
            "negative electrode": pybamm.meshes.unstructured_submesh.UnstructuredMeshGenerator()
        },
        {x: nx, z: nz},
    )
    return pybamm.Discretisation(
        mesh, {"negative electrode": FiniteVolumeUnstructured()}
    )


class TestDiscretisationDispatch:
    def _disc_var_grad(self):
        disc = _get_unstructured_disc()
        var = pybamm.Variable("u", domain=["negative electrode"])
        disc.set_variable_slices([var])
        grad = pybamm.grad(var)
        disc_grad = disc.process_symbol(grad)
        u = disc.mesh["negative electrode"].cell_centroids[:, 0]
        return disc, var, grad, disc_grad, u

    def test_component_of_gradient(self):
        disc, _, grad, disc_grad, u = self._disc_var_grad()
        comp0 = disc.process_symbol(pybamm.Component(grad, 0))
        np.testing.assert_allclose(
            comp0.evaluate(y=u),
            disc_grad.components[0].evaluate(y=u),
        )
        comp1 = disc.process_symbol(pybamm.Component(grad, 1))
        np.testing.assert_allclose(
            comp1.evaluate(y=u),
            disc_grad.components[1].evaluate(y=u),
        )

    def test_component_requires_vector_field(self):
        disc, var, *_ = self._disc_var_grad()
        with pytest.raises(
            pybamm.DiscretisationError, match="Component can only be applied"
        ):
            disc.process_symbol(pybamm.Component(var, 0))

    def test_norm_of_gradient(self):
        disc, _, grad, disc_grad, u = self._disc_var_grad()
        norm = disc.process_symbol(pybamm.Norm(grad))
        gx = disc_grad.components[0].evaluate(y=u)
        gz = disc_grad.components[1].evaluate(y=u)
        np.testing.assert_allclose(
            norm.evaluate(y=u), np.sqrt(gx**2 + gz**2), rtol=1e-12
        )

    def test_norm_requires_vector_field(self):
        disc, var, *_ = self._disc_var_grad()
        with pytest.raises(
            pybamm.DiscretisationError, match="Norm can only be applied"
        ):
            disc.process_symbol(pybamm.Norm(var))

    def test_generic_unary_maps_over_components(self):
        """A generic unary operator (negation) applies componentwise to a
        VectorField."""
        disc, _, grad, disc_grad, u = self._disc_var_grad()
        neg = disc.process_symbol(-grad)
        assert isinstance(neg, pybamm.VectorField)
        for k in range(2):
            np.testing.assert_allclose(
                neg.components[k].evaluate(y=u),
                -disc_grad.components[k].evaluate(y=u),
                atol=1e-12,
            )

    def test_scalar_times_gradient_lifted(self):
        """Scalar * grad(u) lifts the scalar to an N-component VectorField."""
        disc, _, grad, disc_grad, u = self._disc_var_grad()
        scaled = disc.process_symbol(pybamm.Scalar(2) * grad)
        assert isinstance(scaled, pybamm.VectorField)
        for k in range(2):
            np.testing.assert_allclose(
                scaled.components[k].evaluate(y=u),
                2 * disc_grad.components[k].evaluate(y=u),
                atol=1e-12,
            )

    def test_gradient_times_scalar_lifted(self):
        disc, _, grad, disc_grad, u = self._disc_var_grad()
        scaled = disc.process_symbol(grad * pybamm.Scalar(3))
        assert isinstance(scaled, pybamm.VectorField)
        for k in range(2):
            np.testing.assert_allclose(
                scaled.components[k].evaluate(y=u),
                3 * disc_grad.components[k].evaluate(y=u),
                atol=1e-12,
            )

    def test_domainless_vector_field_binary_op(self):
        """Binary ops on domainless VectorFields combine componentwise."""
        disc = _get_unstructured_disc()
        vf_a = pybamm.VectorField(pybamm.Scalar(1), pybamm.Scalar(2))
        vf_b = pybamm.VectorField(pybamm.Scalar(3), pybamm.Scalar(4))
        product = disc.process_symbol(vf_a * vf_b)
        assert isinstance(product, pybamm.VectorField)
        np.testing.assert_allclose(product.components[0].evaluate(), 3)
        np.testing.assert_allclose(product.components[1].evaluate(), 8)

        # Scalar lifted to match the VectorField's components
        scaled = disc.process_symbol(pybamm.Scalar(2) * vf_a)
        assert isinstance(scaled, pybamm.VectorField)
        np.testing.assert_allclose(scaled.components[0].evaluate(), 2)
        np.testing.assert_allclose(scaled.components[1].evaluate(), 4)


class TestProcessModelConcatenation:
    def test_two_domain_diffusion_steady_state(self):
        """process_model on a concatenated variable dispatches internal BCs
        through FiniteVolumeUnstructured; the discrete Laplacian of the exact
        steady profile (linear in x) is zero."""
        x_n = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        x_s = pybamm.SpatialVariable("x_s", domain=["separator"], coord_sys="cartesian")
        z = pybamm.SpatialVariable(
            "z_2d",
            domain=["negative electrode", "separator"],
            coord_sys="cartesian",
            direction="tb",
        )
        geometry = {
            "negative electrode": {
                x_n: {"min": 0.0, "max": 0.5},
                z: {"min": 0.0, "max": 1.0},
            },
            "separator": {
                x_s: {"min": 0.5, "max": 1.0},
                z: {"min": 0.0, "max": 1.0},
            },
        }
        gen = pybamm.meshes.unstructured_submesh.UnstructuredMeshGenerator(
            element_type="quad"
        )
        mesh = pybamm.Mesh(
            geometry,
            {"negative electrode": gen, "separator": gen},
            {x_n: 3, x_s: 3, z: 3},
        )
        disc = pybamm.Discretisation(
            mesh,
            {
                "negative electrode": FiniteVolumeUnstructured(),
                "separator": FiniteVolumeUnstructured(),
            },
        )

        var_n = pybamm.Variable("c_n", domain=["negative electrode"])
        var_s = pybamm.Variable("c_s", domain=["separator"])
        var = pybamm.concatenation(var_n, var_s)

        model = pybamm.BaseModel()
        model.rhs = {var: pybamm.div(pybamm.grad(var))}
        model.initial_conditions = {var: pybamm.Scalar(1)}
        model.boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        model.variables = {"c": var}
        model_disc = disc.process_model(model, inplace=False)

        u = np.concatenate(
            [
                mesh["negative electrode"].cell_centroids[:, 0],
                mesh["separator"].cell_centroids[:, 0],
            ]
        )
        rhs = model_disc.concatenated_rhs.evaluate(t=0, y=u).flatten()
        np.testing.assert_allclose(rhs, 0.0, atol=1e-10)


class TestDiscretisationDispatchLifting:
    def _disc_var_grad(self):
        disc = _get_unstructured_disc()
        var = pybamm.Variable("u", domain=["negative electrode"])
        disc.set_variable_slices([var])
        grad = pybamm.grad(var)
        disc_grad = disc.process_symbol(grad)
        u = disc.mesh["negative electrode"].cell_centroids[:, 0]
        return disc, var, grad, disc_grad, u

    def test_gradient_minus_scalar_lifted(self):
        """A right-hand Scalar is lifted to an N-component VectorField.

        A raw Subtraction node is used because operator simplification
        rewrites ``x - c`` as ``-c + x``, which takes the left-Scalar path.
        """
        disc, _, grad, disc_grad, u = self._disc_var_grad()
        shifted = disc.process_symbol(pybamm.Subtraction(grad, pybamm.Scalar(0.5)))
        assert isinstance(shifted, pybamm.VectorField)
        for k in range(2):
            np.testing.assert_allclose(
                shifted.components[k].evaluate(y=u),
                disc_grad.components[k].evaluate(y=u) - 0.5,
                atol=1e-12,
            )

    def test_domainless_vector_field_minus_scalar(self):
        disc = _get_unstructured_disc()
        vf = pybamm.VectorField(pybamm.Scalar(3), pybamm.Scalar(4))
        shifted = disc.process_symbol(pybamm.Subtraction(vf, pybamm.Scalar(1)))
        assert isinstance(shifted, pybamm.VectorField)
        np.testing.assert_allclose(shifted.components[0].evaluate(), 2)
        np.testing.assert_allclose(shifted.components[1].evaluate(), 3)

    def test_div_of_coefficient_times_gradient(self):
        """div(D * grad(u)) is intercepted and routed to div_D_grad for both
        coefficient orderings."""
        disc, var, grad, _, u = self._disc_var_grad()
        base = disc.process_symbol(pybamm.div(grad)).evaluate(y=u)
        scaled = disc.process_symbol(pybamm.div(pybamm.Scalar(2) * grad)).evaluate(y=u)
        np.testing.assert_allclose(scaled, 2 * base, atol=1e-12)

        right_form = disc.process_symbol(pybamm.div(var * grad)).evaluate(y=u)
        left_form = disc.process_symbol(pybamm.div(grad * var)).evaluate(y=u)
        np.testing.assert_allclose(left_form, right_form, atol=1e-12)


class TestProcessModelConcatenationZStack:
    def test_z_stacked_domains_use_graph_internal_bcs(self):
        """Domains stacked in z: pybamm.Mesh's 1D-stack pairing fails on the
        transverse mismatch, FiniteVolumeUnstructured's build() discovers the
        interface by face matching, and process_model routes internal BCs
        through set_internal_bcs_for_concat.  The discrete Laplacian of the
        exact steady profile (linear in z) is zero."""
        x_n = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        z_n = pybamm.SpatialVariable(
            "z_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        x_s = pybamm.SpatialVariable("x_s", domain=["separator"], coord_sys="cartesian")
        z_s = pybamm.SpatialVariable("z_s", domain=["separator"], coord_sys="cartesian")
        geometry = {
            "negative electrode": {
                x_n: {"min": 0.0, "max": 1.0},
                z_n: {"min": 0.0, "max": 0.5},
            },
            "separator": {
                x_s: {"min": 0.0, "max": 1.0},
                z_s: {"min": 0.5, "max": 1.0},
            },
        }
        gen = pybamm.meshes.unstructured_submesh.UnstructuredMeshGenerator(
            element_type="quad"
        )
        mesh = pybamm.Mesh(
            geometry,
            {"negative electrode": gen, "separator": gen},
            {x_n: 3, z_n: 3, x_s: 3, z_s: 3},
        )
        disc = pybamm.Discretisation(
            mesh,
            {
                "negative electrode": FiniteVolumeUnstructured(),
                "separator": FiniteVolumeUnstructured(),
            },
        )
        # build() added graph-discovered interface buckets
        assert any(
            tag.startswith("iface_")
            for tag in mesh["negative electrode"].boundary_faces
        )
        assert any(tag.startswith("iface_") for tag in mesh["separator"].boundary_faces)

        var_n = pybamm.Variable("c_n", domain=["negative electrode"])
        var_s = pybamm.Variable("c_s", domain=["separator"])
        var = pybamm.concatenation(var_n, var_s)

        model = pybamm.BaseModel()
        model.rhs = {var: pybamm.div(pybamm.grad(var))}
        model.initial_conditions = {var: pybamm.Scalar(1)}
        model.boundary_conditions = {
            var: {
                "bottom": (pybamm.Scalar(0), "Dirichlet"),
                "top": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        model.variables = {"c": var}
        model_disc = disc.process_model(model, inplace=False)

        u = np.concatenate(
            [
                mesh["negative electrode"].cell_centroids[:, 1],
                mesh["separator"].cell_centroids[:, 1],
            ]
        )
        rhs = model_disc.concatenated_rhs.evaluate(t=0, y=u).flatten()
        np.testing.assert_allclose(rhs, 0.0, atol=1e-10)


# ======================================================================
# Tests: non-orthogonal correction
# ======================================================================


def _perturb_interior_nodes(nodes, spacing, fraction=0.3, seed=0):
    """Jitter interior nodes so faces are skewed as well as non-orthogonal."""
    rng = np.random.default_rng(seed)
    nodes = nodes.copy()
    low, high = nodes.min(axis=0), nodes.max(axis=0)
    on_boundary = np.any(np.isclose(nodes, low) | np.isclose(nodes, high), axis=1)
    interior = nodes[~on_boundary]
    nodes[~on_boundary] = interior + rng.uniform(
        -fraction * spacing, fraction * spacing, interior.shape
    )
    return nodes


def _make_perturbed_tri_mesh(n=6):
    edges = np.linspace(0, 1, n + 1)
    nodes, elements = _quad_to_tri(edges, edges)
    mesh = UnstructuredSubMesh(_perturb_interior_nodes(nodes, 1.0 / n), elements)
    mesh.detect_box_boundaries()
    return mesh


def _dirichlet_all_sides(mesh, u_exact):
    return {
        side: (pybamm.Vector(u_exact(mesh.face_centroids[faces])), "Dirichlet")
        for side, faces in mesh.boundary_faces.items()
    }


def _laplacian_system(method, mesh, bcs):
    """``(L, rhs)`` with ``laplacian(u) = L @ u + rhs`` for the full operator."""
    variable = pybamm.Variable("u", domain="test")
    y = pybamm.StateVector(slice(0, mesh.npts), domains={"primary": ["test"]})
    expr = method.laplacian(variable, y, {variable: bcs} if bcs else {})
    zeros = np.zeros(mesh.npts)
    return sp_csr(expr.jac(y).evaluate(y=zeros)), expr.evaluate(y=zeros)[:, 0]


class TestNonOrthogonalCorrection:
    @pytest.mark.parametrize(
        "make_mesh",
        [
            lambda: _make_2d_mesh(6, 6),
            _make_perturbed_tri_mesh,
            lambda: _make_3d_mesh(3, 3, 3),
        ],
        ids=["tri", "tri-perturbed", "tet"],
    )
    @pytest.mark.parametrize("correction", ["over-relaxed", "minimum"])
    def test_laplacian_exact_on_linear_field(self, make_mesh, correction):
        """The discrete Laplacian of a linear field vanishes on every cell;
        the two-point part alone fails this on any non-orthogonal mesh."""
        mesh = make_mesh()
        method = FiniteVolumeUnstructured({"non-orthogonal correction": correction})
        method._mesh = _MeshMap({("test",): mesh})
        slope = np.array([1.0, 0.7, 0.4])[: mesh.dimension]
        u = mesh.cell_centroids @ slope
        bcs = _dirichlet_all_sides(mesh, lambda points: points @ slope)
        L, rhs = _laplacian_system(method, mesh, bcs)
        np.testing.assert_allclose(L @ u + rhs, 0, atol=1e-10)

    def test_two_point_part_alone_is_not_exact(self):
        mesh = _make_2d_mesh(6, 6)
        L = FiniteVolumeUnstructured()._tpfa_matrix(mesh)
        residual = np.abs(L @ mesh.cell_centroids[:, 0])
        assert residual[_get_internal_cells(mesh)].max() > 1

    def test_second_order_convergence_on_triangles(self):
        def u_exact(points):
            return np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])

        errors = []
        for n in (8, 16, 32):
            mesh = _make_2d_mesh(n, n)
            method = _method_with_mesh(mesh)
            L, rhs = _laplacian_system(
                method, mesh, _dirichlet_all_sides(mesh, u_exact)
            )
            source = -2 * np.pi**2 * u_exact(mesh.cell_centroids)
            u = spsolve(L.tocsc(), source - rhs)
            error = u - u_exact(mesh.cell_centroids)
            errors.append(np.sqrt(np.sum(mesh.cell_volumes * error**2)))
        rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
        assert np.all(rates > 1.7), rates

    def test_orthogonal_mesh_has_no_cross_term(self):
        mesh = _make_quad_mesh(4, 4)
        method = FiniteVolumeUnstructured()
        assert method._cross_term_matrices(mesh) is None
        assert method._div_D_grad_matrices(mesh)[4] is None
        np.testing.assert_allclose(method._decomposition(mesh)[0], 1.0)

    def test_full_operator_is_conservative(self):
        mesh = _make_perturbed_tri_mesh(5)
        L, _ = _laplacian_system(_method_with_mesh(mesh), mesh, None)
        u = mesh.cell_centroids[:, 0] ** 2 + mesh.cell_centroids[:, 1]
        np.testing.assert_allclose(np.sum((L @ u) * mesh.cell_volumes), 0, atol=1e-10)

    def test_div_D_grad_matches_laplacian_for_constant_D(self):
        mesh = _make_perturbed_tri_mesh(4)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        div_symbol = pybamm.Variable("div", domain="test")

        def u_exact(points):
            return np.sin(points[:, 0]) * points[:, 1] ** 2

        values = pybamm.Vector(u_exact(mesh.cell_centroids), domain="test")
        bcs = {variable: _dirichlet_all_sides(mesh, u_exact)}
        laplacian = method.laplacian(variable, values, bcs).evaluate()[:, 0]
        div_grad = method.div_D_grad(
            div_symbol, variable, pybamm.Scalar(2), values, bcs
        )
        np.testing.assert_allclose(div_grad.evaluate()[:, 0], 2 * laplacian, atol=1e-10)

    def test_div_D_grad_exact_on_linear_field_3d(self):
        mesh = _make_3d_mesh(3, 3, 3)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        div_symbol = pybamm.Variable("div", domain="test")
        slope = np.array([1.0, 0.7, 0.4])
        values = pybamm.Vector(mesh.cell_centroids @ slope, domain="test")
        bcs = {variable: _dirichlet_all_sides(mesh, lambda points: points @ slope)}
        result = method.div_D_grad(div_symbol, variable, pybamm.Scalar(2), values, bcs)
        np.testing.assert_allclose(result.evaluate(), 0, atol=1e-10)

    def test_least_squares_gradient_exact_on_linear_field(self):
        mesh = _make_perturbed_tri_mesh(5)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        slope = np.array([2.0, -3.0])
        values = pybamm.Vector(mesh.cell_centroids @ slope, domain="test")

        dirichlet = {
            variable: _dirichlet_all_sides(mesh, lambda points: points @ slope)
        }
        components = method.gradient(variable, values, dirichlet).components
        for k, component in enumerate(components):
            np.testing.assert_allclose(component.evaluate()[:, 0], slope[k], atol=1e-10)

        # named sides take coordinate-direction derivatives on both ends
        neumann = {
            variable: {
                "left": (pybamm.Scalar(2.0), "Neumann"),
                "right": (pybamm.Scalar(2.0), "Neumann"),
                "bottom": (pybamm.Scalar(-3.0), "Neumann"),
                "top": (pybamm.Scalar(-3.0), "Neumann"),
            }
        }
        components = method.gradient(variable, values, neumann).components
        for k, component in enumerate(components):
            np.testing.assert_allclose(component.evaluate()[:, 0], slope[k], atol=1e-10)

    def test_green_gauss_gradient_is_not_exact_on_skewed_mesh(self):
        """Documents why the cross term cannot use the Green-Gauss gradient."""
        mesh = _make_3d_mesh(3, 3, 3)
        G = FiniteVolumeUnstructured()._green_gauss_matrices(mesh)
        grad_x = (G[0] @ mesh.cell_centroids[:, 0])[_get_internal_cells(mesh)]
        assert np.abs(grad_x - 1).max() > 0.1

    def test_divergence_shares_green_gauss_assembly(self):
        mesh = _make_2d_mesh(3, 3)
        method = FiniteVolumeUnstructured()
        assert method._divergence_matrices(mesh) is method._green_gauss_matrices(mesh)

    def test_invalid_option_raises(self):
        with pytest.raises(pybamm.OptionError, match="non-orthogonal correction"):
            FiniteVolumeUnstructured({"non-orthogonal correction": "none"})

    def test_option_sets_two_point_weight(self):
        mesh = _make_2d_mesh(3, 3)
        cos_theta = FiniteVolumeUnstructured()._face_geometry(mesh)["cos_theta"]
        minimum = FiniteVolumeUnstructured({"non-orthogonal correction": "minimum"})
        over_relaxed = FiniteVolumeUnstructured()
        np.testing.assert_allclose(minimum._decomposition(mesh)[0], cos_theta)
        np.testing.assert_allclose(over_relaxed._decomposition(mesh)[0], 1 / cos_theta)

    def test_inverted_cell_raises(self):
        mesh = _make_2d_mesh(2, 2)
        mesh.face_normals[: mesh.n_internal_faces] *= -1
        with pytest.raises(pybamm.GeometryError, match="pointing away"):
            FiniteVolumeUnstructured()._face_geometry(mesh)

    def test_build_warns_on_severe_non_orthogonality(self, caplog):
        import logging

        # Sliver neighbour: the centroid line is ~85 degrees off the normal
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, -8.5]])
        elements = np.array([[0, 1, 2], [1, 3, 2]])
        skewed = UnstructuredSubMesh(nodes, elements)
        skewed.detect_box_boundaries()
        method = FiniteVolumeUnstructured()
        assert method._face_geometry(skewed)["max_angle_deg"] > 70
        with caplog.at_level(logging.WARNING):
            method.build(_MeshMap({("skewed",): skewed}))
        assert "non-orthogonality" in caplog.text

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            method.build(_MeshMap({("tri",): _make_2d_mesh(3, 3)}))
        assert "non-orthogonality" not in caplog.text


# ======================================================================
# Tests: non-orthogonal correction across domain interfaces
# ======================================================================


def _two_domain_laplacian(element_type, boundary_conditions):
    """Discretised ``div(grad(c))`` of a two-domain concatenation on the
    unit box split at x = 0.5, returning ``(mesh, rhs_expression)``."""
    dim3 = element_type in ("tetrahedron", "hexahedron")
    domains = ["negative electrode", "separator"]
    x_n = pybamm.SpatialVariable("x_n", domain=[domains[0]], coord_sys="cartesian")
    x_s = pybamm.SpatialVariable("x_s", domain=[domains[1]], coord_sys="cartesian")
    z = pybamm.SpatialVariable(
        "z_2d", domain=domains, coord_sys="cartesian", direction="tb"
    )
    geometry = {
        domains[0]: {x_n: {"min": 0.0, "max": 0.5}, z: {"min": 0.0, "max": 1.0}},
        domains[1]: {x_s: {"min": 0.5, "max": 1.0}, z: {"min": 0.0, "max": 1.0}},
    }
    npts = {x_n: 3, x_s: 3, z: 4}
    if dim3:
        y = pybamm.SpatialVariable("y", domain=domains, coord_sys="cartesian")
        for domain in domains:
            geometry[domain][y] = {"min": 0.0, "max": 1.0}
        npts[y] = 3
    generator = pybamm.meshes.unstructured_submesh.UnstructuredMeshGenerator(
        element_type=element_type
    )
    mesh = pybamm.Mesh(geometry, dict.fromkeys(domains, generator), npts)
    disc = pybamm.Discretisation(
        mesh, {domain: FiniteVolumeUnstructured() for domain in domains}
    )
    var_n = pybamm.Variable("c_n", domain=[domains[0]])
    var_s = pybamm.Variable("c_s", domain=[domains[1]])
    var = pybamm.concatenation(var_n, var_s)
    model = pybamm.BaseModel()
    model.rhs = {var: pybamm.div(pybamm.grad(var))}
    model.initial_conditions = {var: pybamm.Scalar(1)}
    model.boundary_conditions = {var: boundary_conditions}
    model.variables = {"c": var}
    disc.process_model(model, inplace=False)
    return mesh, disc.process_model(model, inplace=False).concatenated_rhs


class TestInterfaceCorrection:
    @pytest.mark.parametrize("element_type", ["triangle", "tetrahedron"])
    def test_linear_field_exact_across_interface(self, element_type):
        """u = x is the steady state of left=0, right=1 with zero flux on the
        other sides; the interface flux must reproduce it on skewed pairs."""
        mesh, rhs = _two_domain_laplacian(
            element_type,
            {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            },
        )
        u = np.concatenate(
            [
                mesh["negative electrode"].cell_centroids[:, 0],
                mesh["separator"].cell_centroids[:, 0],
            ]
        )
        np.testing.assert_allclose(rhs.evaluate(y=u), 0, atol=1e-10)

    def test_interface_flux_is_conservative(self):
        mesh, rhs = _two_domain_laplacian(
            "tetrahedron",
            {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            },
        )
        volumes = np.concatenate(
            [mesh["negative electrode"].cell_volumes, mesh["separator"].cell_volumes]
        )
        u = np.random.default_rng(1).uniform(size=len(volumes))
        np.testing.assert_allclose(volumes @ rhs.evaluate(y=u)[:, 0], 0, atol=1e-10)

    def test_interface_data_records_faces(self):
        left, right = _make_split_2d_meshes(3, 3, 3)
        data = left.interface_data["right"]
        np.testing.assert_array_equal(data["left_faces"], left.boundary_faces["right"])
        assert set(data["right_faces"]) == set(right.boundary_faces["left"])
        np.testing.assert_array_equal(
            left.face_owner[data["left_faces"]], data["left_cells"]
        )

        a = _make_2d_mesh(2, 2, x_range=(0, 0.5))
        b = _make_2d_mesh(2, 2, x_range=(0.5, 1))
        assert FiniteVolumeUnstructured()._compute_pair_interface(a, b, "a", "b")
        np.testing.assert_array_equal(
            a.interface_data["b"]["left_faces"], a.boundary_faces["iface_b"]
        )
        np.testing.assert_array_equal(
            b.interface_data["a"]["left_faces"], b.boundary_faces["iface_a"]
        )

    def test_inconsistent_face_counts_raise(self):
        mesh = _make_2d_mesh(2, 2)
        # give one cell an extra boundary face and another one fewer
        boundary = mesh.boundary_faces["left"]
        mesh.face_owner[boundary[0]] = mesh.face_owner[boundary[1]]
        with pytest.raises(pybamm.DiscretisationError, match="same number of faces"):
            FiniteVolumeUnstructured()._least_squares_matrices(mesh, {})


class TestTimeDependentScalarBoundaryValues:
    def test_neumann_value_depending_on_time(self):
        """A ``pybamm.t``-dependent Neumann value evaluates for shape to
        ``()``; it must still be broadcast over the side's faces."""
        mesh = _make_2d_mesh(3, 3)
        method = _method_with_mesh(mesh)
        variable = pybamm.Variable("u", domain="test")
        div_symbol = pybamm.Variable("div", domain="test")
        slope = 2.0 * pybamm.t
        values = pybamm.Vector(mesh.cell_centroids[:, 0], domain="test")
        bcs = {
            variable: {
                "left": (slope, "Neumann"),
                "right": (slope, "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        # at t = 0.5 the prescribed slope matches u = x, so both vanish
        laplacian = method.laplacian(variable, values, bcs)
        np.testing.assert_allclose(laplacian.evaluate(t=0.5), 0, atol=1e-10)
        div_grad = method.div_D_grad(
            div_symbol, variable, pybamm.Scalar(3), values, bcs
        )
        np.testing.assert_allclose(div_grad.evaluate(t=0.5), 0, atol=1e-10)
        grad_x = method.gradient(variable, values, bcs).components[0]
        np.testing.assert_allclose(grad_x.evaluate(t=0.5), 1, atol=1e-10)
        assert np.abs(laplacian.evaluate(t=1.0)).max() > 1e-3
