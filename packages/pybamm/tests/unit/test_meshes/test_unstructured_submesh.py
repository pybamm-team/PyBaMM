import numpy as np

import pybamm
from pybamm.meshes.unstructured_submesh import (
    TaggedSubMeshGenerator,
    UnstructuredMeshGenerator,
    UnstructuredSubMesh,
    UserSuppliedUnstructuredMesh,
    _hex_grid,
    _hex_to_tet,
    _quad_to_tri,
    compute_interface_data,
)

# ======================================================================
# Helpers
# ======================================================================


def _unit_square_two_triangles():
    """Unit square [0,1]x[0,1] split into 2 triangles."""
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    elements = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    return nodes, elements


def _unit_cube_five_tets():
    """Unit cube [0,1]^3 split into 5 tets (pattern A)."""
    nodes = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    elements = np.array(
        [
            [0, 1, 2, 5],
            [0, 2, 3, 7],
            [0, 5, 7, 4],
            [2, 5, 7, 6],
            [0, 2, 5, 7],
        ],
        dtype=int,
    )
    return nodes, elements


# ======================================================================
# TestUnstructuredSubMesh
# ======================================================================


class TestUnstructuredSubMesh:
    def test_2d_single_triangle(self):
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        elements = np.array([[0, 1, 2]], dtype=int)

        mesh = UnstructuredSubMesh(nodes, elements)

        assert mesh.npts == 1
        assert mesh.dimension == 2
        np.testing.assert_allclose(mesh.cell_volumes, [0.5])
        np.testing.assert_allclose(mesh.cell_centroids, [[1 / 3, 1 / 3]])
        assert mesh.n_internal_faces == 0
        assert len(mesh.faces) == 3

    def test_2d_two_triangles(self):
        nodes, elements = _unit_square_two_triangles()
        mesh = UnstructuredSubMesh(nodes, elements)

        assert mesh.npts == 2
        assert mesh.dimension == 2
        assert mesh.n_internal_faces == 1
        # 4 boundary edges + 1 internal = 5 total
        assert len(mesh.faces) == 5

        # Owner and neighbor of internal face
        owner = mesh.face_owner[0]
        neighbor = mesh.face_neighbor[0]
        assert owner != neighbor
        assert {owner, neighbor} == {0, 1}

    def test_2d_cell_volumes(self):
        nodes, elements = _unit_square_two_triangles()
        mesh = UnstructuredSubMesh(nodes, elements)

        np.testing.assert_allclose(mesh.cell_volumes, [0.5, 0.5])
        np.testing.assert_allclose(mesh.cell_volumes.sum(), 1.0)

    def test_2d_face_normals_orientation(self):
        """All normals should point outward from the owner cell."""
        nodes, elements = _unit_square_two_triangles()
        mesh = UnstructuredSubMesh(nodes, elements)

        for f in range(len(mesh.faces)):
            owner_centroid = mesh.cell_centroids[mesh.face_owner[f]]
            to_face = mesh.face_centroids[f] - owner_centroid
            dot = np.dot(mesh.face_normals[f], to_face)
            assert dot >= -1e-14, f"Face {f}: normal not outward (dot={dot})"

    def test_2d_boundary_face_identification(self):
        x_edges = np.linspace(0, 2, 5)
        z_edges = np.linspace(0, 1, 4)
        nodes, elements = _quad_to_tri(x_edges, z_edges)
        mesh = UnstructuredSubMesh(nodes, elements)
        mesh.detect_box_boundaries()

        assert "left" in mesh.boundary_faces
        assert "right" in mesh.boundary_faces
        assert "bottom" in mesh.boundary_faces
        assert "top" in mesh.boundary_faces

        # All left boundary faces should have face centroid x ≈ 0
        left_centroids = mesh.face_centroids[mesh.boundary_faces["left"]]
        np.testing.assert_allclose(left_centroids[:, 0], 0.0, atol=1e-14)

        # All right boundary faces should have face centroid x ≈ 2
        right_centroids = mesh.face_centroids[mesh.boundary_faces["right"]]
        np.testing.assert_allclose(right_centroids[:, 0], 2.0, atol=1e-14)

    def test_3d_single_tet(self):
        nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        elements = np.array([[0, 1, 2, 3]], dtype=int)

        mesh = UnstructuredSubMesh(nodes, elements)

        assert mesh.npts == 1
        assert mesh.dimension == 3
        np.testing.assert_allclose(mesh.cell_volumes, [1 / 6])
        np.testing.assert_allclose(mesh.cell_centroids, [[0.25, 0.25, 0.25]])
        assert mesh.n_internal_faces == 0
        assert len(mesh.faces) == 4

    def test_3d_two_tets(self):
        # Two tets sharing a triangular face
        nodes = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float
        )
        elements = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=int)

        mesh = UnstructuredSubMesh(nodes, elements)

        assert mesh.npts == 2
        assert mesh.dimension == 3
        assert mesh.n_internal_faces == 1

        owner = mesh.face_owner[0]
        neighbor = mesh.face_neighbor[0]
        assert {owner, neighbor} == {0, 1}

    def test_3d_cell_volumes(self):
        nodes, elements = _unit_cube_five_tets()
        mesh = UnstructuredSubMesh(nodes, elements)

        np.testing.assert_allclose(mesh.cell_volumes.sum(), 1.0, atol=1e-14)

    def test_3d_face_areas(self):
        # Regular tet with edge length 1
        nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        elements = np.array([[0, 1, 2, 3]], dtype=int)
        mesh = UnstructuredSubMesh(nodes, elements)

        # 3 axis-aligned faces with area 0.5
        # 1 hypotenuse face with area sqrt(3)/2
        areas = np.sort(mesh.face_areas)
        np.testing.assert_allclose(areas[:3], 0.5, atol=1e-14)
        np.testing.assert_allclose(areas[3], np.sqrt(3) / 2, atol=1e-14)

    def test_3d_face_normals_orientation(self):
        nodes, elements = _unit_cube_five_tets()
        mesh = UnstructuredSubMesh(nodes, elements)

        for f in range(len(mesh.faces)):
            owner_centroid = mesh.cell_centroids[mesh.face_owner[f]]
            to_face = mesh.face_centroids[f] - owner_centroid
            dot = np.dot(mesh.face_normals[f], to_face)
            assert dot >= -1e-14, f"Face {f}: normal not outward (dot={dot})"

    def test_3d_boundary_face_identification(self):
        x_edges = np.linspace(0, 1, 3)
        y_edges = np.linspace(0, 1, 3)
        z_edges = np.linspace(0, 1, 3)
        nodes, elements = _hex_to_tet(x_edges, y_edges, z_edges)
        mesh = UnstructuredSubMesh(nodes, elements)
        mesh.detect_box_boundaries()

        for tag in ("left", "right", "front", "back", "bottom", "top"):
            assert tag in mesh.boundary_faces, f"Missing boundary tag '{tag}'"
            assert len(mesh.boundary_faces[tag]) > 0

    def test_custom_boundary_faces(self):
        nodes, elements = _unit_square_two_triangles()
        custom_bnd = {"my_boundary": np.array([3, 4])}
        mesh = UnstructuredSubMesh(nodes, elements, boundary_faces=custom_bnd)

        assert "my_boundary" in mesh.boundary_faces
        np.testing.assert_array_equal(mesh.boundary_faces["my_boundary"], [3, 4])

    def test_unsupported_element_raises(self):
        """2D cell with 5 verts, or 3D cell with 5 verts, should raise."""
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], dtype=float)
        elements = np.array([[0, 1, 2, 3, 4]], dtype=int)
        import pytest

        with pytest.raises(pybamm.GeometryError, match="Unsupported"):
            UnstructuredSubMesh(nodes, elements)

    def test_nonmanifold_face_raises(self):
        """A face shared by more than two cells must raise, not vanish."""
        import pytest

        # Three triangles all sharing the edge (0, 1): a non-manifold fan.
        nodes = np.array([[0, 0], [1, 0], [0.5, 1], [0.5, -1], [1.5, 0.5]], dtype=float)
        elements = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=int)
        with pytest.raises(pybamm.GeometryError, match="non-manifold"):
            UnstructuredSubMesh(nodes, elements)

    def test_2d_quad_mesh_basic(self):
        """Quadrilateral element type: geometry and connectivity."""
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        elements = np.array([[0, 1, 2, 3]], dtype=int)
        mesh = UnstructuredSubMesh(nodes, elements)
        mesh.detect_box_boundaries()

        assert mesh.element_type == "quad"
        np.testing.assert_allclose(mesh.cell_volumes, [1.0])
        # 4 boundary edges, no internal faces
        assert mesh.n_internal_faces == 0
        assert mesh._n_boundary_faces == 4
        # Standard boundary tags should be present
        assert "left" in mesh.boundary_faces
        assert "right" in mesh.boundary_faces
        assert "top" in mesh.boundary_faces
        assert "bottom" in mesh.boundary_faces

    def test_2d_quad_trapezoid_centroid_exact(self):
        """Quad centroids are geometric centroids, not vertex means.

        Symmetric trapezoid: the vertex mean gives y = 0.5, the true
        area centroid is y = 4/9.
        """
        nodes = np.array([[0, 0], [4, 0], [3, 1], [1, 1]], dtype=float)
        elements = np.array([[0, 1, 2, 3]], dtype=int)
        mesh = UnstructuredSubMesh(nodes, elements)

        np.testing.assert_allclose(mesh.cell_volumes, [3.0])
        np.testing.assert_allclose(mesh.cell_centroids, [[2.0, 4.0 / 9.0]])

    def test_3d_hex_frustum_centroid_exact(self):
        """Planar-faced but non-parallelogram hexes get exact centroids.

        Square frustum, base side 2 at z=0, top side 1 at z=1: all six
        faces planar (passes the warp check), volume 7/3, and the true
        centroid height is 11/28 — the vertex mean would give 1/2.
        The trapezoidal side faces likewise have exact area centroids.
        """
        nodes = np.array(
            [
                [-1, -1, 0],
                [1, -1, 0],
                [1, 1, 0],
                [-1, 1, 0],
                [-0.5, -0.5, 1],
                [0.5, -0.5, 1],
                [0.5, 0.5, 1],
                [-0.5, 0.5, 1],
            ],
            dtype=float,
        )
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
        mesh = UnstructuredSubMesh(nodes, elements)

        np.testing.assert_allclose(mesh.cell_volumes, [7.0 / 3.0])
        np.testing.assert_allclose(
            mesh.cell_centroids, [[0.0, 0.0, 11.0 / 28.0]], atol=1e-14
        )

        # Side face (outward normal toward -y) is a trapezoid in the plane
        # y = -1 + z/2: exact centroid (0, -7/9, 4/9)
        bnd_start = mesh._boundary_face_start
        front = bnd_start + int(np.argmin(mesh.face_normals[bnd_start:, 1]))
        np.testing.assert_allclose(
            mesh.face_centroids[front],
            [0.0, -7.0 / 9.0, 4.0 / 9.0],
            atol=1e-14,
        )

    def test_2d_quad_grid_two_cells(self):
        """Two adjacent quads share 1 internal face."""
        nodes = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=float)
        elements = np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=int)
        mesh = UnstructuredSubMesh(nodes, elements)

        assert mesh.element_type == "quad"
        assert mesh.n_internal_faces == 1
        np.testing.assert_allclose(mesh.cell_volumes, [1.0, 1.0])

    def test_3d_hex_mesh_basic(self):
        """Hexahedron element type: unit-cube volume and 6 boundary faces."""
        nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=float,
        )
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
        mesh = UnstructuredSubMesh(nodes, elements)

        assert mesh.element_type == "hexahedron"
        np.testing.assert_allclose(mesh.cell_volumes, [1.0])
        assert mesh.n_internal_faces == 0
        assert mesh._n_boundary_faces == 6
        np.testing.assert_allclose(mesh.face_areas, np.ones(6))

    def test_warped_hexahedron_raises(self):
        """A hex with a non-planar face is rejected at construction.

        Volumes of warped hexes depend on an arbitrary choice of face
        triangulation (the two diagonal splits of a warped unit-cube face
        differ by ~20%), so they are disallowed rather than approximated.
        """
        import pytest

        nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1.6, 1.6, 1.6],
                [0, 1, 1],
            ],
            dtype=float,
        )
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
        with pytest.raises(pybamm.GeometryError, match=r"non-planar"):
            UnstructuredSubMesh(nodes, elements)

        # Sub-tolerance jitter (well below 1e-8 relative) must still pass
        nodes_jitter = nodes.copy()
        nodes_jitter[6] = [1, 1, 1 + 1e-12]
        mesh = UnstructuredSubMesh(nodes_jitter, elements)
        np.testing.assert_allclose(mesh.cell_volumes, [1.0], rtol=1e-9)

    def test_2d_boundary_loops(self):
        """boundary_loops returns a matplotlib Path around the outer edge."""
        nodes, elements = _unit_square_two_triangles()
        mesh = UnstructuredSubMesh(nodes, elements)

        paths = mesh.boundary_loops()
        assert paths is not None
        assert len(paths) >= 1
        # Outer loop should contain the centre of the unit square
        assert paths[0].contains_point((0.5, 0.5))
        assert not paths[0].contains_point((-0.5, 0.5))

    def test_3d_boundary_loops_returns_none(self):
        """boundary_loops is 2D-only; 3D mesh returns None."""
        nodes, elements = _unit_cube_five_tets()
        mesh = UnstructuredSubMesh(nodes, elements)
        assert mesh.boundary_loops() is None

    def test_contains_points_3d_unit_cube(self):
        nodes, elements = _unit_cube_five_tets()
        mesh = UnstructuredSubMesh(nodes, elements)

        inside = np.array([[0.5, 0.5, 0.5]])
        outside = np.array([[2.0, 2.0, 2.0]])
        assert mesh.contains_points_3d(inside)[0]
        assert not mesh.contains_points_3d(outside)[0]

    def test_contains_points_3d_hex_mesh(self):
        """contains_points_3d on a hex mesh exercises the quad-face branch."""
        nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=float,
        )
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
        mesh = UnstructuredSubMesh(nodes, elements)

        assert mesh.contains_points_3d(np.array([[0.5, 0.5, 0.5]]))[0]
        assert not mesh.contains_points_3d(np.array([[2.0, 2.0, 2.0]]))[0]

    def test_optimize_ordering_single_cell_noop(self):
        """optimize_ordering with 1 cell returns without permuting."""
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        elements = np.array([[0, 1, 2]], dtype=int)
        mesh = UnstructuredSubMesh(nodes, elements)
        mesh.optimize_ordering()
        assert mesh.npts == 1

    def test_generator_wrong_dimension_raises(self):
        """UnstructuredMeshGenerator rejects non-2D/3D lims."""
        import pytest

        gen = UnstructuredMeshGenerator()
        x = pybamm.SpatialVariable("x_n", domain=["negative electrode"])
        lims = {x: {"min": 0.0, "max": 1.0}}
        with pytest.raises(pybamm.GeometryError, match="supports 2D and 3D"):
            gen(lims, {"x_n": 3})

    def test_generator_unknown_element_type_raises(self):
        """UnstructuredMeshGenerator rejects bogus element_type."""
        import pytest

        gen = UnstructuredMeshGenerator(element_type="pentagon")
        x = pybamm.SpatialVariable("x_n", domain=["negative electrode"])
        z = pybamm.SpatialVariable(
            "z_2d",
            domain=["negative electrode"],
            direction="tb",
        )
        lims = {x: {"min": 0.0, "max": 1.0}, z: {"min": 0.0, "max": 1.0}}
        with pytest.raises(pybamm.GeometryError, match="Unsupported 2D element_type"):
            gen(lims, {"x_n": 2, "z_2d": 2})

    def test_generator_tetrahedron_element_type(self):
        """3D generator honours element_type='tetrahedron' (not just hexahedron)."""
        x = pybamm.SpatialVariable("x_n", domain=["negative electrode"])
        y = pybamm.SpatialVariable("y", domain=["negative electrode"])
        z = pybamm.SpatialVariable("z", domain=["negative electrode"])
        lims = {
            x: {"min": 0.0, "max": 1.0},
            y: {"min": 0.0, "max": 1.0},
            z: {"min": 0.0, "max": 1.0},
        }
        npts = {"x_n": 2, "y": 2, "z": 2}

        hex_mesh = UnstructuredMeshGenerator()(lims, npts)
        assert hex_mesh.element_type == "hexahedron"  # 3D default

        tet_mesh = UnstructuredMeshGenerator(element_type="tetrahedron")(lims, npts)
        assert tet_mesh.element_type == "tetrahedron"
        assert tet_mesh.npts == 8 * 6  # 6 tets per hex (Kuhn split)
        np.testing.assert_allclose(tet_mesh.cell_volumes.sum(), 1.0, atol=1e-14)

    def test_generator_unknown_3d_element_type_raises(self):
        """3D generator rejects a bogus element_type."""
        import pytest

        gen = UnstructuredMeshGenerator(element_type="dodecahedron")
        x = pybamm.SpatialVariable("x_n", domain=["negative electrode"])
        y = pybamm.SpatialVariable("y", domain=["negative electrode"])
        z = pybamm.SpatialVariable("z", domain=["negative electrode"])
        lims = {
            x: {"min": 0.0, "max": 1.0},
            y: {"min": 0.0, "max": 1.0},
            z: {"min": 0.0, "max": 1.0},
        }
        with pytest.raises(pybamm.GeometryError, match="Unsupported 3D element_type"):
            gen(lims, {"x_n": 2, "y": 2, "z": 2})

    def test_generator_quad_element_type(self):
        """Generator with element_type='quad' produces quad submesh."""
        gen = UnstructuredMeshGenerator(element_type="quad")
        x = pybamm.SpatialVariable("x_n", domain=["negative electrode"])
        z = pybamm.SpatialVariable(
            "z_2d",
            domain=["negative electrode"],
            direction="tb",
        )
        lims = {x: {"min": 0.0, "max": 1.0}, z: {"min": 0.0, "max": 1.0}}
        sub = gen(lims, {"x_n": 2, "z_2d": 2})
        assert sub.element_type == "quad"
        assert sub.npts == 4

    def test_generator_parse_lims_with_string_var(self):
        """_parse_lims accepts string variable names and skips 'tabs'."""
        gen = UnstructuredMeshGenerator()
        spatial_vars, spatial_lims = gen._parse_lims(
            {
                "r_n": {"min": 0.0, "max": 1.0},
                "r_p": {"min": 0.0, "max": 1.0},
                "tabs": {},
            }
        )
        assert len(spatial_vars) == 2
        assert len(spatial_lims) == 2


# ======================================================================
# TestUnstructuredMeshGenerator
# ======================================================================


class TestUnstructuredMeshGenerator:
    def test_2d_generator_basic(self):
        x = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        z = pybamm.SpatialVariable(
            "z_2d",
            domain=["negative electrode", "separator", "positive electrode"],
            coord_sys="cartesian",
            direction="tb",
        )

        lims = {x: {"min": 0.0, "max": 1.0}, z: {"min": 0.0, "max": 1.0}}
        npts = {"x_n": 4, "z_2d": 3}

        gen = UnstructuredMeshGenerator()
        mesh = gen(lims, npts)

        assert isinstance(mesh, UnstructuredSubMesh)
        assert mesh.dimension == 2
        assert mesh.npts == 4 * 3 * 2  # 4*3 quads, 2 tris each
        np.testing.assert_allclose(mesh.cell_volumes.sum(), 1.0, atol=1e-14)

    def test_2d_generator_mesh_integration(self):
        x = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        z = pybamm.SpatialVariable(
            "z_2d",
            domain=["negative electrode"],
            coord_sys="cartesian",
        )
        geometry = {
            "negative electrode": {
                x: {"min": 0.0, "max": 1.0},
                z: {"min": 0.0, "max": 2.0},
            }
        }
        gen = UnstructuredMeshGenerator()
        mesh = pybamm.Mesh(
            geometry,
            {"negative electrode": gen},
            {x: 3, z: 4},
        )
        submesh = mesh["negative electrode"]
        assert isinstance(submesh, UnstructuredSubMesh)
        assert submesh.dimension == 2
        assert submesh.npts == 3 * 4 * 2

    def test_3d_generator_basic(self):
        x = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        y = pybamm.SpatialVariable(
            "y", domain=["negative electrode"], coord_sys="cartesian"
        )
        z = pybamm.SpatialVariable(
            "z", domain=["negative electrode"], coord_sys="cartesian"
        )

        lims = {
            x: {"min": 0.0, "max": 1.0},
            y: {"min": 0.0, "max": 1.0},
            z: {"min": 0.0, "max": 1.0},
        }
        npts = {"x_n": 2, "y": 2, "z": 2}

        gen = UnstructuredMeshGenerator()
        mesh = gen(lims, npts)

        assert isinstance(mesh, UnstructuredSubMesh)
        assert mesh.dimension == 3
        assert mesh.npts == 2 * 2 * 2  # 8 hex cells
        np.testing.assert_allclose(mesh.cell_volumes.sum(), 1.0, atol=1e-14)

    def test_3d_generator_mesh_integration(self):
        x = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        y = pybamm.SpatialVariable(
            "y", domain=["negative electrode"], coord_sys="cartesian"
        )
        z = pybamm.SpatialVariable(
            "z", domain=["negative electrode"], coord_sys="cartesian"
        )

        geometry = {
            "negative electrode": {
                x: {"min": 0.0, "max": 1.0},
                y: {"min": 0.0, "max": 1.0},
                z: {"min": 0.0, "max": 1.0},
            }
        }
        gen = UnstructuredMeshGenerator()
        mesh = pybamm.Mesh(
            geometry,
            {"negative electrode": gen},
            {x: 2, y: 2, z: 2},
        )
        submesh = mesh["negative electrode"]
        assert isinstance(submesh, UnstructuredSubMesh)
        assert submesh.dimension == 3
        assert submesh.npts == 2 * 2 * 2  # 8 hex cells

    def test_interface_conformity_2d(self):
        """Adjacent domains with the same z grid produce matching interface faces."""
        z = pybamm.SpatialVariable(
            "z_2d",
            domain=["negative electrode", "separator"],
            coord_sys="cartesian",
        )
        x_n = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        x_s = pybamm.SpatialVariable("x_s", domain=["separator"], coord_sys="cartesian")

        gen = UnstructuredMeshGenerator()
        left = gen(
            {x_n: {"min": 0.0, "max": 1.0}, z: {"min": 0.0, "max": 1.0}},
            {"x_n": 3, "z_2d": 4},
        )
        right = gen(
            {x_s: {"min": 1.0, "max": 2.0}, z: {"min": 0.0, "max": 1.0}},
            {"x_s": 3, "z_2d": 4},
        )

        # The right boundary of left and left boundary of right should match
        left_right_bnd = left.boundary_faces["right"]
        right_left_bnd = right.boundary_faces["left"]

        assert len(left_right_bnd) == len(right_left_bnd)

        left_transverse = np.sort(left.face_centroids[left_right_bnd, 1])
        right_transverse = np.sort(right.face_centroids[right_left_bnd, 1])
        np.testing.assert_allclose(left_transverse, right_transverse, atol=1e-14)

    def test_interface_conformity_3d(self):
        """Adjacent 3D domains with the same y,z grid produce matching interface faces."""
        x_n = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        x_s = pybamm.SpatialVariable("x_s", domain=["separator"], coord_sys="cartesian")
        y = pybamm.SpatialVariable(
            "y", domain=["negative electrode", "separator"], coord_sys="cartesian"
        )
        z = pybamm.SpatialVariable(
            "z", domain=["negative electrode", "separator"], coord_sys="cartesian"
        )

        gen = UnstructuredMeshGenerator()
        left = gen(
            {
                x_n: {"min": 0, "max": 1},
                y: {"min": 0, "max": 1},
                z: {"min": 0, "max": 1},
            },
            {"x_n": 2, "y": 2, "z": 2},
        )
        right = gen(
            {
                x_s: {"min": 1, "max": 2},
                y: {"min": 0, "max": 1},
                z: {"min": 0, "max": 1},
            },
            {"x_s": 2, "y": 2, "z": 2},
        )

        left_right_bnd = left.boundary_faces["right"]
        right_left_bnd = right.boundary_faces["left"]

        assert len(left_right_bnd) == len(right_left_bnd)
        assert len(left_right_bnd) > 0


# ======================================================================
# TestFileGenerators
# ======================================================================


class TestFileGenerators:
    @staticmethod
    def _write_two_triangle_vtu(path, tags=None):
        """Unit square as 2 triangles, z=0, optional per-cell integer tags."""
        import meshio

        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        cells = [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))]
        cell_data = {"tag": [np.asarray(tags)]} if tags is not None else {}
        meshio.write(str(path), meshio.Mesh(points, cells, cell_data=cell_data))

    def test_user_supplied_loads_whole_mesh(self, tmp_path):
        """Full file load: z=0 points are trimmed to a 2D mesh."""
        import pytest

        pytest.importorskip("meshio")
        path = tmp_path / "square.vtu"
        self._write_two_triangle_vtu(path)

        gen = UserSuppliedUnstructuredMesh(str(path))
        sub = gen({"x_n": {"min": 0.0, "max": 1.0}}, {})

        assert isinstance(sub, UnstructuredSubMesh)
        assert sub.dimension == 2
        assert sub.npts == 2
        np.testing.assert_allclose(sub.cell_volumes.sum(), 1.0)
        assert "square.vtu" in repr(gen)
        # File meshes get no guessed tags: boundary names must come from the
        # mesh file (boundary_mapping) or be set explicitly
        assert sub.boundary_faces == {}

    def test_user_supplied_subdomain_filtering(self, tmp_path):
        """subdomain_mapping selects only the cells with the matching tag."""
        import pytest

        pytest.importorskip("meshio")
        path = tmp_path / "tagged.vtu"
        self._write_two_triangle_vtu(path, tags=[1, 2])

        gen = UserSuppliedUnstructuredMesh(
            str(path), subdomain_mapping={"negative electrode": 1}
        )
        sub = gen({"x_n": {"min": 0.0, "max": 1.0}}, {})
        assert sub.npts == 1

        # A mesh with no cell data at all cannot satisfy a subdomain_mapping
        import meshio

        gen_bad = UserSuppliedUnstructuredMesh(
            str(path), subdomain_mapping={"negative electrode": 1}
        )
        gen_bad._cached_mesh = meshio.Mesh(
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float),
            [("triangle", np.array([[0, 1, 2]]))],
        )
        with pytest.raises(pybamm.GeometryError, match="cell data tag"):
            gen_bad({"x_n": {"min": 0.0, "max": 1.0}}, {})

    def test_user_supplied_boundary_mapping_tags_faces(self):
        """boundary_mapping maps tagged facet groups to boundary face names."""
        import pytest

        meshio = pytest.importorskip("meshio")
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        tris = np.array([[0, 1, 2], [0, 2, 3]])
        lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        gen = UserSuppliedUnstructuredMesh(
            "unused.vtu", boundary_mapping={"seal": 1, "vent": 2}
        )
        gen._cached_mesh = meshio.Mesh(
            points,
            [("triangle", tris), ("line", lines)],
            cell_data={"gmsh:physical": [np.array([0, 0]), np.array([1, 1, 2, 2])]},
        )
        sub = gen({"x_n": {"min": 0.0, "max": 1.0}}, {})

        assert set(sub.boundary_faces) == {"seal", "vent"}
        seal = sub.face_centroids[sub.boundary_faces["seal"]]
        vent = sub.face_centroids[sub.boundary_faces["vent"]]
        # seal = bottom + right edges, vent = top + left edges
        np.testing.assert_allclose(sorted(map(tuple, seal)), [(0.5, 0.0), (1.0, 0.5)])
        np.testing.assert_allclose(sorted(map(tuple, vent)), [(0.0, 0.5), (0.5, 1.0)])

    def test_tagged_generator_boundary_mapping(self, monkeypatch):
        """TaggedSubMeshGenerator resolves surface physical groups to tags."""
        import pytest

        meshio = pytest.importorskip("meshio")
        nodes, elements = _unit_cube_five_tets()
        # Bottom cube face as the two triangles the 5-tet split puts there
        base_tris = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = meshio.Mesh(
            nodes,
            [("tetra", elements), ("triangle", base_tris)],
            cell_data={"gmsh:physical": [np.full(5, 1), np.full(2, 10)]},
            field_data={"anode": np.array([1, 3]), "base": np.array([10, 2])},
        )
        monkeypatch.setattr(
            TaggedSubMeshGenerator, "_read", classmethod(lambda cls, path: mesh)
        )
        gen = TaggedSubMeshGenerator(
            "anode", "unused.msh", boundary_mapping={"bottom_seal": "base"}
        )
        sub = gen({}, {})
        assert set(sub.boundary_faces) == {"bottom_seal"}
        centroids = sub.face_centroids[sub.boundary_faces["bottom_seal"]]
        np.testing.assert_allclose(centroids[:, 2], 0.0, atol=1e-14)
        assert len(centroids) == 2

    def test_user_supplied_no_supported_cells_raises(self):
        """A mesh with only unsupported cell types raises."""
        import pytest

        meshio = pytest.importorskip("meshio")
        gen = UserSuppliedUnstructuredMesh("unused.vtu")
        gen._cached_mesh = meshio.Mesh(
            np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
            [("line", np.array([[0, 1]]))],
        )
        with pytest.raises(pybamm.GeometryError, match="No supported cells"):
            gen({"x_n": {"min": 0.0, "max": 1.0}}, {})

    def test_user_supplied_hexahedron_cells_raise(self):
        """Hexahedral file meshes are rejected, even alongside tets."""
        import pytest

        meshio = pytest.importorskip("meshio")
        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=float,
        )
        hex_cells = ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))
        tet_cells = ("tetra", np.array([[0, 1, 2, 4]]))

        gen = UserSuppliedUnstructuredMesh("unused.vtu")
        gen._cached_mesh = meshio.Mesh(points, [hex_cells])
        with pytest.raises(pybamm.GeometryError, match="Hexahedral cells"):
            gen({"x_n": {"min": 0.0, "max": 1.0}}, {})

        # Mixed tet+hex must also raise, not silently drop the hex cells
        gen_mixed = UserSuppliedUnstructuredMesh("unused.vtu")
        gen_mixed._cached_mesh = meshio.Mesh(points, [tet_cells, hex_cells])
        with pytest.raises(pybamm.GeometryError, match="Hexahedral cells"):
            gen_mixed({"x_n": {"min": 0.0, "max": 1.0}}, {})

    def test_domain_name_from_lims(self):
        """String and SpatialVariable keys map to electrode domains; 'tabs' skipped."""
        f = UserSuppliedUnstructuredMesh._domain_name_from_lims
        assert f({"x_n": {}}) == "negative electrode"
        assert f({"x_s": {}}) == "separator"
        assert f({"tabs": {}, "x_p": {}}) == "positive electrode"
        assert f({"r_n": {}}) is None
        x = pybamm.SpatialVariable("x_n", domain=["negative electrode"])
        assert f({x: {}}) == "negative electrode"

    @staticmethod
    def _tagged_gmsh_mesh():
        """Synthetic meshio mesh: 5 tets of the unit cube, two physical groups."""
        import meshio

        nodes, elements = _unit_cube_five_tets()
        return meshio.Mesh(
            nodes,
            [("tetra", elements)],
            cell_data={"gmsh:physical": [np.array([1, 1, 1, 2, 2])]},
            field_data={"anode": np.array([1, 3]), "cathode": np.array([2, 3])},
        )

    def test_tagged_generator_extracts_region(self, monkeypatch):
        import pytest

        pytest.importorskip("meshio")
        monkeypatch.setattr(
            TaggedSubMeshGenerator,
            "_read",
            classmethod(lambda cls, path: self._tagged_gmsh_mesh()),
        )
        gen = TaggedSubMeshGenerator("anode", "unused.msh", scale=2.0)
        sub = gen({}, {})
        assert isinstance(sub, UnstructuredSubMesh)
        assert sub.npts == 3  # cells tagged 1
        # scale multiplies coordinates: unit cube -> side 2
        np.testing.assert_allclose(sub.vertices.max(axis=0), [2.0, 2.0, 2.0])

    def test_tagged_generator_missing_region_raises(self, monkeypatch):
        import pytest

        pytest.importorskip("meshio")
        monkeypatch.setattr(
            TaggedSubMeshGenerator,
            "_read",
            classmethod(lambda cls, path: self._tagged_gmsh_mesh()),
        )
        gen = TaggedSubMeshGenerator("does-not-exist", "unused.msh")
        with pytest.raises(pybamm.GeometryError, match="not in mesh field_data"):
            gen({}, {})

    def test_tagged_generator_region_without_tets_raises(self, monkeypatch):
        import pytest

        pytest.importorskip("meshio")
        mesh = self._tagged_gmsh_mesh()
        # Physical group 9 exists in field_data but tags no tet cells
        mesh.field_data["empty"] = np.array([9, 3])
        monkeypatch.setattr(
            TaggedSubMeshGenerator, "_read", classmethod(lambda cls, path: mesh)
        )
        gen = TaggedSubMeshGenerator("empty", "unused.msh")
        with pytest.raises(pybamm.GeometryError, match="no tets"):
            gen({}, {})

    def test_tagged_generator_cache_reads_and_invalidates(self, tmp_path):
        import os

        import pytest

        meshio = pytest.importorskip("meshio")
        TaggedSubMeshGenerator._read_cached.cache_clear()

        path = tmp_path / "cache_demo.msh"
        meshio.write(
            str(path),
            meshio.Mesh(
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
                [("tetra", np.array([[0, 1, 2, 3]]))],
            ),
            file_format="gmsh22",
            binary=False,
        )

        first = TaggedSubMeshGenerator._read(path)
        # unchanged file: cache hit returns the same object
        assert TaggedSubMeshGenerator._read(path) is first
        # str and pathlib.Path collapse to a single cache entry
        assert TaggedSubMeshGenerator._read(str(path)) is first
        # a newer modification time invalidates the entry and re-reads
        stat = os.stat(path)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        assert TaggedSubMeshGenerator._read(path) is not first


# ======================================================================
# TestComputeInterfaceData
# ======================================================================


class TestComputeInterfaceData:
    def test_2d_interface_matching(self):
        gen = UnstructuredMeshGenerator()
        x_n = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        x_s = pybamm.SpatialVariable("x_s", domain=["separator"], coord_sys="cartesian")
        z = pybamm.SpatialVariable(
            "z_2d",
            domain=["negative electrode", "separator"],
            coord_sys="cartesian",
        )

        left = gen(
            {x_n: {"min": 0, "max": 1}, z: {"min": 0, "max": 1}},
            {"x_n": 3, "z_2d": 3},
        )
        right = gen(
            {x_s: {"min": 1, "max": 2}, z: {"min": 0, "max": 1}},
            {"x_s": 3, "z_2d": 3},
        )

        result = compute_interface_data(left, right)

        assert len(result["left_cells"]) == len(result["right_cells"])
        assert len(result["face_areas"]) == len(result["left_cells"])
        assert len(result["cell_distances"]) == len(result["left_cells"])
        assert np.all(result["cell_distances"] > 0)
        assert np.all(result["face_areas"] > 0)

    def test_3d_interface_matching(self):
        gen = UnstructuredMeshGenerator()
        x_n = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        x_s = pybamm.SpatialVariable("x_s", domain=["separator"], coord_sys="cartesian")
        y = pybamm.SpatialVariable(
            "y", domain=["negative electrode", "separator"], coord_sys="cartesian"
        )
        z = pybamm.SpatialVariable(
            "z", domain=["negative electrode", "separator"], coord_sys="cartesian"
        )

        left = gen(
            {
                x_n: {"min": 0, "max": 1},
                y: {"min": 0, "max": 1},
                z: {"min": 0, "max": 1},
            },
            {"x_n": 2, "y": 2, "z": 2},
        )
        right = gen(
            {
                x_s: {"min": 1, "max": 2},
                y: {"min": 0, "max": 1},
                z: {"min": 0, "max": 1},
            },
            {"x_s": 2, "y": 2, "z": 2},
        )

        result = compute_interface_data(left, right)

        assert len(result["left_cells"]) > 0
        assert len(result["left_cells"]) == len(result["right_cells"])
        assert np.all(result["cell_distances"] > 0)
        assert np.all(result["face_areas"] > 0)

    def test_interface_data_stored_on_submesh(self):
        gen = UnstructuredMeshGenerator()
        x_n = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        x_s = pybamm.SpatialVariable("x_s", domain=["separator"], coord_sys="cartesian")
        z = pybamm.SpatialVariable(
            "z_2d",
            domain=["negative electrode", "separator"],
            coord_sys="cartesian",
        )

        left = gen(
            {x_n: {"min": 0, "max": 1}, z: {"min": 0, "max": 1}},
            {"x_n": 2, "z_2d": 2},
        )
        right = gen(
            {x_s: {"min": 1, "max": 2}, z: {"min": 0, "max": 1}},
            {"x_s": 2, "z_2d": 2},
        )

        compute_interface_data(
            left, right, left_name="negative electrode", right_name="separator"
        )

        assert "separator" in left.interface_data
        assert "negative electrode" in right.interface_data

        left_to_right = left.interface_data["separator"]
        right_to_left = right.interface_data["negative electrode"]

        np.testing.assert_array_equal(
            left_to_right["left_cells"], right_to_left["right_cells"]
        )
        np.testing.assert_array_equal(
            left_to_right["right_cells"], right_to_left["left_cells"]
        )

    def test_no_matching_boundary_faces_raises(self):
        """A mesh without the required boundary bucket cannot be paired."""
        import pytest

        ye = np.linspace(0, 1, 3)
        ze = np.linspace(0, 1, 3)
        left = UnstructuredSubMesh(*_hex_grid(np.linspace(0, 1, 3), ye, ze))
        nodes_r, elems_r = _hex_grid(np.linspace(1, 2, 3), ye, ze)
        # Empty boundary_faces: no "left" bucket to match against
        right = UnstructuredSubMesh(nodes_r, elems_r, boundary_faces={})

        with pytest.raises(pybamm.GeometryError, match="matching boundary faces"):
            compute_interface_data(left, right)

    def test_non_bijective_pairing_raises(self):
        """Surplus faces on one side of an interface must not vanish silently.

        Every left face matches a right face exactly, but the right mesh
        exposes one extra 'left' face — its flux would silently disappear
        from the coupling.
        """
        import pytest

        ye = np.linspace(0, 1, 3)
        ze = np.linspace(0, 1, 3)
        left = UnstructuredSubMesh(*_hex_grid(np.linspace(0, 1, 3), ye, ze))
        right = UnstructuredSubMesh(*_hex_grid(np.linspace(1, 2, 3), ye, ze))
        left.detect_box_boundaries()
        right.detect_box_boundaries()
        right.boundary_faces["left"] = np.append(
            right.boundary_faces["left"], right.boundary_faces["top"][0]
        )

        with pytest.raises(pybamm.GeometryError, match=r"one-to-one"):
            compute_interface_data(left, right)

    def test_transverse_mismatch_raises(self):
        """Interface faces at different transverse positions cannot be paired."""
        import pytest

        ze = np.linspace(0, 1, 3)
        left = UnstructuredSubMesh(
            *_hex_grid(np.linspace(0, 1, 3), np.linspace(0, 1, 3), ze)
        )
        left.detect_box_boundaries()
        # y grid shifted by 0.37: same face count, wrong transverse positions
        right = UnstructuredSubMesh(
            *_hex_grid(np.linspace(1, 2, 3), np.linspace(0.37, 1.37, 3), ze)
        )
        right.detect_box_boundaries()

        with pytest.raises(pybamm.GeometryError, match="do not match"):
            compute_interface_data(left, right)


# ======================================================================
# TestMeshIntegration
# ======================================================================


class TestMeshIntegration:
    def test_ghost_mesh_excluded(self):
        x = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        z = pybamm.SpatialVariable(
            "z_2d",
            domain=["negative electrode"],
            coord_sys="cartesian",
        )
        geometry = {
            "negative electrode": {
                x: {"min": 0.0, "max": 1.0},
                z: {"min": 0.0, "max": 1.0},
            }
        }
        gen = UnstructuredMeshGenerator()
        mesh = pybamm.Mesh(
            geometry,
            {"negative electrode": gen},
            {x: 3, z: 3},
        )

        ghost_keys = [k for k in mesh if "ghost" in str(k)]
        assert len(ghost_keys) == 0

    def test_combine_submeshes(self):
        x_n = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        x_s = pybamm.SpatialVariable("x_s", domain=["separator"], coord_sys="cartesian")
        x_p = pybamm.SpatialVariable(
            "x_p", domain=["positive electrode"], coord_sys="cartesian"
        )
        z = pybamm.SpatialVariable(
            "z_2d",
            domain=["negative electrode", "separator", "positive electrode"],
            coord_sys="cartesian",
        )

        geometry = {
            "negative electrode": {x_n: {"min": 0, "max": 1}, z: {"min": 0, "max": 1}},
            "separator": {x_s: {"min": 1, "max": 1.5}, z: {"min": 0, "max": 1}},
            "positive electrode": {
                x_p: {"min": 1.5, "max": 2.5},
                z: {"min": 0, "max": 1},
            },
        }

        gen = UnstructuredMeshGenerator()
        mesh = pybamm.Mesh(
            geometry,
            {
                "negative electrode": gen,
                "separator": gen,
                "positive electrode": gen,
            },
            {x_n: 3, x_s: 2, x_p: 3, z: 4},
        )

        n_neg = mesh["negative electrode"].npts
        n_sep = mesh["separator"].npts
        n_pos = mesh["positive electrode"].npts

        combined = mesh[("negative electrode", "separator", "positive electrode")]
        assert combined.npts == n_neg + n_sep + n_pos

    def test_combine_conforming_interface_welds_into_internal_faces(self):
        """A matching interface welds, so the combined mesh is one component."""
        ye = np.linspace(0, 1, 3)
        ze = np.linspace(0, 1, 4)
        left = UnstructuredSubMesh(*_hex_grid(np.linspace(0, 1, 3), ye, ze))
        right = UnstructuredSubMesh(*_hex_grid(np.linspace(1, 2, 3), ye, ze))

        combined = UnstructuredSubMesh.combine([left, right])

        assert combined.npts == left.npts + right.npts
        # coincident interface nodes are welded, not duplicated
        assert (
            combined.vertices.shape[0]
            < left.vertices.shape[0] + right.vertices.shape[0]
        )
        # the x=1 seam is now internal, so flux can cross it
        n_int = combined._boundary_face_start
        on_seam = np.abs(combined.face_centroids[:n_int, 0] - 1.0) < 1e-12
        assert on_seam.sum() > 0

    def test_combine_tetrahedron_domains_conforming(self):
        """Tet domains from the generator combine into one connected mesh.

        Uses an odd left nx so the cumulative-offset regeneration in
        ``combine`` actually fires (an even offset leaves the parity of
        ``(i + i_offset + j + k)`` unchanged and the branch is a no-op).
        The combined mesh must describe the same cells as the per-domain
        meshes: PyBaMM reads both views of the same region.
        """
        x_n = pybamm.SpatialVariable("x_n", domain=["negative electrode"])
        x_s = pybamm.SpatialVariable("x_s", domain=["separator"])
        y = pybamm.SpatialVariable("y", domain=["negative electrode", "separator"])
        z = pybamm.SpatialVariable("z", domain=["negative electrode", "separator"])
        gen = UnstructuredMeshGenerator(element_type="tetrahedron")
        left = gen(
            {
                x_n: {"min": 0, "max": 1},
                y: {"min": 0, "max": 1},
                z: {"min": 0, "max": 1},
            },
            {"x_n": 3, "y": 2, "z": 2},
        )
        right = gen(
            {
                x_s: {"min": 1, "max": 2},
                y: {"min": 0, "max": 1},
                z: {"min": 0, "max": 1},
            },
            {"x_s": 2, "y": 2, "z": 2},
        )
        # Must not raise: interface tets triangulate identically on both sides.
        combined = UnstructuredSubMesh.combine([left, right])
        assert combined.element_type == "tetrahedron"
        assert combined.npts == left.npts + right.npts

        # Both views of each domain must agree cell-for-cell
        n_left = left.npts
        np.testing.assert_allclose(
            combined.cell_centroids[:n_left], left.cell_centroids
        )
        np.testing.assert_allclose(
            combined.cell_centroids[n_left:], right.cell_centroids
        )
        np.testing.assert_allclose(combined.cell_volumes[n_left:], right.cell_volumes)

    def test_combine_propagates_custom_boundary_tags(self):
        """Non-standard boundary tags survive combining via centroid matching."""
        ye = np.linspace(0, 1, 3)
        ze = np.linspace(0, 1, 3)
        left = UnstructuredSubMesh(*_hex_grid(np.linspace(0, 1, 3), ye, ze))
        right = UnstructuredSubMesh(*_hex_grid(np.linspace(1, 2, 3), ye, ze))
        left.detect_box_boundaries()
        right.detect_box_boundaries()
        left.boundary_faces["tab_top"] = left.boundary_faces.pop("top")

        combined = UnstructuredSubMesh.combine([left, right])

        assert "tab_top" in combined.boundary_faces
        # The recovered faces sit on the z=1 plane of the left half only
        tab_centroids = combined.face_centroids[combined.boundary_faces["tab_top"]]
        np.testing.assert_allclose(tab_centroids[:, 2], 1.0)
        assert tab_centroids[:, 0].max() <= 1.0

    def test_mesh_pairs_interfaces_by_geometry_not_dict_order(self):
        """Adjacency comes from boundary-plane positions, not insertion order.

        The separator (x: 1-2) is declared before the negative electrode
        (x: 0-1); consecutive-order pairing would try sep.right vs neg.left
        (planes x=2 vs x=0), find nothing, and drop the coupling.
        """
        x_n = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        x_s = pybamm.SpatialVariable("x_s", domain=["separator"], coord_sys="cartesian")
        z_a = pybamm.SpatialVariable(
            "z_2d", domain=["negative electrode"], coord_sys="cartesian"
        )
        z_b = pybamm.SpatialVariable(
            "z_2d", domain=["separator"], coord_sys="cartesian"
        )

        geometry = {
            "separator": {x_s: {"min": 1, "max": 2}, z_b: {"min": 0, "max": 1}},
            "negative electrode": {
                x_n: {"min": 0, "max": 1},
                z_a: {"min": 0, "max": 1},
            },
        }
        gen = UnstructuredMeshGenerator()
        mesh = pybamm.Mesh(
            geometry,
            {"negative electrode": gen, "separator": gen},
            {x_n: 3, x_s: 3, z_a: 4, z_b: 4},
        )

        assert "separator" in mesh["negative electrode"].interface_data
        assert "negative electrode" in mesh["separator"].interface_data
        # The coupling must join abutting cells across x=1, not staple the
        # far edges (x=0 and x=2) together. For the default triangle cells
        # the abutting centroids sit dx/3 from the interface with a dz/3
        # vertical offset, so the distance is one cell scale — not the
        # geometry's full span
        iface = mesh["negative electrode"].interface_data["separator"]
        np.testing.assert_allclose(
            iface["cell_distances"], np.hypot(2.0 / 9.0, 1.0 / 12.0)
        )

    def test_mesh_skips_interface_for_mismatched_grids(self, caplog):
        """Mesh.__init__ leaves interface_data empty when pairing fails,
        and says so: a skipped interface means no flux couples the domains,
        which must be visible rather than silent."""
        x_n = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        x_s = pybamm.SpatialVariable("x_s", domain=["separator"], coord_sys="cartesian")
        z_a = pybamm.SpatialVariable(
            "z_2d", domain=["negative electrode"], coord_sys="cartesian"
        )
        z_b = pybamm.SpatialVariable(
            "z_2d", domain=["separator"], coord_sys="cartesian"
        )

        # Transverse extents disagree (z: 0-1 vs 0-2), so pairing cannot match
        geometry = {
            "negative electrode": {
                x_n: {"min": 0, "max": 1},
                z_a: {"min": 0, "max": 1},
            },
            "separator": {x_s: {"min": 1, "max": 2}, z_b: {"min": 0, "max": 2}},
        }
        gen = UnstructuredMeshGenerator()
        with caplog.at_level("WARNING", logger="pybamm"):
            mesh = pybamm.Mesh(
                geometry,
                {"negative electrode": gen, "separator": gen},
                {x_n: 3, x_s: 3, z_a: 4, z_b: 4},
            )

        assert mesh["negative electrode"].interface_data == {}
        assert mesh["separator"].interface_data == {}
        assert "No interface coupling between 'negative electrode'" in caplog.text
        assert "separator" in caplog.text

    def test_combine_battery_scale_with_interface_jitter(self):
        """Welding tolerances scale with the mesh, not with 1 metre.

        Two 100 um domains whose shared interface is offset by 3 nm — the
        precision loss a reduced-precision mesh file or a unit conversion
        produces. The weld must absorb it (it is ~1e-4 of an edge), and
        custom boundary tags must survive it.
        """
        um = 1e-6
        ye = np.linspace(0, 100 * um, 3)
        ze = np.linspace(0, 100 * um, 3)
        left = UnstructuredSubMesh(*_hex_grid(np.linspace(0, 100 * um, 6), ye, ze))
        right = UnstructuredSubMesh(
            *_hex_grid(np.linspace(100 * um + 3e-9, 200 * um, 6), ye, ze)
        )
        left.detect_box_boundaries()
        right.detect_box_boundaries()
        left.boundary_faces["tab_top"] = left.boundary_faces.pop("top")

        combined = UnstructuredSubMesh.combine([left, right])

        assert combined.npts == left.npts + right.npts
        assert "tab_top" in combined.boundary_faces
        tab = combined.face_centroids[combined.boundary_faces["tab_top"]]
        np.testing.assert_allclose(tab[:, 2], 100 * um, rtol=1e-6)

    def test_combine_mixed_element_types_raises(self):
        """Combining domains of different element types names both types."""
        import pytest

        tri_nodes, tri_elems = _unit_square_two_triangles()
        quad_nodes = np.array([[1, 0], [2, 0], [2, 1], [1, 1]], dtype=float)
        quad_elems = np.array([[0, 1, 2, 3]], dtype=int)
        tri = UnstructuredSubMesh(tri_nodes, tri_elems)
        quad = UnstructuredSubMesh(quad_nodes, quad_elems)

        with pytest.raises(
            pybamm.GeometryError, match=r"quad.*triangle|triangle.*quad"
        ):
            UnstructuredSubMesh.combine([tri, quad])

    def test_combine_disconnected_domains_raises(self):
        """A non-conforming interface must raise, not solve silently to garbage."""
        import pytest

        ye = np.linspace(0, 1, 3)
        left = UnstructuredSubMesh(
            *_hex_grid(np.linspace(0, 1, 3), ye, np.linspace(0, 1, 4))
        )
        # mismatched transverse (z) grid: no interface node coincides
        right = UnstructuredSubMesh(
            *_hex_grid(np.linspace(1, 2, 3), ye, np.linspace(0, 1, 5))
        )

        with pytest.raises(pybamm.GeometryError, match=r"disconnected regions"):
            UnstructuredSubMesh.combine([left, right])

    def test_interface_data_computed_automatically(self):
        """Mesh.__init__ should compute interface data between adjacent domains."""
        x_n = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        x_s = pybamm.SpatialVariable("x_s", domain=["separator"], coord_sys="cartesian")
        z = pybamm.SpatialVariable(
            "z_2d",
            domain=["negative electrode", "separator"],
            coord_sys="cartesian",
        )

        geometry = {
            "negative electrode": {x_n: {"min": 0, "max": 1}, z: {"min": 0, "max": 1}},
            "separator": {x_s: {"min": 1, "max": 2}, z: {"min": 0, "max": 1}},
        }

        gen = UnstructuredMeshGenerator()
        mesh = pybamm.Mesh(
            geometry,
            {"negative electrode": gen, "separator": gen},
            {x_n: 3, x_s: 3, z: 4},
        )

        neg_mesh = mesh["negative electrode"]
        sep_mesh = mesh["separator"]

        assert "separator" in neg_mesh.interface_data
        assert "negative electrode" in sep_mesh.interface_data
        assert len(neg_mesh.interface_data["separator"]["left_cells"]) > 0


class TestBandwidthOptimization:
    """Tests for _hex_grid loop ordering and optimize_ordering (RCM)."""

    @staticmethod
    def _bandwidth(submesh):
        n_int = submesh._boundary_face_start
        owners = submesh.face_owner[:n_int]
        neighbors = submesh.face_neighbor
        return int(np.max(np.abs(owners.astype(int) - neighbors.astype(int))))

    def test_hex_grid_optimal_loop_order(self):
        """_hex_grid should order cells so bandwidth = product of two smallest dims."""
        for nx, ny, nz in [(3, 10, 5), (2, 4, 20), (7, 3, 3), (5, 5, 5)]:
            nodes, elems = _hex_grid(
                np.linspace(0, 1, nx + 1),
                np.linspace(0, 1, ny + 1),
                np.linspace(0, 1, nz + 1),
            )
            mesh = UnstructuredSubMesh(nodes, elems, coord_sys="cartesian")
            bw = self._bandwidth(mesh)
            dims = sorted([nx, ny, nz])
            expected = dims[0] * dims[1]
            assert bw == expected, (
                f"nx={nx} ny={ny} nz={nz}: bw={bw}, expected={expected}"
            )

    def test_optimize_ordering_reduces_bandwidth(self):
        """optimize_ordering (RCM) should not increase bandwidth."""
        nodes, elems = _hex_grid(
            np.linspace(0, 1, 4),
            np.linspace(0, 1, 11),
            np.linspace(0, 1, 6),
        )
        mesh = UnstructuredSubMesh(nodes, elems, coord_sys="cartesian")
        bw_before = self._bandwidth(mesh)
        mesh.optimize_ordering()
        bw_after = self._bandwidth(mesh)
        assert bw_after <= bw_before

    def test_optimize_ordering_preserves_geometry(self):
        """Cell volumes and centroids must be the same set after reordering."""
        nodes, elems = _hex_grid(
            np.linspace(0, 1, 4),
            np.linspace(0, 1, 6),
            np.linspace(0, 1, 4),
        )
        mesh = UnstructuredSubMesh(nodes, elems, coord_sys="cartesian")
        vols_before = np.sort(mesh.cell_volumes)
        cents_before = mesh.cell_centroids[np.lexsort(mesh.cell_centroids.T)]

        mesh.optimize_ordering()

        vols_after = np.sort(mesh.cell_volumes)
        cents_after = mesh.cell_centroids[np.lexsort(mesh.cell_centroids.T)]
        np.testing.assert_allclose(vols_before, vols_after)
        np.testing.assert_allclose(cents_before, cents_after)

    def test_optimize_ordering_preserves_interface_data(self):
        """Interface cell centroids should point to the same physical cells."""
        ye = np.linspace(0, 1, 6)
        ze = np.linspace(0, 1, 4)
        nodes_l, elems_l = _hex_grid(np.linspace(0, 1, 4), ye, ze)
        mesh_l = UnstructuredSubMesh(nodes_l, elems_l, coord_sys="cartesian")
        mesh_l.detect_box_boundaries()
        nodes_r, elems_r = _hex_grid(np.linspace(1, 2, 4), ye, ze)
        mesh_r = UnstructuredSubMesh(nodes_r, elems_r, coord_sys="cartesian")
        mesh_r.detect_box_boundaries()
        compute_interface_data(mesh_l, mesh_r, "left", "right")

        iface = mesh_l.interface_data["right"]
        centroids_pre = mesh_l.cell_centroids[iface["left_cells"]].copy()

        mesh_l.optimize_ordering()

        iface = mesh_l.interface_data["right"]
        centroids_post = mesh_l.cell_centroids[iface["left_cells"]]
        np.testing.assert_allclose(centroids_pre, centroids_post)

    def test_optimize_ordering_preserves_interface_pairing(self):
        """``right_cells`` must keep indexing the neighbour after reordering."""
        ye = np.linspace(0, 1, 4)
        ze = np.linspace(0, 1, 3)
        # Different x resolutions, so permuting the neighbour's indices by this
        # mesh's permutation would run off the end of the neighbour.
        nodes_l, elems_l = _hex_grid(np.linspace(0, 1, 4), ye, ze)
        mesh_l = UnstructuredSubMesh(nodes_l, elems_l, coord_sys="cartesian")
        mesh_l.detect_box_boundaries()
        nodes_r, elems_r = _hex_grid(np.linspace(1, 2, 3), ye, ze)
        mesh_r = UnstructuredSubMesh(nodes_r, elems_r, coord_sys="cartesian")
        mesh_r.detect_box_boundaries()
        assert mesh_l.npts > mesh_r.npts
        compute_interface_data(mesh_l, mesh_r, "left", "right")

        iface = mesh_l.interface_data["right"]
        self_pre = mesh_l.cell_centroids[iface["left_cells"]].copy()
        other_pre = mesh_r.cell_centroids[iface["right_cells"]].copy()

        mesh_l.optimize_ordering()

        iface = mesh_l.interface_data["right"]
        assert iface["right_cells"].max() < mesh_r.npts
        np.testing.assert_allclose(self_pre, mesh_l.cell_centroids[iface["left_cells"]])
        np.testing.assert_allclose(
            other_pre, mesh_r.cell_centroids[iface["right_cells"]]
        )

        # The neighbour's mirrored view of this mesh must follow the permutation
        mirror = mesh_r.interface_data["left"]
        np.testing.assert_allclose(
            self_pre, mesh_l.cell_centroids[mirror["right_cells"]]
        )
        np.testing.assert_allclose(
            other_pre, mesh_r.cell_centroids[mirror["left_cells"]]
        )
