import numpy as np

import pybamm
from pybamm.meshes.unstructured_submesh import (
    UnstructuredMeshGenerator,
    UnstructuredSubMesh,
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

        for tag in ("left", "right", "front", "back", "bottom", "top"):
            assert tag in mesh.boundary_faces, f"Missing boundary tag '{tag}'"
            assert len(mesh.boundary_faces[tag]) > 0

    def test_custom_boundary_faces(self):
        nodes, elements = _unit_square_two_triangles()
        custom_bnd = {"my_boundary": np.array([3, 4])}
        mesh = UnstructuredSubMesh(nodes, elements, boundary_faces=custom_bnd)

        assert "my_boundary" in mesh.boundary_faces
        np.testing.assert_array_equal(mesh.boundary_faces["my_boundary"], [3, 4])


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

        ghost_keys = [k for k in mesh.keys() if "ghost" in str(k)]
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
