import pytest
import numpy as np
import meshio
from unittest import mock

from pybamm.meshes.three_dimensional_submeshes import SubMesh3D, PyGmshMeshGenerator


class TestSubMesh3D:
    def test_initialization(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        cells = [("tetra", np.array([[0, 1, 2, 3]]))]
        mesh = meshio.Mesh(points, cells)

        submesh = SubMesh3D(mesh)

        assert submesh.coord_sys == "cartesian"
        assert len(submesh.internal_boundaries) == 0
        assert submesh.npts == 1
        assert len(submesh.points) == 4
        assert len(submesh.tetrahedra) == 1

    def test_volumes_and_centers(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        cells = [("tetra", np.array([[0, 1, 2, 3]]))]
        mesh = meshio.Mesh(points, cells)

        submesh = SubMesh3D(mesh)

        assert np.isclose(submesh.volumes[0], 1 / 6)

        assert np.allclose(submesh.nodes[0], np.array([0.25, 0.25, 0.25]))

    def test_adjacency(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        cells = [("tetra", np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))]
        mesh = meshio.Mesh(points, cells)

        submesh = SubMesh3D(mesh)

        assert 1 in submesh.adjacency_list[0]
        assert 0 in submesh.adjacency_list[1]
        assert len(submesh.adjacency_list[0]) == 1
        assert len(submesh.adjacency_list[1]) == 1

    def test_empty_mesh(self):
        points = np.zeros((0, 3))
        cells = [("tetra", np.zeros((0, 4), dtype=int))]
        mesh = meshio.Mesh(points, cells)

        submesh = SubMesh3D(mesh)

        assert submesh.npts == 0
        assert len(submesh.volumes) == 0
        assert len(submesh.nodes) == 0

    def test_no_tetrahedra(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        cells = [("triangle", np.array([[0, 1, 2]]))]
        mesh = meshio.Mesh(points, cells)

        submesh = SubMesh3D(mesh)

        assert submesh.npts == 0
        assert len(submesh.volumes) == 0
        assert len(submesh.nodes) == 0


class TestPyGmshMeshGenerator:
    def test_initialization(self):
        generator = PyGmshMeshGenerator(mesh_size=0.2)
        assert generator.mesh_size == 0.2

    @mock.patch("pygmsh.geo.Geometry")
    def test_generate_rectangular(self, mock_geometry):
        mock_instance = mock_geometry.return_value.__enter__.return_value
        mock_instance.generate_mesh.return_value = meshio.Mesh(
            points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            cells=[("tetra", np.array([[0, 1, 2, 3]]))],
        )

        generator = PyGmshMeshGenerator(mesh_size=0.1)

        params = {"x": [0, 1], "y": [0, 1], "z": [0, 1]}
        submesh = generator.generate("rectangular", params)

        mock_instance.add_box.assert_called_once_with(
            x0=0, y0=0, z0=0, x1=1, y1=1, z1=1, mesh_size=0.1
        )

        assert isinstance(submesh, SubMesh3D)
        assert len(submesh.tetrahedra) == 1

    @mock.patch("pygmsh.geo.Geometry")
    def test_generate_cylindrical(self, mock_geometry):
        mock_instance = mock_geometry.return_value.__enter__.return_value
        mock_instance.generate_mesh.return_value = meshio.Mesh(
            points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            cells=[("tetra", np.array([[0, 1, 2, 3]]))],
        )

        generator = PyGmshMeshGenerator(mesh_size=0.1)

        params = {"radius": 1.0, "height": 2.0}
        submesh = generator.generate("cylindrical", params)

        mock_instance.add_cylinder.assert_called_once_with(
            [0, 0, 0], [0, 0, 2.0], 1.0, mesh_size=0.1
        )

        assert isinstance(submesh, SubMesh3D)
        assert len(submesh.tetrahedra) == 1

    @mock.patch("pygmsh.geo.Geometry")
    def test_generate_spiral(self, mock_geometry):
        mock_instance = mock_geometry.return_value.__enter__.return_value
        mock_instance.generate_mesh.return_value = meshio.Mesh(
            points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            cells=[("tetra", np.array([[0, 1, 2, 3]]))],
        )
        mock_instance.add_parametric_surface.return_value = "surface"

        generator = PyGmshMeshGenerator(mesh_size=0.1)

        params = {"inner_radius": 0.5, "outer_radius": 1.0, "height": 2.0, "turns": 3}
        submesh = generator.generate("spiral", params)

        mock_instance.add_parametric_surface.assert_called_once()
        mock_instance.add_surface_loop.assert_called_once_with(["surface"])

        assert isinstance(submesh, SubMesh3D)
        assert len(submesh.tetrahedra) == 1

    def test_unknown_geometry_type(self):
        generator = PyGmshMeshGenerator()

        with pytest.raises(ValueError, match="Unknown 3D geometry"):
            generator.generate("unknown_type", {})

    @pytest.mark.integration
    def test_real_mesh_generation(self):
        generator = PyGmshMeshGenerator(mesh_size=0.5)

        params = {"x": [0, 1], "y": [0, 1], "z": [0, 1]}
        submesh = generator.generate("rectangular", params)

        assert isinstance(submesh, SubMesh3D)
        assert len(submesh.tetrahedra) > 0
        assert submesh.npts > 0


class TestSubMesh3DIntegration:
    def test_integration_with_pybamm_mesh(self):
        # x = pybamm.SpatialVariable("x", domain=["domain"], coord_sys="cartesian")
        # y = pybamm.SpatialVariable("y", domain=["domain"], coord_sys="cartesian")
        # z = pybamm.SpatialVariable("z", domain=["domain"], coord_sys="cartesian")

        # geometry = {
        #     "domain": {
        #         x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        #         y: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        #         z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        #     }
        # }

        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        cells = [("tetra", np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))]
        mesh_data = meshio.Mesh(points, cells)
        mock_submesh = SubMesh3D(mesh_data)

        mesh_dict = {"domain": mock_submesh}

        assert "domain" in mesh_dict
        assert isinstance(mesh_dict["domain"], SubMesh3D)
        assert len(mesh_dict["domain"].tetrahedra) == 2
        assert mesh_dict["domain"].npts == 2
