#
# Test for the Finite Volume Mesh class
#

import pytest
import pybamm
import numpy as np


class TestMesh:
    @pytest.fixture(scope="class")
    def parameter_values(cls):
        return pybamm.ParameterValues(
            {
                "Negative electrode thickness [m]": 0.1,
                "Separator thickness [m]": 0.2,
                "Positive electrode thickness [m]": 0.3,
                "Negative particle radius [m]": 0.4,
                "Positive particle radius [m]": 0.5,
            }
        )

    @pytest.fixture(scope="class")
    def submesh_types(cls):
        return {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.SubMesh0D,
        }

    @pytest.fixture(scope="class")
    def var_pts(cls):
        return {
            "negative electrode": 10,
            "separator": 10,
            "positive electrode": 12,
            "negative particle": 5,
            "positive particle": 6,
        }

    @pytest.fixture(scope="class")
    def mesh(cls, parameter_values, submesh_types, var_pts):
        geometry = pybamm.battery_geometry()
        parameter_values.process_geometry(geometry)
        return pybamm.Mesh(geometry, submesh_types, var_pts)

    def test_mesh_creation_no_parameters(self, submesh_types, var_pts):
        geometry = {"negative particle": {"r": (0, 1)}}

        var_pts = {"negative particle": 20}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        assert mesh["negative particle"].edges[0] == 0
        assert mesh["negative particle"].edges[-1] == 1

        # check number of edges and nodes
        assert len(mesh["negative particle"].nodes) == var_pts["negative particle"]
        assert (
            len(mesh["negative particle"].edges)
            == len(mesh["negative particle"].nodes) + 1
        )

    def test_mesh_creation(self, mesh):
        # check boundary locations
        assert mesh["negative electrode"].edges[0] == 0
        assert mesh["positive electrode"].edges[-1] == pytest.approx(0.6)

        # check internal boundary locations
        assert mesh["negative electrode"].edges[-1] == mesh["separator"].edges[0]
        assert mesh["positive electrode"].edges[0] == mesh["separator"].edges[-1]
        for domain in mesh.base_domains:
            if domain != "current collector":
                assert len(mesh[domain].edges) == len(mesh[domain].nodes) + 1

    def test_init_failure(self, submesh_types, var_pts):
        geometry = pybamm.battery_geometry()

        with pytest.raises(KeyError, match="Points not given"):
            pybamm.Mesh(geometry, submesh_types, {})

        geometry = pybamm.battery_geometry(options={"dimensionality": 1})
        with pytest.raises(KeyError, match="Points not given"):
            pybamm.Mesh(geometry, submesh_types, var_pts)

        # Not processing geometry parameters
        geometry = pybamm.battery_geometry()

        with pytest.raises(pybamm.DiscretisationError, match="Parameter values"):
            pybamm.Mesh(geometry, submesh_types, var_pts)

        # Geometry has an unrecognized variable type
        geometry["negative electrode"] = {"x": (0, pybamm.Variable("var"))}
        with pytest.raises(NotImplementedError, match="for symbol var"):
            pybamm.Mesh(geometry, submesh_types, var_pts)

    def test_mesh_sizes(self, mesh, var_pts):
        assert mesh["negative electrode"].npts == var_pts["negative electrode"]
        assert mesh["separator"].npts == var_pts["separator"]
        assert mesh["positive electrode"].npts == var_pts["positive electrode"]

        assert (
            len(mesh["negative electrode"].edges) - 1 == var_pts["negative electrode"]
        )
        assert len(mesh["separator"].edges) - 1 == var_pts["separator"]
        assert (
            len(mesh["positive electrode"].edges) - 1 == var_pts["positive electrode"]
        )

    def test_combine_submeshes(self, mesh):
        # create submesh
        submesh = mesh[("negative electrode", "separator")]
        assert submesh.edges[0] == 0
        assert submesh.edges[-1] == mesh["separator"].edges[-1]
        np.testing.assert_almost_equal(
            submesh.nodes
            - np.concatenate(
                [mesh["negative electrode"].nodes, mesh["separator"].nodes]
            ),
            0,
        )
        assert submesh.internal_boundaries == [0.1]
        with pytest.raises(pybamm.DomainError):
            mesh.combine_submeshes("negative electrode", "positive electrode")

    def test_combine_submeshes_errors(self, parameter_values, submesh_types, var_pts):
        # test errors
        geometry = {
            "negative electrode": {"x": (0, 0.5)},
            "negative particle": {"r": (0.5, 1)},
        }
        parameter_values.process_geometry(geometry)

        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        with pytest.raises(pybamm.DomainError, match="trying"):
            mesh.combine_submeshes("negative electrode", "negative particle")

        with pytest.raises(
            ValueError, match="Submesh domains being combined cannot be empty"
        ):
            mesh.combine_submeshes()

    def test_ghost_cells(self, mesh):
        np.testing.assert_array_equal(
            mesh["negative electrode_left ghost cell"].edges[1],
            mesh["negative electrode"].edges[0],
        )
        np.testing.assert_array_equal(
            mesh["negative electrode_left ghost cell"].edges[0],
            -mesh["negative electrode"].edges[1],
        )
        np.testing.assert_array_equal(
            mesh["positive electrode_right ghost cell"].edges[0],
            mesh["positive electrode"].edges[-1],
        )

    def test_unimplemented_meshes(self):
        var_pts = {"negative electrode": (10, 10)}
        geometry = {"negative electrode": {"x": (0, 1), "y": (0, 1)}}
        submesh_types = {"negative electrode": pybamm.Uniform1DSubMesh}
        with pytest.raises(pybamm.GeometryError):
            pybamm.Mesh(geometry, submesh_types, var_pts)

    def test_1plus1_d_tabs_left_right(self):
        param = pybamm.ParameterValues(
            values={
                "Electrode width [m]": 0.4,
                "Electrode height [m]": 0.5,
                "Negative tab centre z-coordinate [m]": 0.0,
                "Positive tab centre z-coordinate [m]": 0.5,
                "Negative electrode thickness [m]": 0.3,
                "Separator thickness [m]": 0.3,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        geometry = pybamm.battery_geometry(
            include_particles=False, options={"dimensionality": 1}
        )
        param.process_geometry(geometry)

        var_pts = {"x_n": 10, "x_s": 7, "x_p": 12, "z": 24}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.Uniform1DSubMesh,
        }

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # negative tab should be "left"
        assert mesh["current collector"].tabs["negative tab"] == "left"

        # positive tab should be "right"
        assert mesh["current collector"].tabs["positive tab"] == "right"

    def test_1plus1_d_tabs_right_left(self):
        param = pybamm.ParameterValues(
            values={
                "Electrode width [m]": 0.4,
                "Electrode height [m]": 0.5,
                "Negative tab centre z-coordinate [m]": 0.5,
                "Positive tab centre z-coordinate [m]": 0.0,
                "Negative electrode thickness [m]": 0.3,
                "Separator thickness [m]": 0.3,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        geometry = pybamm.battery_geometry(
            include_particles=False, options={"dimensionality": 1}
        )
        param.process_geometry(geometry)

        var_pts = {"x_n": 10, "x_s": 7, "x_p": 12, "z": 24}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.Uniform1DSubMesh,
        }

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # negative tab should be "right"
        assert mesh["current collector"].tabs["negative tab"] == "right"

        # positive tab should be "left"
        assert mesh["current collector"].tabs["positive tab"] == "left"

    def test_to_json(self):
        geometry = {"negative particle": {"r": (0, 1)}}

        submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
        var_pts = {"negative particle": 20}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        mesh_json = mesh.to_json()

        expected_json = {
            "submesh_pts": {"negative particle": 20},
            "base_domains": ["negative particle"],
        }

        assert mesh_json == expected_json


class TestMeshGenerator:
    def test_init_name(self):
        mesh_generator = pybamm.MeshGenerator(pybamm.SubMesh0D)
        assert mesh_generator.__repr__() == "Generator for SubMesh0D"
