#
# Test for the Finite Volume Mesh class
#

import pytest
import pybamm
import numpy as np


def get_param():
    return pybamm.ParameterValues(
        {
            "Negative electrode thickness [m]": 0.1,
            "Separator thickness [m]": 0.2,
            "Positive electrode thickness [m]": 0.3,
            "Negative particle radius [m]": 0.4,
            "Positive particle radius [m]": 0.5,
        }
    )


class TestMesh:
    @pytest.fixture(scope="class")
    def submesh_types(self):
        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.SubMesh0D,
        }
        return submesh_types

    def test_mesh_creation_no_parameters(self):
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        geometry = {
            "negative particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }

        submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
        var_pts = {r: 20}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check geometry
        assert mesh.geometry == geometry

        # check boundary locations
        assert mesh["negative particle"].edges[0] == 0
        assert mesh["negative particle"].edges[-1] == 1

        # check number of edges and nodes
        assert len(mesh["negative particle"].nodes) == var_pts[r]
        assert (
            len(mesh["negative particle"].edges)
            == len(mesh["negative particle"].nodes) + 1
        )

        # errors if old format
        geometry = {
            "negative particle": {
                "primary": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
            }
        }
        with pytest.raises(
            pybamm.GeometryError, match="Geometry should no longer be given keys"
        ):
            mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

    def test_mesh_creation(self, submesh_types):
        param = get_param()

        geometry = pybamm.battery_geometry()
        param.process_geometry(geometry)

        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12, "r_n": 5, "r_p": 6}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check geometry
        assert mesh.geometry == geometry

        # check boundary locations
        assert mesh["negative electrode"].edges[0] == 0
        assert mesh["positive electrode"].edges[-1] == pytest.approx(0.6)

        # check internal boundary locations
        assert mesh["negative electrode"].edges[-1] == mesh["separator"].edges[0]
        assert mesh["positive electrode"].edges[0] == mesh["separator"].edges[-1]
        for domain in mesh.base_domains:
            if domain != "current collector":
                assert len(mesh[domain].edges) == len(mesh[domain].nodes) + 1

    def test_init_failure(self, submesh_types):
        geometry = pybamm.battery_geometry()

        with pytest.raises(KeyError, match="Points not given"):
            pybamm.Mesh(geometry, submesh_types, {})

        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12}
        geometry = pybamm.battery_geometry(options={"dimensionality": 1})
        with pytest.raises(KeyError, match="Points not given"):
            pybamm.Mesh(geometry, submesh_types, var_pts)

        # Not processing geometry parameters
        geometry = pybamm.battery_geometry()

        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12, "r_n": 5, "r_p": 6}

        with pytest.raises(pybamm.DiscretisationError, match="Parameter values"):
            pybamm.Mesh(geometry, submesh_types, var_pts)

        # Geometry has an unrecognized variable type
        geometry["negative electrode"] = {
            "x_n": {"min": 0, "max": pybamm.Variable("var")}
        }
        with pytest.raises(NotImplementedError, match="for symbol var"):
            pybamm.Mesh(geometry, submesh_types, var_pts)

    def test_mesh_sizes(self, submesh_types):
        param = get_param()

        geometry = pybamm.battery_geometry()
        param.process_geometry(geometry)

        # provide mesh properties
        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12, "r_n": 5, "r_p": 6}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        assert mesh["negative electrode"].npts == var_pts["x_n"]
        assert mesh["separator"].npts == var_pts["x_s"]
        assert mesh["positive electrode"].npts == var_pts["x_p"]

        assert len(mesh["negative electrode"].edges) - 1 == var_pts["x_n"]
        assert len(mesh["separator"].edges) - 1 == var_pts["x_s"]
        assert len(mesh["positive electrode"].edges) - 1 == var_pts["x_p"]

    def test_mesh_sizes_using_standard_spatial_vars(self, submesh_types):
        param = get_param()

        geometry = pybamm.battery_geometry()
        param.process_geometry(geometry)

        # provide mesh properties
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 12, var.r_n: 5, var.r_p: 6}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        assert mesh["negative electrode"].npts == var_pts[var.x_n]
        assert mesh["separator"].npts == var_pts[var.x_s]
        assert mesh["positive electrode"].npts == var_pts[var.x_p]

        assert len(mesh["negative electrode"].edges) - 1 == var_pts[var.x_n]
        assert len(mesh["separator"].edges) - 1 == var_pts[var.x_s]
        assert len(mesh["positive electrode"].edges) - 1 == var_pts[var.x_p]

    def test_combine_submeshes(self, submesh_types):
        param = get_param()

        geometry = pybamm.battery_geometry()
        param.process_geometry(geometry)

        # provide mesh properties
        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12, "r_n": 5, "r_p": 6}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

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

        # test errors
        geometry = {
            "negative electrode": {"x_n": {"min": 0, "max": 0.5}},
            "negative particle": {"r_n": {"min": 0.5, "max": 1}},
        }
        param.process_geometry(geometry)

        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        with pytest.raises(pybamm.DomainError, match="trying"):
            mesh.combine_submeshes("negative electrode", "negative particle")

        with pytest.raises(
            ValueError, match="Submesh domains being combined cannot be empty"
        ):
            mesh.combine_submeshes()

        # test symbolic submesh
        new_submesh_types = submesh_types.copy()
        geometry = {
            "negative electrode": {
                "x_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}
            },
            "separator": {
                "x_s": {"min": pybamm.Scalar(1), "max": pybamm.InputParameter("L")}
            },
        }
        param.process_geometry(geometry)
        for k in new_submesh_types:
            new_submesh_types[k] = pybamm.SymbolicUniform1DSubMesh
        mesh = pybamm.Mesh(geometry, new_submesh_types, var_pts)
        submesh = mesh[("negative electrode", "separator")]
        mesh.combine_submeshes("negative electrode", "separator")
        assert (
            submesh.length
            == mesh["separator"].length + mesh["negative electrode"].length
        )
        assert submesh.min == mesh["negative electrode"].min

    def test_ghost_cells(self, submesh_types):
        param = get_param()

        geometry = pybamm.battery_geometry()
        param.process_geometry(geometry)

        # provide mesh properties
        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12, "r_n": 5, "r_p": 6}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

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

        # test symbolic mesh
        geometry = {
            "negative electrode": {
                "x_n": {"min": pybamm.Scalar(0), "max": pybamm.InputParameter("L_n")}
            },
            "separator": {
                "x_s": {
                    "min": pybamm.InputParameter("L_n"),
                    "max": pybamm.InputParameter("L"),
                }
            },
        }
        submesh_types = {
            "negative electrode": pybamm.SymbolicUniform1DSubMesh,
            "separator": pybamm.SymbolicUniform1DSubMesh,
        }
        var_pts = {
            "x_n": 10,
            "x_s": 10,
        }
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        np.testing.assert_array_equal(
            mesh["negative electrode_left ghost cell"].edges[1],
            mesh["negative electrode"].edges[0],
        )
        np.testing.assert_array_equal(
            mesh["negative electrode_left ghost cell"].edges[0],
            -mesh["negative electrode"].edges[1],
        )
        np.testing.assert_array_equal(
            mesh["separator_left ghost cell"].edges[1],
            mesh["separator"].edges[0],
        )
        np.testing.assert_array_equal(
            mesh["separator_left ghost cell"].edges[0],
            -mesh["separator"].edges[1],
        )

    def test_symbolic_mesh_ghost_cells(self, submesh_types):
        param = get_param()

        submesh_types.update({"negative electrode": pybamm.SymbolicUniform1DSubMesh})

        geometry = pybamm.battery_geometry()
        param.process_geometry(geometry)

        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12, "r_n": 5, "r_p": 6}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        lgs_submesh = mesh["negative electrode_left ghost cell"]
        rgs_submesh = mesh["negative electrode_right ghost cell"]
        submesh = mesh["negative electrode"]

        np.testing.assert_array_equal(
            lgs_submesh.edges[1] * lgs_submesh.length + lgs_submesh.min,
            submesh.edges[0],
        )
        np.testing.assert_array_equal(
            rgs_submesh.edges[0] * rgs_submesh.length + rgs_submesh.min,
            submesh.edges[-1] * submesh.length + submesh.min,
        )

    def test_mesh_coord_sys(self, submesh_types):
        param = get_param()

        geometry = pybamm.battery_geometry()
        param.process_geometry(geometry)

        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12, "r_n": 5, "r_p": 6}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        for submesh in mesh.values():
            if not isinstance(submesh, pybamm.SubMesh0D):
                assert submesh.coord_sys in pybamm.KNOWN_COORD_SYS

    def test_unimplemented_meshes(self):
        var_pts = {"x_n": 10, "y": 10}
        geometry = {
            "negative electrode": {
                "x_n": {"min": 0, "max": 1},
                "y": {"min": 0, "max": 1},
            }
        }
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
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        geometry = {
            "negative particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }

        submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
        var_pts = {r: 20}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        mesh_json = mesh.to_json()

        expected_json = {
            "submesh_pts": {"negative particle": {"r": 20}},
            "base_domains": ["negative particle"],
        }

        assert mesh_json == expected_json

    def setup_method(self):
        self.geometry = {
            "3d_lo": {
                "x": {"min": 0, "max": 1},
                "y": {"min": 0, "max": 1},
                "z": {"min": 0, "max": 1},
            },
            "3d_hi": {
                "x": {"min": 1, "max": 2},
                "y": {"min": 0, "max": 1},
                "z": {"min": 0, "max": 1},
            },
        }

        self.submesh_types = {
            "3d_lo": pybamm.Uniform3DSubMesh,
            "3d_hi": pybamm.Uniform3DSubMesh,
        }

        self.var_pts = {"x": 4, "y": 3, "z": 5}

    def test_create_mesh(self):
        param = get_param()
        param.process_geometry(self.geometry)
        mesh = pybamm.Mesh(self.geometry, self.submesh_types, self.var_pts)
        assert ("3d_lo",) in mesh
        assert ("3d_hi",) in mesh

        submesh = mesh[("3d_lo",)]
        # Check that we have the right number of edges in each dimension
        assert len(submesh.edges_x) == self.var_pts["x"] + 1  # 5 = var_pts["x"] + 1
        assert len(submesh.edges_y) == self.var_pts["y"] + 1  # 4 = var_pts["y"] + 1
        assert len(submesh.edges_z) == self.var_pts["z"] + 1  # 6 = var_pts["z"] + 1
        assert submesh.coord_sys == "cartesian"
        assert submesh.dimension == 3
        assert submesh.npts == self.var_pts["x"] * self.var_pts["y"] * self.var_pts["z"]

    def test_combine_3d_submeshes(self):
        # Use geometry that makes concatenation axis clear
        geometry = {
            "negative electrode": {
                "x": {"min": 0, "max": 1},
                "y": {"min": 0, "max": 1},
                "z": {"min": 0, "max": 1},
            },
            "separator": {
                "x": {"min": 1, "max": 2},  # Adjacent in x-direction
                "y": {"min": 0, "max": 1},
                "z": {"min": 0, "max": 1},
            },
        }
        submesh_types = {
            "negative electrode": pybamm.Uniform3DSubMesh,
            "separator": pybamm.Uniform3DSubMesh,
        }
        param = get_param()
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, submesh_types, self.var_pts)

        combined = mesh.combine_submeshes("negative electrode", "separator")
        assert combined.coord_sys == "cartesian"
        assert combined.dimension == 3
        # Check that x-dimension was extended (concatenated)
        assert len(combined.edges_x) == self.var_pts["x"] * 2 + 1
        # Y and Z should remain the same
        assert len(combined.edges_y) == self.var_pts["y"] + 1
        assert len(combined.edges_z) == self.var_pts["z"] + 1
        # Should have one internal boundary at x=1
        assert len(combined.internal_boundaries) == 1
        assert combined.internal_boundaries[0] == 1.0

    def test_combine_errors(self):
        # Test mismatched dimensions
        geometry = {
            "3d_domain": {
                "x": {"min": 0, "max": 1},
                "y": {"min": 0, "max": 1},
                "z": {"min": 0, "max": 1},
            },
            "1d_domain": {"r_n": {"min": 0, "max": 1}},
        }
        submesh_types = {
            "3d_domain": pybamm.Uniform3DSubMesh,
            "1d_domain": pybamm.Uniform1DSubMesh,
        }
        var_pts = {"x": 2, "y": 2, "z": 2, "r_n": 2}

        param = get_param()
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        with pytest.raises(pybamm.DomainError, match="different dimensions"):
            mesh.combine_submeshes("3d_domain", "1d_domain")

        with pytest.raises(ValueError, match="cannot be empty"):
            mesh.combine_submeshes()

    def test_3d_mesh_properties(self):
        """Test additional 3D-specific properties"""
        param = get_param()
        param.process_geometry(self.geometry)
        mesh = pybamm.Mesh(self.geometry, self.submesh_types, self.var_pts)

        submesh = mesh["3d_lo"]

        # Test node counts in each dimension
        assert submesh.npts_x == self.var_pts["x"]
        assert submesh.npts_y == self.var_pts["y"]
        assert submesh.npts_z == self.var_pts["z"]

        # Test total node count
        expected_total = self.var_pts["x"] * self.var_pts["y"] * self.var_pts["z"]
        assert submesh.npts == expected_total
        assert submesh.nodes.shape == (expected_total, 3)  # 3D coordinates

        # Test edge arrays are properly sized
        assert submesh.edges_x.shape == (self.var_pts["x"] + 1,)
        assert submesh.edges_y.shape == (self.var_pts["y"] + 1,)
        assert submesh.edges_z.shape == (self.var_pts["z"] + 1,)

    def test_3d_mesh_boundaries(self):
        geometry = {
            "left_domain": {
                "x": {"min": 0, "max": 1},
                "y": {"min": 0, "max": 1},
                "z": {"min": 0, "max": 1},
            },
            "right_domain": {
                "x": {"min": 1, "max": 2},  # Adjacent in x
                "y": {"min": 0, "max": 1},
                "z": {"min": 0, "max": 1},
            },
            "front_domain": {
                "x": {"min": 0, "max": 1},
                "y": {"min": 0, "max": 1},
                "z": {"min": 0, "max": 1},
            },
            "back_domain": {
                "x": {"min": 0, "max": 1},
                "y": {"min": 1, "max": 2},  # Adjacent in y to front_domain
                "z": {"min": 0, "max": 1},
            },
        }
        submesh_types = {k: pybamm.Uniform3DSubMesh for k in geometry.keys()}

        param = get_param()
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, submesh_types, self.var_pts)

        combined_x = mesh.combine_submeshes("left_domain", "right_domain")
        assert len(combined_x.edges_x) == 2 * self.var_pts["x"] + 1
        assert len(combined_x.internal_boundaries) == 1

        combined_y = mesh.combine_submeshes("front_domain", "back_domain")
        assert len(combined_y.edges_y) == 2 * self.var_pts["y"] + 1
        assert len(combined_y.internal_boundaries) == 1


class TestMeshGenerator:
    def test_init_name(self):
        mesh_generator = pybamm.MeshGenerator(pybamm.SubMesh0D)
        assert mesh_generator.__repr__() == "Generator for SubMesh0D"
