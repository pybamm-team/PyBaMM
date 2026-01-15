#
# Test for the Finite Volume Mesh class
#

import numpy as np
import pytest

import pybamm


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
            pybamm.GeometryError, match=r"Geometry should no longer be given keys"
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

        with pytest.raises(KeyError, match=r"Points not given"):
            pybamm.Mesh(geometry, submesh_types, {})

        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12}
        geometry = pybamm.battery_geometry(options={"dimensionality": 1})
        with pytest.raises(KeyError, match=r"Points not given"):
            pybamm.Mesh(geometry, submesh_types, var_pts)

        # Not processing geometry parameters
        geometry = pybamm.battery_geometry()

        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12, "r_n": 5, "r_p": 6}

        with pytest.raises(pybamm.DiscretisationError, match=r"Parameter values"):
            pybamm.Mesh(geometry, submesh_types, var_pts)

        # Geometry has an unrecognized variable type
        geometry["negative electrode"] = {
            "x_n": {"min": 0, "max": pybamm.Variable("var")}
        }
        with pytest.raises(NotImplementedError, match=r"for symbol var"):
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

        with pytest.raises(pybamm.DomainError, match=r"trying"):
            mesh.combine_submeshes("negative electrode", "negative particle")

        with pytest.raises(
            ValueError, match=r"Submesh domains being combined cannot be empty"
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

    def test_combine_submeshes_2d(self):
        # 2D geometry
        x = pybamm.SpatialVariable(
            "x", domain=["negative electrode", "separator"], direction="lr"
        )
        z = pybamm.SpatialVariable(
            "z", domain=["negative electrode", "separator"], direction="tb"
        )
        geometry = {
            "negative electrode": {
                x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
                z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            },
            "separator": {
                x: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)},
            },
        }
        submesh_types = {
            "negative electrode": pybamm.Uniform2DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
        }
        var_pts = {
            x: 10,
            z: 10,
        }
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        with pytest.raises(pybamm.GeometryError, match=r"Cannot combine"):
            mesh[("negative electrode", "separator")]

        geometry = {
            "negative electrode": {
                x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
                z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            },
            "separator": {
                x: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)},
                z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(0.5)},
            },
        }
        submesh_types = {
            "negative electrode": pybamm.Uniform2DSubMesh,
            "separator": pybamm.Uniform2DSubMesh,
        }
        var_pts = {
            x: 10,
            z: 10,
        }
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        with pytest.raises(pybamm.DomainError, match=r"lr edges are not aligned"):
            mesh[("negative electrode_left ghost cell", "separator")]

        geometry = {
            "left": {
                x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
                z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            },
            "right": {
                x: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)},
                z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(0.5)},
            },
        }
        submesh_types = {
            "left": pybamm.Uniform2DSubMesh,
            "right": pybamm.Uniform2DSubMesh,
        }
        var_pts = {
            x: 10,
            z: 10,
        }
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        with pytest.raises(pybamm.DomainError, match=r"tb edges are not aligned"):
            mesh[("left", "right")]

        geometry = {
            "top": {
                x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
                z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            },
            "bottom": {
                x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
                z: {"min": pybamm.Scalar(2), "max": pybamm.Scalar(3)},
            },
        }
        submesh_types = {
            "top": pybamm.Uniform2DSubMesh,
            "bottom": pybamm.Uniform2DSubMesh,
        }
        var_pts = {
            x: 10,
            z: 10,
        }
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        with pytest.raises(pybamm.DomainError, match=r"tb edges are not aligned"):
            mesh[("top", "bottom")]

        geometry = {
            "top": {
                x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
                z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            },
            "bottom": {
                x: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)},
                z: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)},
            },
        }
        submesh_types = {
            "top": pybamm.Uniform2DSubMesh,
            "bottom": pybamm.Uniform2DSubMesh,
        }
        var_pts = {
            x: 10,
            z: 10,
        }
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        with pytest.raises(pybamm.DomainError, match=r"lr edges are not aligned"):
            mesh[("top", "bottom")]

        geometry = {
            "top": {
                x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
                z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            },
            "bottom": {
                x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
                z: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)},
            },
        }
        submesh_types = {
            "top": pybamm.Uniform2DSubMesh,
            "bottom": pybamm.Uniform2DSubMesh,
        }
        var_pts = {
            x: 10,
            z: 10,
        }
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        assert mesh[("top", "bottom")].edges_lr[0] == 0
        assert mesh[("top", "bottom")].edges_lr[-1] == 1
        assert mesh[("top", "bottom")].edges_tb[0] == 0
        assert mesh[("top", "bottom")].edges_tb[-1] == 2

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

    def test_compute_var_pts_from_thicknesses_cell_size(self):
        from pybamm.meshes.meshes import compute_var_pts_from_thicknesses

        electrode_thicknesses = {
            "negative electrode": 100e-6,
            "separator": 25e-6,
            "positive electrode": 100e-6,
        }

        cell_size = 5e-6  # 5 micrometres per cell
        var_pts = compute_var_pts_from_thicknesses(electrode_thicknesses, cell_size)

        assert isinstance(var_pts, dict)
        assert all(isinstance(v, dict) for v in var_pts.values())
        assert var_pts["negative electrode"]["x_n"] == 20
        assert var_pts["separator"]["x_s"] == 5
        assert var_pts["positive electrode"]["x_p"] == 20

    def test_compute_var_pts_from_thicknesses_invalid_thickness_type(self):
        from pybamm.meshes.meshes import compute_var_pts_from_thicknesses

        with pytest.raises(TypeError):
            compute_var_pts_from_thicknesses(["not", "a", "dict"], 1e-6)

    def test_compute_var_pts_from_thicknesses_invalid_grid_size(self):
        from pybamm.meshes.meshes import compute_var_pts_from_thicknesses

        electrode_thicknesses = {"negative electrode": 100e-6}
        with pytest.raises(ValueError):
            compute_var_pts_from_thicknesses(electrode_thicknesses, -1e-6)


class TestMeshGenerator:
    def test_init_name(self):
        mesh_generator = pybamm.MeshGenerator(pybamm.SubMesh0D)
        assert mesh_generator.__repr__() == "Generator for SubMesh0D"
