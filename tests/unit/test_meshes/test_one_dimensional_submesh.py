import pytest
import pybamm
import numpy as np


@pytest.fixture()
def geometry():
    geometry = {"negative particle": {"r": (0, 1)}}
    return geometry


@pytest.fixture()
def tabs():
    tabs = {"negative": {"z_centre": 0}, "positive": {"z_centre": 1}}
    return tabs


class TestSubMesh1D:
    def test_tabs(self, tabs):
        edges = np.linspace(0, 1, 10)
        mesh = pybamm.SubMesh1D(edges, None, tabs=tabs)
        assert mesh.tabs["negative tab"] == "left"
        assert mesh.tabs["positive tab"] == "right"

    def test_exceptions(self):
        edges = np.linspace(0, 1, 10)
        tabs = {"negative": {"z_centre": 0.2}, "positive": {"z_centre": 1}}
        with pytest.raises(pybamm.GeometryError):
            pybamm.SubMesh1D(edges, None, tabs=tabs)

    def test_to_json(self, tabs):
        edges = np.linspace(0, 1, 10)
        mesh = pybamm.SubMesh1D(edges, None, tabs=tabs)

        mesh_json = mesh.to_json()

        expected_json = {
            "edges": [
                0.0,
                0.1111111111111111,
                0.2222222222222222,
                0.3333333333333333,
                0.4444444444444444,
                0.5555555555555556,
                0.6666666666666666,
                0.7777777777777777,
                0.8888888888888888,
                1.0,
            ],
            "coord_sys": None,
            "tabs": {"negative tab": "left", "positive tab": "right"},
        }

        assert mesh_json == expected_json

        # check tabs work
        new_mesh = pybamm.Uniform1DSubMesh._from_json(mesh_json)
        assert mesh.tabs == new_mesh.tabs


class Test1DSubMeshes:
    @pytest.mark.parametrize(
        "submesh_type",
        [
            "uniform",
            "exponential",
            "exponential_stretch",
            "exponential_left",
            "exponential_right",
            "chebyshev",
            "user",
        ],
    )
    def test_mesh_creation(self, geometry, submesh_type):
        if submesh_type == "uniform":
            submesh = pybamm.Uniform1DSubMesh
        elif submesh_type == "exponential":
            submesh_params = {"side": "symmetric"}
            submesh = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params)
        elif submesh_type == "exponential_stretch":
            submesh_params = {"side": "symmetric", "stretch": 1.5}
            submesh = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params)
        elif submesh_type == "exponential_left":
            submesh_params = {"side": "left"}
            submesh = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params)
        elif submesh_type == "exponential_right":
            submesh_params = {"side": "right", "stretch": 2}
            submesh = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params)
        elif submesh_type == "chebyshev":
            submesh = pybamm.Chebyshev1DSubMesh
        elif submesh_type == "user":
            edges = np.array([0, 0.3, 1])
            submesh_params = {"edges": edges}
            submesh = pybamm.MeshGenerator(pybamm.UserSupplied1DSubMesh, submesh_params)
        # test odd number of nodes for some of the exponential submeshes
        if submesh_type in ["exponential", "exponential_stretch"]:
            var_pts = {"negative particle": 21}
        elif submesh_type == "user":
            var_pts = {"negative particle": len(edges) - 1}
        else:
            var_pts = {"negative particle": 20}

        submesh_types = {"negative particle": submesh}

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


class TestUser1DSubMesh:
    def test_exceptions(self):
        edges = np.array([0, 0.3, 1])
        submesh_params = {"edges": edges}
        mesh = pybamm.MeshGenerator(pybamm.UserSupplied1DSubMesh, submesh_params)

        # error if npts+1 != len(edges)
        domain = pybamm.Domain({"x": (0, 1)})
        npts = 10
        with pytest.raises(pybamm.GeometryError):
            mesh(domain, npts)

        # error if lims[0] not equal to edges[0]
        domain = pybamm.Domain({"x": (0.1, 1)})
        npts = len(edges) - 1
        with pytest.raises(pybamm.GeometryError):
            mesh(domain, npts)

        # error if lims[-1] not equal to edges[-1]
        domain = pybamm.Domain({"x": (0, 10)})
        npts = len(edges) - 1
        with pytest.raises(pybamm.GeometryError):
            mesh(domain, npts)

        # no user points
        mesh = pybamm.MeshGenerator(pybamm.UserSupplied1DSubMesh)
        with pytest.raises(pybamm.GeometryError, match="User mesh requires"):
            mesh(None, None)


class TestSpectralVolume1DSubMesh:
    def test_exceptions(self):
        edges = np.array([0, 0.3, 1])
        submesh_params = {"edges": edges}
        mesh = pybamm.MeshGenerator(pybamm.SpectralVolume1DSubMesh, submesh_params)

        # error if npts+1 != len(edges)
        domain = pybamm.Domain({"x": (0, 1)})
        npts = 10
        with pytest.raises(pybamm.GeometryError):
            mesh(domain, npts)

        # error if lims[0] not equal to edges[0]
        domain = pybamm.Domain({"x": (0.1, 1)})
        npts = len(edges) - 1
        with pytest.raises(pybamm.GeometryError):
            mesh(domain, npts)

        # error if lims[-1] not equal to edges[-1]
        domain = pybamm.Domain({"x": (0, 10)})
        npts = len(edges) - 1
        with pytest.raises(pybamm.GeometryError):
            mesh(domain, npts)

    def test_mesh_creation_no_parameters(self, geometry):
        edges = np.array([0, 0.3, 1])
        order = 3
        submesh_params = {"edges": edges, "order": order}
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(
                pybamm.SpectralVolume1DSubMesh, submesh_params
            )
        }
        var_pts = {"negative particle": len(edges) - 1}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        assert mesh["negative particle"].edges[0] == 0
        assert mesh["negative particle"].edges[-1] == 1

        # check number of edges and nodes
        assert len(mesh["negative particle"].sv_nodes) == var_pts["negative particle"]
        assert (
            len(mesh["negative particle"].nodes) == order * var_pts["negative particle"]
        )
        assert (
            len(mesh["negative particle"].edges)
            == len(mesh["negative particle"].nodes) + 1
        )

        # check Chebyshev subdivision locations
        for a, b in zip(
            mesh["negative particle"].edges.tolist(),
            [0, 0.075, 0.225, 0.3, 0.475, 0.825, 1],
        ):
            assert a == pytest.approx(b)

        # test uniform submesh creation
        submesh_params = {"order": order}
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(
                pybamm.SpectralVolume1DSubMesh, submesh_params
            )
        }
        var_pts = {"negative particle": 2}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        for a, b in zip(
            mesh["negative particle"].edges.tolist(),
            [0.0, 0.125, 0.375, 0.5, 0.625, 0.875, 1.0],
        ):
            assert a == pytest.approx(b)
