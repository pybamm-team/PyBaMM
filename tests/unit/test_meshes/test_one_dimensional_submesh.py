import numpy as np
import pytest

import pybamm


@pytest.fixture()
def r():
    r = pybamm.SpatialVariable(
        "r", domain=["negative particle"], coord_sys="spherical polar"
    )
    return r


@pytest.fixture()
def x():
    return pybamm.SpatialVariable(
        "x", domain=["negative electrode"], coord_sys="cartesian"
    )


@pytest.fixture()
def geometry(r):
    geometry = {
        "negative particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
    }
    return geometry


class TestSubMesh1D:
    def test_tabs(self):
        edges = np.linspace(0, 1, 10)
        tabs = {"negative": {"z_centre": 0}, "positive": {"z_centre": 1}}
        mesh = pybamm.SubMesh1D(edges, None, tabs=tabs)
        assert mesh.tabs["negative tab"] == "left"
        assert mesh.tabs["positive tab"] == "right"

    def test_ghost_cell_top_bottom_error(self):
        edges = np.linspace(0, 1, 10)
        tabs = {"negative": {"z_centre": 0}, "positive": {"z_centre": 1}}
        mesh = pybamm.SubMesh1D(edges, None, tabs=tabs)
        with pytest.raises(NotImplementedError, match=r"left and right ghost cells"):
            mesh.create_ghost_cell("top")

    def test_exceptions(self):
        edges = np.linspace(0, 1, 10)
        tabs = {"negative": {"z_centre": 0.2}, "positive": {"z_centre": 1}}
        with pytest.raises(pybamm.GeometryError):
            pybamm.SubMesh1D(edges, None, tabs=tabs)

    def test_to_json(self):
        edges = np.linspace(0, 1, 10)
        tabs = {"negative": {"z_centre": 0}, "positive": {"z_centre": 1}}
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


class TestUniform1DSubMesh:
    def test_exceptions(self):
        lims = {"a": 1, "b": 2}
        with pytest.raises(pybamm.GeometryError):
            pybamm.Uniform1DSubMesh(lims, None)

    def test_symmetric_mesh_creation_no_parameters(self, r, geometry):
        submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
        var_pts = {r: 20}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        assert mesh["negative particle"].edges[0] == 0
        assert mesh["negative particle"].edges[-1] == 1

        # check number of edges and nodes
        assert len(mesh["negative particle"].nodes) == var_pts[r]
        assert (
            len(mesh["negative particle"].edges)
            == len(mesh["negative particle"].nodes) + 1
        )


class TestSymbolicUniform1DSubMesh:
    def test_exceptions(self, r):
        lims = {"a": 1, "b": 2}
        with pytest.raises(pybamm.GeometryError):
            pybamm.SymbolicUniform1DSubMesh(lims, None)
        lims = {"x_n": {"min": 0, "max": 1}}
        npts = {"x_n": 10}
        tabs = {"negative": {"z_centre": 0}, "positive": {"z_centre": 1}}
        lims["tabs"] = tabs

        with pytest.raises(NotImplementedError):
            pybamm.SymbolicUniform1DSubMesh(lims, npts, tabs=tabs)

        submesh_types = {"negative particle": pybamm.SymbolicUniform1DSubMesh}
        var_pts = {r: 20}
        geometry = {
            "negative particle": {
                r: {"min": pybamm.InputParameter("min"), "max": pybamm.Scalar(2)}
            }
        }
        with pytest.raises(pybamm.GeometryError):
            pybamm.Mesh(geometry, submesh_types, var_pts)

    def test_mesh_creation(self, r, x):
        submesh_types = {"negative particle": pybamm.SymbolicUniform1DSubMesh}
        var_pts = {r: 20}
        geometry = {
            "negative particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(2)}}
        }

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        assert mesh["negative particle"].edges[0] == 0
        assert mesh["negative particle"].edges[-1] == 1

        # check scaling and min/max
        assert mesh["negative particle"].length == 2
        assert mesh["negative particle"].min == 0

        # check number of edges and nodes
        assert len(mesh["negative particle"].nodes) == var_pts[r]
        assert (
            len(mesh["negative particle"].edges)
            == len(mesh["negative particle"].nodes) + 1
        )

        # Check that length and min are scaled correctly
        submesh_types = {"negative electrode": pybamm.SymbolicUniform1DSubMesh}
        var_pts = {x: 20}
        geometry = {
            "negative electrode": {
                x: {"min": pybamm.InputParameter("min"), "max": pybamm.Scalar(2)}
            }
        }
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        assert mesh["negative electrode"].length == pybamm.Scalar(
            2
        ) - pybamm.InputParameter("min")
        assert mesh["negative electrode"].min == pybamm.InputParameter("min")


class TestExponential1DSubMesh:
    @pytest.mark.parametrize("odd_even", [20, 21])
    def test_symmetric_mesh_creation(self, r, geometry, odd_even):
        submesh_params = {"side": "symmetric", "stretch": 1.5}
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(
                pybamm.Exponential1DSubMesh, submesh_params
            )
        }
        var_pts = {r: odd_even}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        assert mesh["negative particle"].edges[0] == 0
        assert mesh["negative particle"].edges[-1] == 1

        # check number of edges and nodes
        assert len(mesh["negative particle"].nodes) == var_pts[r]
        assert (
            len(mesh["negative particle"].edges)
            == len(mesh["negative particle"].nodes) + 1
        )

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_mesh_creation_with_side(self, r, geometry, side):
        submesh_params = {"side": side}
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(
                pybamm.Exponential1DSubMesh, submesh_params
            )
        }
        var_pts = {r: 20}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        assert mesh["negative particle"].edges[0] == 0
        assert mesh["negative particle"].edges[-1] == 1

        # check number of edges and nodes
        assert len(mesh["negative particle"].nodes) == var_pts[r]
        assert (
            len(mesh["negative particle"].edges)
            == len(mesh["negative particle"].nodes) + 1
        )

        # check monotonicity based on side
        if side == "left":
            # spacing should increase from left to right
            assert np.all(np.diff(np.diff(mesh["negative particle"].edges)) > 0)
        else:  # right
            # spacing should decrease from left to right
            assert np.all(np.diff(np.diff(mesh["negative particle"].edges)) < 0)

    @pytest.mark.parametrize("side", ["left", "right", "symmetric"])
    def test_mesh_creation_non_zero_min(self, r, side):
        geometry = {
            "negative particle": {r: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}}
        }
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(
                pybamm.Exponential1DSubMesh, {"side": side}
            )
        }
        var_pts = {r: 20}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        assert mesh["negative particle"].edges[0] == 1
        assert mesh["negative particle"].edges[-1] == 2
        # check number of edges and nodes
        assert len(mesh["negative particle"].nodes) == var_pts[r]
        assert (
            len(mesh["negative particle"].edges)
            == len(mesh["negative particle"].nodes) + 1
        )
        # check monotonically increasing
        assert np.all(np.diff(mesh["negative particle"].edges) > 0)


class TestChebyshev1DSubMesh:
    def test_mesh_creation_no_parameters(self, r, geometry):
        submesh_types = {"negative particle": pybamm.Chebyshev1DSubMesh}
        var_pts = {r: 20}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        assert mesh["negative particle"].edges[0] == 0
        assert mesh["negative particle"].edges[-1] == 1

        # check number of edges and nodes
        assert len(mesh["negative particle"].nodes) == var_pts[r]
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
        lims = {"x_n": {"min": 0, "max": 1}}
        npts = {"x_n": 10}
        with pytest.raises(pybamm.GeometryError):
            mesh(lims, npts)

        # error if lims[0] not equal to edges[0]
        lims = {"x_n": {"min": 0.1, "max": 1}}
        npts = {"x_n": len(edges) - 1}
        with pytest.raises(pybamm.GeometryError):
            mesh(lims, npts)

        # error if lims[-1] not equal to edges[-1]
        lims = {"x_n": {"min": 0, "max": 10}}
        npts = {"x_n": len(edges) - 1}
        with pytest.raises(pybamm.GeometryError):
            mesh(lims, npts)

        # no user points
        mesh = pybamm.MeshGenerator(pybamm.UserSupplied1DSubMesh)
        with pytest.raises(pybamm.GeometryError, match=r"User mesh requires"):
            mesh(None, None)

    def test_mesh_creation_no_parameters(self, r, geometry):
        edges = np.array([0, 0.3, 1])
        submesh_params = {"edges": edges}
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(
                pybamm.UserSupplied1DSubMesh, submesh_params
            )
        }
        var_pts = {r: len(edges) - 1}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        assert mesh["negative particle"].edges[0] == 0
        assert mesh["negative particle"].edges[-1] == 1

        # check number of edges and nodes
        assert len(mesh["negative particle"].nodes) == var_pts[r]
        assert (
            len(mesh["negative particle"].edges)
            == len(mesh["negative particle"].nodes) + 1
        )


class TestSpectralVolume1DSubMesh:
    def test_exceptions(self):
        edges = np.array([0, 0.3, 1])
        submesh_params = {"edges": edges}
        mesh = pybamm.MeshGenerator(pybamm.SpectralVolume1DSubMesh, submesh_params)

        # error if npts+1 != len(edges)
        lims = {"x_n": {"min": 0, "max": 1}}
        npts = {"x_n": 10}
        with pytest.raises(pybamm.GeometryError):
            mesh(lims, npts)

        # error if lims[0] not equal to edges[0]
        lims = {"x_n": {"min": 0.1, "max": 1}}
        npts = {"x_n": len(edges) - 1}
        with pytest.raises(pybamm.GeometryError):
            mesh(lims, npts)

        # error if lims[-1] not equal to edges[-1]
        lims = {"x_n": {"min": 0, "max": 10}}
        npts = {"x_n": len(edges) - 1}
        with pytest.raises(pybamm.GeometryError):
            mesh(lims, npts)

    def test_mesh_creation_no_parameters(self, r, geometry):
        edges = np.array([0, 0.3, 1])
        order = 3
        submesh_params = {"edges": edges, "order": order}
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(
                pybamm.SpectralVolume1DSubMesh, submesh_params
            )
        }
        var_pts = {r: len(edges) - 1}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        assert mesh["negative particle"].edges[0] == 0
        assert mesh["negative particle"].edges[-1] == 1

        # check number of edges and nodes
        assert len(mesh["negative particle"].sv_nodes) == var_pts[r]
        assert len(mesh["negative particle"].nodes) == order * var_pts[r]
        assert (
            len(mesh["negative particle"].edges)
            == len(mesh["negative particle"].nodes) + 1
        )

        # check Chebyshev subdivision locations
        for a, b in zip(
            mesh["negative particle"].edges.tolist(),
            [0, 0.075, 0.225, 0.3, 0.475, 0.825, 1],
            strict=False,
        ):
            assert a == pytest.approx(b)

        # test uniform submesh creation
        submesh_params = {"order": order}
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(
                pybamm.SpectralVolume1DSubMesh, submesh_params
            )
        }
        var_pts = {r: 2}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        for a, b in zip(
            mesh["negative particle"].edges.tolist(),
            [0.0, 0.125, 0.375, 0.5, 0.625, 0.875, 1.0],
            strict=False,
        ):
            assert a == pytest.approx(b)
