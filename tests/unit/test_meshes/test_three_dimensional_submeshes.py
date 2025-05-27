import pytest
import numpy as np
import pybamm

from pybamm.meshes.three_dimensional_submeshes import (
    SubMesh3D,
    Uniform3DSubMesh,
    ScikitFemGenerator3D,
)
from pybamm.meshes.meshes import Mesh


@pytest.fixture
def x():
    return pybamm.SpatialVariable("x", domain=["my 3d domain"], coord_sys="cartesian")


@pytest.fixture
def y():
    return pybamm.SpatialVariable("y", domain=["my 3d domain"], coord_sys="cartesian")


@pytest.fixture
def z():
    return pybamm.SpatialVariable("z", domain=["my 3d domain"], coord_sys="cartesian")


@pytest.fixture
def geometry(x, y, z):
    return {
        "my 3d domain": {
            x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            y: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(2)},
            z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(3)},
        }
    }


class TestSubMesh3DReadLims:
    def test_read_lims_wrong_number(self):
        lims = {"x": {}, "y": {}}
        with pytest.raises(pybamm.GeometryError):
            SubMesh3D.read_lims(lims.copy())

        lims = {"x": {}, "y": {}, "z": {}, "w": {}}
        with pytest.raises(pybamm.GeometryError):
            SubMesh3D.read_lims(lims.copy())

    def test_read_lims_string_keys(self, x, y, z):
        lims = {
            "x": {"min": 0, "max": 1},
            "y": {"min": 0, "max": 1},
            "z": {"min": 0, "max": 1},
            "tabs": {"foo": "bar"},
        }
        var_x, lims_x, var_y, lims_y, var_z, lims_z, tabs = SubMesh3D.read_lims(lims)
        assert var_x.name == x.name
        assert var_y.name == y.name
        assert var_z.name == z.name
        assert tabs == {"foo": "bar"}


class TestUniform3DSubMesh:
    def test_exception_wrong_lims(self):
        # missing one axis
        lims = {"x": {}, "y": {}}
        with pytest.raises(pybamm.GeometryError):
            Uniform3DSubMesh(lims, {"x": 1, "y": 1, "z": 1})

    def test_basic_uniform_mesh(self, geometry):
        pts = {"x": 4, "y": 5, "z": 6}
        sub = Uniform3DSubMesh(geometry["my 3d domain"], pts)

        # edges should span [0,1], [0,2], [0,3]
        assert np.allclose(sub.edges_x, np.linspace(0, 1, pts["x"] + 1))
        assert np.allclose(sub.edges_y, np.linspace(0, 2, pts["y"] + 1))
        assert np.allclose(sub.edges_z, np.linspace(0, 3, pts["z"] + 1))

        # node counts
        assert sub.npts_x == 4
        assert sub.npts_y == 5
        assert sub.npts_z == 6
        assert sub.npts == 4 * 5 * 6

        # nodes shape
        nodes = sub.nodes
        assert nodes.shape == (4 * 5 * 6, 3)
        # centers: first node in x is at (0.5/4, 0.5*2/5, 0.5*3/6)
        assert np.allclose(nodes[0], [0.5 / 4, 1 / 5, 1 / 4])

    def test_to_json_and_from_json(self, geometry):
        pts = {"x": 2, "y": 2, "z": 2}
        sub = Uniform3DSubMesh(geometry["my 3d domain"], pts)
        j = sub.to_json()
        # round-trip via the Mesh._from_json
        re_sub = SubMesh3D._from_json(j)
        assert np.allclose(re_sub.edges_x, sub.edges_x)
        assert np.allclose(re_sub.edges_y, sub.edges_y)
        assert np.allclose(re_sub.edges_z, sub.edges_z)

    def test_create_ghost_cells(self, geometry):
        pts = {"x": 3, "y": 3, "z": 3}
        sub = Uniform3DSubMesh(geometry["my 3d domain"], pts)
        # left & right
        left = sub.create_ghost_cell("left")
        assert left.edges_x[1] == sub.edges_x[0]
        assert left.edges_x[0] == 2 * sub.edges_x[0] - sub.edges_x[1]
        right = sub.create_ghost_cell("right")
        assert right.edges_x[0] == sub.edges_x[-1]
        # front & back
        front = sub.create_ghost_cell("front")
        assert front.edges_y[1] == sub.edges_y[0]
        back = sub.create_ghost_cell("back")
        assert back.edges_y[0] == sub.edges_y[-1]
        # bottom & top
        bottom = sub.create_ghost_cell("bottom")
        assert bottom.edges_z[1] == sub.edges_z[0]
        top = sub.create_ghost_cell("top")
        assert top.edges_z[0] == sub.edges_z[-1]


class TestMeshIntegration3D:
    def test_mesh_factory_uniform(self, geometry):
        submesh_types = {"my 3d domain": Uniform3DSubMesh}
        var_pts = {"x": 2, "y": 3, "z": 4}
        mesh = Mesh(geometry, submesh_types, var_pts)

        sub = mesh["my 3d domain"]
        # dimension & nodes
        assert sub.dimension == 3
        assert sub.nodes.shape == (2 * 3 * 4, 3)
        assert any("left ghost cell" in k[0] for k in mesh.keys())


def test_meshpy_box_generator(geometry):
    gen = ScikitFemGenerator3D("box", max_volume=1e-2)
    sub = gen(geometry["my 3d domain"], {})
    assert hasattr(sub, "nodes")
    assert hasattr(sub, "elements")
    assert sub.dimension == 3
