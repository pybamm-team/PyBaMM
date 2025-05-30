import pytest
import pybamm


@pytest.fixture()
def x():
    x = pybamm.SpatialVariable("x", domain=["my 2d domain"], coord_sys="cartesian")
    return x


@pytest.fixture()
def y():
    return pybamm.SpatialVariable("y", domain=["my 2d domain"], coord_sys="cartesian")


@pytest.fixture()
def geometry(x, y):
    geometry = {
        "my 2d domain": {
            x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            y: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
    }
    return geometry


class TestUniform2DSubMesh:
    def test_exceptions(self):
        lims = {"a": 1}
        with pytest.raises(pybamm.GeometryError):
            pybamm.Uniform2DSubMesh(lims, None)

    def test_symmetric_mesh_creation_no_parameters(self, x, y, geometry):
        submesh_types = {"my 2d domain": pybamm.Uniform2DSubMesh}
        var_pts = {x: 20, y: 20}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        assert mesh["my 2d domain"].edges_lr[0] == 0
        assert mesh["my 2d domain"].edges_lr[-1] == 1
        assert mesh["my 2d domain"].edges_tb[0] == 0
        assert mesh["my 2d domain"].edges_tb[-1] == 1

        # check number of edges and nodes
        assert len(mesh["my 2d domain"].nodes_lr) == var_pts[x]
        assert len(mesh["my 2d domain"].nodes_tb) == var_pts[y]
        assert (
            len(mesh["my 2d domain"].edges_lr) == len(mesh["my 2d domain"].nodes_lr) + 1
        )
        assert (
            len(mesh["my 2d domain"].edges_tb) == len(mesh["my 2d domain"].nodes_tb) + 1
        )
