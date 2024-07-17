import pybamm
import pytest


class TestSubMesh0D:
    def test_exceptions(self):
        position = {"x": {"position": 0}, "y": {"position": 0}}
        with pytest.raises(pybamm.GeometryError):
            pybamm.SubMesh0D(position)

    def test_init(self):
        position = {"x": {"position": 1}}
        generator = pybamm.SubMesh0D
        mesh = generator(position, None)
        mesh.add_ghost_meshes()
