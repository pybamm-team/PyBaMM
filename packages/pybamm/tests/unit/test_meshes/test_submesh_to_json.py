#
# Tests for SubMesh and MeshGenerator to_config / from_config
#


import pybamm
from pybamm.expression_tree.operations.serialise import Serialise


class TestSubMeshToConfig:
    """Tests for SubMesh class to_config and from_config."""

    def test_submesh_to_config_returns_class_and_module(self):
        """SubMesh subclass to_config() returns the kernel class_reference shape."""
        data = pybamm.Uniform1DSubMesh.to_config()
        assert isinstance(data, dict)
        assert data["$type"] == "type"
        assert (
            data["class"] == "pybamm.meshes.one_dimensional_submeshes.Uniform1DSubMesh"
        )

    def test_submesh_from_config_returns_same_class(self):
        """SubMesh.from_config(to_config()) returns the same class."""
        cls = pybamm.Uniform1DSubMesh
        data = cls.to_config()
        restored = pybamm.SubMesh.from_config(data)
        assert restored is cls

    def test_submesh_round_trip_submesh0d(self):
        """SubMesh0D round-trips via to_config/from_config."""
        data = pybamm.SubMesh0D.to_config()
        assert data["class"] == "pybamm.meshes.zero_dimensional_submesh.SubMesh0D"
        assert pybamm.SubMesh.from_config(data) is pybamm.SubMesh0D


class TestMeshGeneratorToConfig:
    """Tests for MeshGenerator to_config and from_config."""

    def test_mesh_generator_to_config_has_class_module(self):
        """MeshGenerator.to_config() contains the kernel class_reference."""
        gen = pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)
        data = gen.to_config()
        assert data["$type"] == "type"
        assert (
            data["class"] == "pybamm.meshes.one_dimensional_submeshes.Uniform1DSubMesh"
        )

    def test_mesh_generator_to_config_includes_submesh_params_when_non_empty(self):
        """MeshGenerator with submesh_params includes them in to_config()."""
        params = {"some_key": 1}
        gen = pybamm.MeshGenerator(pybamm.Uniform1DSubMesh, submesh_params=params)
        data = gen.to_config()
        assert data.get("submesh_params") == params

    def test_mesh_generator_from_config_same_type_and_params(self):
        """MeshGenerator.from_config(to_config()) returns equivalent generator."""
        gen = pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)
        restored = pybamm.MeshGenerator.from_config(gen.to_config())
        assert restored.submesh_type is gen.submesh_type
        assert restored.submesh_params == gen.submesh_params

    def test_mesh_generator_from_config_with_params(self):
        """MeshGenerator with submesh_params round-trips correctly."""
        params = {"a": 1, "b": 2}
        gen = pybamm.MeshGenerator(pybamm.Uniform1DSubMesh, submesh_params=params)
        restored = pybamm.MeshGenerator.from_config(gen.to_config())
        assert restored.submesh_type is pybamm.Uniform1DSubMesh
        assert restored.submesh_params == params


class TestSubmeshTypesDictRoundTrip:
    """Dict round-trip tests (classes and MeshGenerators)."""

    def test_dict_round_trip_classes(self):
        """Dict of SubMesh classes round-trips via to_config/from_config."""
        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.SubMesh0D,
        }
        serialised = {k: v.to_config() for k, v in submesh_types.items()}
        restored = {k: pybamm.SubMesh.from_config(v) for k, v in serialised.items()}
        assert set(restored.keys()) == set(submesh_types.keys())
        for domain, submesh_type in submesh_types.items():
            assert restored[domain] is submesh_type

    def test_dict_round_trip_mesh_generators(self):
        """Dict of MeshGenerator instances round-trips correctly."""
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "current collector": pybamm.MeshGenerator(
                pybamm.SubMesh0D, submesh_params={"x": 1}
            ),
        }
        serialised = {k: v.to_config() for k, v in submesh_types.items()}
        restored = {
            k: pybamm.MeshGenerator.from_config(v) for k, v in serialised.items()
        }
        assert set(restored.keys()) == set(submesh_types.keys())
        for domain, submesh_type in submesh_types.items():
            assert restored[domain].submesh_type is submesh_type.submesh_type
            assert restored[domain].submesh_params == submesh_type.submesh_params


class TestSubmeshSerialiseCompatibility:
    """Compatibility between SubMesh/MeshGenerator to_config and Serialise."""

    def test_submesh_to_config_matches_serialise_submesh_item(self):
        """SubMesh.to_config() matches Serialise.serialise_submesh_item(cls)."""
        cls = pybamm.Uniform1DSubMesh
        assert cls.to_config() == Serialise.serialise_submesh_item(cls)

    def test_mesh_generator_to_config_matches_serialise_submesh_item(self):
        """MeshGenerator.to_config() matches Serialise.serialise_submesh_item(self)."""
        gen = pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)
        assert gen.to_config() == Serialise.serialise_submesh_item(gen)

    def test_full_dict_serialise_load_matches_mesh_generator_from_config(self):
        """Full dict via serialise_submesh_types/load_submesh_types matches MeshGenerator.from_config per entry."""
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.Uniform1DSubMesh,  # class, not generator
        }
        full_json = Serialise.serialise_submesh_types(submesh_types)
        loaded = Serialise.load_submesh_types(full_json)
        assert set(loaded.keys()) == set(submesh_types.keys())
        for domain, submesh_info in full_json["submesh_types"].items():
            from_item = pybamm.MeshGenerator.from_config(submesh_info)
            assert loaded[domain].submesh_type is from_item.submesh_type
            assert loaded[domain].submesh_params == from_item.submesh_params


class TestPrebuiltSubMeshToConfig:
    """A prebuilt UnstructuredSubMesh in submesh_types carries its own data."""

    @staticmethod
    def _submesh():
        from pybamm.meshes.unstructured_submesh import _hex_grid

        submesh = pybamm.UnstructuredSubMesh(
            *_hex_grid([0.0, 0.5, 1.0], [0.0, 1.0], [0.0, 1.0, 2.0])
        )
        submesh.detect_box_boundaries()
        return submesh

    def test_serialise_submesh_item_encodes_mesh_data(self):
        submesh = self._submesh()
        data = Serialise.serialise_submesh_item(submesh)
        assert data["$type"] == "pybamm.meshes.unstructured_submesh.UnstructuredSubMesh"
        assert data["elements"] == submesh.elements.tolist()

    def test_load_submesh_types_rebuilds_prebuilt_submesh(self):
        import json

        import numpy as np

        submesh = self._submesh()
        submesh_types = {
            "negative electrode": submesh,
            "negative particle": pybamm.Uniform1DSubMesh,
        }
        data = json.loads(json.dumps(Serialise.serialise_submesh_types(submesh_types)))
        loaded = Serialise.load_submesh_types(data)

        restored = loaded["negative electrode"]
        assert isinstance(restored, pybamm.UnstructuredSubMesh)
        np.testing.assert_array_equal(restored.vertices, submesh.vertices)
        for name, faces in submesh.boundary_faces.items():
            np.testing.assert_array_equal(restored.boundary_faces[name], faces)
        assert loaded["negative particle"].submesh_type is pybamm.Uniform1DSubMesh
