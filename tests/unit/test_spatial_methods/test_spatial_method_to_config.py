#
# Tests for SpatialMethod to_config / from_config
#


import pybamm
from pybamm.expression_tree.operations.serialise import Serialise


class TestSpatialMethodToConfig:
    """Tests for SpatialMethod to_config and from_config."""

    def test_spatial_method_to_config_returns_class_module_options(self):
        """SpatialMethod.to_config() returns dict with 'class', 'module', 'options'."""
        method = pybamm.FiniteVolume()
        data = method.to_config()
        assert isinstance(data, dict)
        assert data["class"] == "FiniteVolume"
        assert "pybamm.spatial_methods" in data["module"]
        assert "options" in data
        assert isinstance(data["options"], dict)

    def test_spatial_method_from_config_same_type_and_options(self):
        """SpatialMethod.from_config(to_config()) returns same type with same options."""
        method = pybamm.FiniteVolume()
        restored = pybamm.SpatialMethod.from_config(method.to_config())
        assert type(restored) is type(method)
        assert restored.options == method.options

    def test_spatial_method_from_config_with_custom_options(self):
        """SpatialMethod with custom options round-trips correctly."""
        custom_options = {
            "extrapolation": {
                "order": {"gradient": "linear", "value": "quadratic"},
                "use bcs": True,
            }
        }
        method = pybamm.FiniteVolume(options=custom_options)
        restored = pybamm.SpatialMethod.from_config(method.to_config())
        assert type(restored) is type(method)
        assert restored.options == custom_options

    def test_zero_dimensional_spatial_method_round_trip(self):
        """ZeroDimensionalSpatialMethod round-trips via to_config/from_config."""
        method = pybamm.ZeroDimensionalSpatialMethod()
        data = method.to_config()
        assert data["class"] == "ZeroDimensionalSpatialMethod"
        restored = pybamm.SpatialMethod.from_config(data)
        assert type(restored) is type(method)
        assert restored.options == method.options


class TestSpatialMethodsDictRoundTrip:
    """Dict round-trip tests for spatial_methods."""

    def test_dict_round_trip_spatial_methods(self):
        """Dict of SpatialMethod instances round-trips via to_config/from_config."""
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "current collector": pybamm.ZeroDimensionalSpatialMethod(),
        }
        serialised = {k: v.to_config() for k, v in spatial_methods.items()}
        restored = {
            k: pybamm.SpatialMethod.from_config(v) for k, v in serialised.items()
        }
        assert set(restored.keys()) == set(spatial_methods.keys())
        for domain in spatial_methods:
            assert type(restored[domain]) is type(spatial_methods[domain])
            assert restored[domain].options == spatial_methods[domain].options


class TestSpatialMethodSerialiseCompatibility:
    """Compatibility between SpatialMethod to_config and Serialise."""

    def test_spatial_method_to_config_matches_serialise_spatial_method_item(self):
        """SpatialMethod.to_config() matches Serialise.serialise_spatial_method_item."""
        method = pybamm.FiniteVolume()
        assert method.to_config() == Serialise.serialise_spatial_method_item(method)

    def test_full_dict_serialise_load_matches_spatial_method_from_config(self):
        """Full dict via serialise_spatial_methods/load_spatial_methods matches from_config per entry."""
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "current collector": pybamm.ZeroDimensionalSpatialMethod(),
        }
        full_json = Serialise.serialise_spatial_methods(spatial_methods)
        loaded = Serialise.load_spatial_methods(full_json)
        assert set(loaded.keys()) == set(spatial_methods.keys())
        for domain, method_info in full_json["spatial_methods"].items():
            from_item = pybamm.SpatialMethod.from_config(method_info)
            assert type(loaded[domain]) is type(from_item)
            assert loaded[domain].options == from_item.options
