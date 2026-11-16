#
# Test dispatching mechanism in entry points
#
import importlib.metadata

import pytest

import pybamm


class TestDispatch:
    def test_get_entries_sorted(self, monkeypatch):
        """Test that entry points are sorted by name, whatever order they are found in"""
        names = ["Zeta2030", "Alpha2020", "Mu2025"]
        entries = [
            importlib.metadata.EntryPoint(name, f"package:{name}", "test_group")
            for name in names
        ]
        monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: entries)
        found = pybamm.dispatch.entry_points.EntryPoint.get_entries("test_group")
        assert [entry.name for entry in found] == sorted(names)

    @pytest.mark.parametrize(
        "entry_points", [pybamm.parameter_sets, pybamm.dispatch.models]
    )
    def test_listed_in_sorted_order(self, entry_points):
        """Test that parameter sets and models are listed in sorted order"""
        assert list(entry_points) == sorted(entry_points)

    def test_model_loads_through_entry_points(self):
        """Test that a model loaded through Model() function is actually functional"""
        # Load model with build=False to avoid full initialization for faster testing
        model = pybamm.Model("SPM", build=False)

        assert hasattr(model, "name")
        assert hasattr(model, "variables")
        assert hasattr(model, "submodels")
        assert hasattr(model, "algebraic")
        assert hasattr(model, "rhs")

        assert "SPM" in str(model)

    def test_getitem_returns_instance(self):
        """Test that __getitem__ returns an instantiated model"""
        model_instance = pybamm.dispatch.models["SPM"]
        assert hasattr(model_instance, "__class__")
        assert model_instance.__class__.__name__ == "SPM"

    def test_model_function(self):
        """Test Model function with arguments and options"""
        model = pybamm.Model("SPM")
        assert model.__class__.__name__ == "SPM"

        # Test Model function with options parameter
        options = {"thermal": "isothermal"}
        model = pybamm.Model("SPM", options)
        assert model.__class__.__name__ == "SPM"
