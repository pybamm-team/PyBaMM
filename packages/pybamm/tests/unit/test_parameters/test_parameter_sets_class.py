#
# Tests for the ParameterSets class
#
import importlib.metadata
import re

import pytest

import pybamm
from pybamm.dispatch import entry_points


class TestParameterSets:
    def test_name_interface(self):
        """Test that pybamm.parameters_sets.<ParameterSetName> returns
        the name of the parameter set and a depreciation warning
        """
        # Expect an error for parameter sets that aren't real
        with pytest.raises(AttributeError):
            pybamm.parameter_sets.not_a_real_parameter_set

    def test_all_registered(self):
        """Check that the registry contains the available entry points."""
        known_entry_points = {
            ep.name for ep in pybamm.parameter_sets.get_entries("pybamm_parameter_sets")
        }
        available_names = set(pybamm.parameter_sets)
        assert available_names <= known_entry_points
        assert known_entry_points - available_names <= {"MolicelP45B_AboutEnergy2026"}
        assert len(available_names) == len(pybamm.parameter_sets)

    @pytest.mark.parametrize(
        ("bpx_version", "available"),
        [(None, False), ("0.5.0", False), ("1.1.1", True), ("1.2.0", False)],
    )
    def test_bpx_parameter_set_availability(self, mocker, bpx_version, available):
        group = "pybamm_parameter_sets"
        name = "MolicelP45B_AboutEnergy2026"
        bundled = next(
            entry
            for entry in pybamm.parameter_sets.get_entries(group)
            if entry.name == name
        )
        other = importlib.metadata.EntryPoint(
            "Other", "other_package.module:get_parameter_values", group
        )
        mocker.patch.object(
            importlib.metadata, "entry_points", return_value=[bundled, other]
        )
        bpx_spec = mocker.sentinel.bpx_spec if bpx_version is not None else None
        mocker.patch.object(entry_points, "find_spec", return_value=bpx_spec)
        mocker.patch.object(importlib.metadata, "version", return_value=bpx_version)
        mocker.patch.object(importlib.metadata.EntryPoint, "load", return_value=dict)
        mocker.patch.object(entry_points.EntryPoint, "_instances", 0)
        mocker.patch.object(entry_points.EntryPoint, "instance")

        registry = entry_points.EntryPoint(group)

        assert {entry.name for entry in registry.get_entries(group)} == {name, "Other"}
        expected = {name, "Other"} if available else {"Other"}
        assert (name in registry) is available
        assert (
            set(registry)
            == dict(registry).keys()
            == dict(registry.items()).keys()
            == expected
        )

    def test_get_docstring(self):
        """Test that :meth:`pybamm.parameter_sets.get_doctstring` works"""
        docstring = pybamm.parameter_sets.get_docstring("Marquis2019")
        assert re.search("Parameters for a Kokam SLPB78205130H cell", docstring)

    def test_iter(self):
        """Test that iterating `pybamm.parameter_sets` iterates over keys"""
        for k in pybamm.parameter_sets:
            assert isinstance(k, str)

    def test_contains_does_not_load(self, mocker):
        """Test that membership checks do not build the parameter set"""

        def get_parameter_values():
            raise ModuleNotFoundError("missing optional dependency")

        mocker.patch.dict(
            pybamm.parameter_sets._all_entries, {"Unloadable": get_parameter_values}
        )
        assert "Unloadable" in pybamm.parameter_sets
        assert "not_a_real_parameter_set" not in pybamm.parameter_sets

    def test_items_skips_missing_dependency(self, mocker):
        """Test that items and values skip sets whose dependencies are missing"""

        def get_parameter_values():
            raise ModuleNotFoundError("missing optional dependency")

        mocker.patch.dict(
            pybamm.parameter_sets._all_entries,
            {"Unloadable": get_parameter_values, "Loadable": lambda: {"a": 1}},
            clear=True,
        )
        assert list(pybamm.parameter_sets.items()) == [("Loadable", {"a": 1})]
        assert list(pybamm.parameter_sets.values()) == [{"a": 1}]
        with pytest.raises(ModuleNotFoundError, match="missing optional"):
            pybamm.parameter_sets["Unloadable"]


class TestModelEntryPoints:
    def test_all_registered(self):
        """Check that all models have been registered with the
        ``pybamm_models`` entry point"""
        known_entry_points = {
            ep.name for ep in pybamm.dispatch.models.get_entries("pybamm_models")
        }
        assert set(pybamm.dispatch.models.keys()) == known_entry_points
        assert len(known_entry_points) == len(pybamm.dispatch.models)

    def test_get_docstring(self):
        """Test that :meth:`pybamm.dispatch.models.get_doctstring` works"""
        docstring = pybamm.dispatch.models.get_docstring("SPM")
        print(docstring)
        assert re.search(
            "Single Particle Model",
            docstring,
        )

    def test_iter(self):
        """Test that iterating `pybamm.models` iterates over keys"""
        for k in pybamm.dispatch.models:
            assert isinstance(k, str)

    def test_get_class_method(self):
        """Test that get_class method returns the class without instantiating"""
        model_class = pybamm.dispatch.models._get_class("SPM")
        # Should be a class, not an instance
        assert callable(model_class)
        assert hasattr(model_class, "__name__")
        assert model_class.__name__ == "SPM"

    def test_getattribute_normal_attributes(self):
        """Test that normal attributes work correctly in __getattribute__"""
        assert hasattr(pybamm.dispatch.models, "group")
        assert pybamm.dispatch.models.group == "pybamm_models"

        assert hasattr(pybamm.dispatch.parameter_sets, "group")
        assert pybamm.dispatch.parameter_sets.group == "pybamm_parameter_sets"
