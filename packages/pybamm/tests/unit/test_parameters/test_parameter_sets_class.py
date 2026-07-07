#
# Tests for the ParameterSets class
#
import re

import pytest

import pybamm


class TestParameterSets:
    def test_name_interface(self):
        """Test that pybamm.parameters_sets.<ParameterSetName> returns
        the name of the parameter set and a depreciation warning
        """
        # Expect an error for parameter sets that aren't real
        with pytest.raises(AttributeError):
            pybamm.parameter_sets.not_a_real_parameter_set

    def test_all_registered(self):
        """Check that all parameter sets have been registered with the
        ``pybamm_parameter_sets`` entry point"""
        known_entry_points = set(
            ep.name for ep in pybamm.parameter_sets.get_entries("pybamm_parameter_sets")
        )
        assert set(pybamm.parameter_sets.keys()) == known_entry_points
        assert len(known_entry_points) == len(pybamm.parameter_sets)

    def test_get_docstring(self):
        """Test that :meth:`pybamm.parameter_sets.get_doctstring` works"""
        docstring = pybamm.parameter_sets.get_docstring("Marquis2019")
        assert re.search("Parameters for a Kokam SLPB78205130H cell", docstring)

    def test_iter(self):
        """Test that iterating `pybamm.parameter_sets` iterates over keys"""
        for k in pybamm.parameter_sets:
            assert isinstance(k, str)


class TestModelEntryPoints:
    def test_all_registered(self):
        """Check that all models have been registered with the
        ``pybamm_models`` entry point"""
        known_entry_points = set(
            ep.name for ep in pybamm.dispatch.models.get_entries("pybamm_models")
        )
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
