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
        model_class = pybamm.dispatch.models.get_class("SPM")
        # Should be a class, not an instance
        assert callable(model_class)
        assert hasattr(model_class, "__name__")
        assert model_class.__name__ == "SPM"

    def test_getitem_returns_instance(self):
        """Test that __getitem__ returns an instantiated model"""
        model_instance = pybamm.dispatch.models["SPM"]
        # Should be an instance, not a class
        assert hasattr(model_instance, "__class__")
        assert model_instance.__class__.__name__ == "SPM"

    def test_model_function_no_args(self):
        """Test Model function with no additional arguments"""
        model = pybamm.Model("SPM")
        assert model.__class__.__name__ == "SPM"

    def test_model_function_with_options(self):
        """Test Model function with options parameter"""
        options = {"thermal": "isothermal"}
        model = pybamm.Model("SPM", options=options)
        assert model.__class__.__name__ == "SPM"

    def test_model_function_with_args_kwargs(self):
        """Test Model function with *args and **kwargs"""
        # Test with kwargs only
        model = pybamm.Model("SPM", build=False)
        assert model.__class__.__name__ == "SPM"

        # Test with options and kwargs
        options = {"thermal": "isothermal"}
        model = pybamm.Model("SPM", options=options, build=False)
        assert model.__class__.__name__ == "SPM"

    def test_invalid_model_key_error(self):
        """Test that invalid model names raise KeyError"""
        with pytest.raises(KeyError, match="Unknown parameter set or model"):
            pybamm.dispatch.models.get_class("NonExistentModel")

        with pytest.raises(KeyError, match="Unknown parameter set or model"):
            pybamm.dispatch.models["NonExistentModel"]

        with pytest.raises(KeyError, match="Unknown parameter set or model"):
            pybamm.Model("NonExistentModel")

    def test_models_no_deprecation_warning(self):
        """Test that accessing models via attribute doesn't trigger deprecation warning"""
        model_names = list(pybamm.dispatch.models.keys())
        if model_names:  # Only run if models exist
            model_name = model_names[0]

            import warnings

            with warnings.catch_warnings(record=True) as warning_list:
                warnings.simplefilter("always")
                try:
                    getattr(pybamm.dispatch.models, model_name)
                except AttributeError:
                    # This is expected - models don't have the deprecation behavior
                    pass

            # Filter out any unrelated warnings
            deprecation_warnings = [
                w for w in warning_list if issubclass(w.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 0

    def test_getattribute_normal_attributes(self):
        """Test that normal attributes work correctly in __getattribute__"""
        assert hasattr(pybamm.dispatch.models, "group")
        assert pybamm.dispatch.models.group == "pybamm_models"

        assert hasattr(pybamm.dispatch.parameter_sets, "group")
        assert pybamm.dispatch.parameter_sets.group == "pybamm_parameter_sets"
