#
# Tests for the ParameterSets class
#
import pytest
import re
import pybamm


class TestParameterSets:
    def test_name_interface(self):
        """Test that pybamm.parameters_sets.<ParameterSetName> returns
        the name of the parameter set and a depreciation warning
        """
        with pytest.warns(DeprecationWarning):
            out = pybamm.parameter_sets.Marquis2019
            assert out == "Marquis2019"

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
