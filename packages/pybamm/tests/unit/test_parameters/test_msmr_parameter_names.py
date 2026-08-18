import pytest

import pybamm
from pybamm.models.full_battery_models.lithium_ion.msmr import (
    is_deprecated_msmr_name,
    replace_deprecated_msmr_name,
)


class TestMSMRParameterNames:
    def test_is_deprecated_msmr_name(self):
        """
        Test the is_deprecated_msmr_name function with various parameter names
        """
        # Valid names (returns True)
        valid_names = [
            "X_p_3",
            "X_n_l_300",
            "Q_p_42",
            "w_n_d_7",
            "U0_p_0",
        ]
        for name in valid_names:
            assert is_deprecated_msmr_name(name) is True

        # Invalid names (returns False)
        invalid_names = [
            "X_p_-3",  # Negative index
            "X_j_3",  # Invalid electrode
            "Y_p_3",  # Invalid base
            "X_p_k_3",  # Invalid qualifier
            "X_p_l",  # Missing electrode
            "X_3",  # Missing index
            "X_p_3_extra",  # Extra components
            "electrode",  # Invalid base
            "particle",  # Invalid base
            "",  # Empty string
        ]
        for name in invalid_names:
            assert is_deprecated_msmr_name(name) is False

            with pytest.raises(ValueError, match=r"Invalid MSMR name"):
                replace_deprecated_msmr_name(name)

    def test_replace_deprecated_msmr_name(self):
        """
        Test the replace_deprecated_msmr_name function with various parameter names
        """
        name_mapping = {
            "X_p_3": "Positive electrode host site occupancy fraction (3)",
            "X_n_5": "Negative electrode host site occupancy fraction (5)",
            "X_p_l_3": "Positive electrode host site occupancy fraction (lithiation) (3)",
            "X_n_d_5": "Negative electrode host site occupancy fraction (delithiation) (5)",
            "Q_p_3": "Positive electrode host site occupancy capacity (3) [A.h]",
            "w_n_2": "Negative electrode host site ideality factor (2)",
            "U0_p_d_4": "Positive electrode host site standard potential (delithiation) (4) [V]",
            "a_p_d_5": "Positive electrode host site charge transfer coefficient (delithiation) (5)",
            "j0_ref_n_0": "Negative electrode host site reference exchange-current density (0) [A.m-2]",
        }
        for old_name, new_name in name_mapping.items():
            assert replace_deprecated_msmr_name(old_name) == new_name

    def test_integration_with_parameter_values(self):
        """
        Test that the MSMR parameter name functions work correctly when integrated
        with the ParameterValues class
        """
        # Define mapping of old parameter names to values
        old_params = {
            "X_p_3": 0.5,
            "X_n_l_2": 0.3,
            "Q_p_1": 2.5,
            "U0_n_d_4": 0.1,
            "j0_ref_n_0": 0.1,
        }

        # Define mapping of old parameter names to their human-readable equivalents
        name_mapping = {
            "X_p_3": "Positive electrode host site occupancy fraction (3)",
            "X_n_l_2": "Negative electrode host site occupancy fraction (lithiation) (2)",
            "Q_p_1": "Positive electrode host site occupancy capacity (1) [A.h]",
            "U0_n_d_4": "Negative electrode host site standard potential (delithiation) (4) [V]",
            "j0_ref_n_0": "Negative electrode host site reference exchange-current density (0) [A.m-2]",
        }

        param_values = pybamm.ParameterValues(old_params)

        # New human-readable keys should be present
        for new_name in name_mapping.values():
            assert new_name in param_values

        # Original keys should remain
        for old_key, val in old_params.items():
            assert old_key in param_values
            assert param_values[old_key] == val

        # And their human-readable counterparts map to the same values
        for old_key, new_key in name_mapping.items():
            assert param_values[new_key] == old_params[old_key]
