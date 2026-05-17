"""Integration tests for voltage as a state behavior.

Verifies that voltage observation works correctly with both default (true)
and legacy (false) settings, and that results are numerically consistent.
"""

import numpy as np
import pytest

import pybamm


class TestVoltageAsStateDefault:
    """Test that voltage as a state works correctly with the new default."""

    @pytest.mark.parametrize(
        "model_cls",
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    )
    def test_default_has_voltage_as_algebraic_state(self, model_cls):
        """Default models should have Voltage [V] as an algebraic state."""
        model = model_cls()
        assert model.options["voltage as a state"] == "true"
        assert "Voltage [V]" in [var.name for var in model.algebraic.keys()]

    @pytest.mark.parametrize(
        "model_cls",
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    )
    def test_legacy_mode_has_voltage_as_expression(self, model_cls):
        """Legacy mode should compute voltage from expression."""
        model = model_cls(options={"voltage as a state": "false"})
        assert model.options["voltage as a state"] == "false"
        algebraic_var_names = [var.name for var in model.algebraic.keys()]
        assert "Voltage [V]" not in algebraic_var_names


class TestVoltageAsStateNumericalConsistency:
    """Test that voltage values are consistent between default and legacy modes."""

    @pytest.mark.parametrize(
        "model_cls",
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    )
    def test_voltage_values_match_legacy(self, model_cls):
        """Voltage values should match between default and legacy modes."""
        t_eval = np.linspace(0, 3600, 100)

        # Default mode (voltage as state)
        model_default = model_cls()
        sim_default = pybamm.Simulation(model_default)
        sol_default = sim_default.solve(t_eval)
        # Evaluate at the explicit t_eval points to avoid shape mismatches
        # caused by the adaptive solver producing different internal step counts
        v_default = sol_default["Voltage [V]"](t_eval)

        # Legacy mode (voltage as expression)
        model_legacy = model_cls(options={"voltage as a state": "false"})
        sim_legacy = pybamm.Simulation(model_legacy)
        sol_legacy = sim_legacy.solve(t_eval)
        v_legacy = sol_legacy["Voltage [V]"](t_eval)

        # Should match within solver tolerance
        np.testing.assert_allclose(v_default, v_legacy, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize(
        "model_cls",
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    )
    def test_voltage_expression_still_available(self, model_cls):
        """Voltage expression should still be observable."""
        t_eval = np.linspace(0, 3600, 100)

        model = model_cls()
        sim = pybamm.Simulation(model)
        sol = sim.solve(t_eval)

        # Evaluate at the explicit t_eval points for a consistent shape
        v = sol["Voltage [V]"](t_eval)
        v_expr = sol["Voltage expression [V]"](t_eval)

        # Voltage and voltage expression should match (algebraic constraint)
        np.testing.assert_allclose(v, v_expr, rtol=1e-5, atol=1e-8)
