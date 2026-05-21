"""Integration tests for voltage as a state behavior.

Verifies that voltage is always an algebraic state in standard (non-basic)
models and that the deprecated option emits a warning.
"""

import warnings

import numpy as np
import pytest

import pybamm


class TestVoltageAlwaysAState:
    """Voltage should always be an algebraic state variable."""

    @pytest.mark.parametrize(
        "model_cls",
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    )
    def test_voltage_is_algebraic_state(self, model_cls):
        model = model_cls()
        algebraic_var_names = [var.name for var in model.algebraic.keys()]
        assert "Voltage [V]" in algebraic_var_names
        assert isinstance(model.variables["Voltage [V]"], pybamm.Variable)

    @pytest.mark.parametrize(
        "model_cls",
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    )
    def test_residuals_reference_variable(self, model_cls):
        """Algebraic residuals should contain the Variable, not the expression."""
        model = model_cls()
        for var, expr in model.algebraic.items():
            if var.name == "Voltage [V]":
                syms = [
                    s
                    for s in expr.pre_order()
                    if isinstance(s, pybamm.Variable) and s.name == "Voltage [V]"
                ]
                assert len(syms) > 0

    @pytest.mark.parametrize(
        "model_cls",
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    )
    def test_voltage_expression_matches_state(self, model_cls):
        model = model_cls()
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 3600])

        v = sol["Voltage [V]"].entries
        v_expr = sol["Voltage expression [V]"].entries
        np.testing.assert_allclose(v, v_expr, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize(
        "model_cls",
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    )
    def test_option_deprecated(self, model_cls):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model_cls(options={"voltage as a state": "false"})
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "deprecated" in str(dep_warnings[0].message)

    @pytest.mark.parametrize(
        "model_cls",
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    )
    def test_deprecated_option_still_makes_voltage_a_state(self, model_cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = model_cls(options={"voltage as a state": "false"})
        algebraic_var_names = [var.name for var in model.algebraic.keys()]
        assert "Voltage [V]" in algebraic_var_names


class TestBasicModelsVoltageExpression:
    """Basic models expose voltage as an expression, not a state."""

    @pytest.mark.parametrize(
        "model_cls",
        [pybamm.lithium_ion.BasicSPM, pybamm.lithium_ion.BasicDFN],
    )
    def test_basic_models_have_voltage_expression(self, model_cls):
        model = model_cls()
        assert "Voltage expression [V]" in model.variables
        assert "Voltage [V]" in model.variables
        assert not isinstance(model.variables["Voltage [V]"], pybamm.Variable)
