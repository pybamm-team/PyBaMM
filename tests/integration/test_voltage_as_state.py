"""Integration tests for the 'voltage as a state' option.

Voltage promotion is registered centrally in BaseBatteryModel._build_model;
these tests cover the opt-in behavior, the conditional defaults, and the
explicit power/resistance operating modes that require it.
"""

import numpy as np
import pytest

import pybamm

OPT_IN = {"voltage as a state": "true"}
LI_ION_MODELS = [
    pybamm.lithium_ion.SPM,
    pybamm.lithium_ion.SPMe,
    pybamm.lithium_ion.DFN,
]


class TestVoltageAsStateOptIn:
    """With the option enabled, voltage is an algebraic state variable."""

    @pytest.mark.parametrize("model_cls", LI_ION_MODELS)
    def test_voltage_is_algebraic_state(self, model_cls):
        model = model_cls(options=OPT_IN)
        algebraic_var_names = [var.name for var in model.algebraic.keys()]
        assert "Voltage [V]" in algebraic_var_names
        assert isinstance(model.variables["Voltage [V]"], pybamm.Variable)

    @pytest.mark.parametrize("model_cls", LI_ION_MODELS)
    def test_residuals_reference_variable(self, model_cls):
        """Algebraic residuals should contain the Variable, not the expression."""
        model = model_cls(options=OPT_IN)
        voltage_eqs = [
            expr for var, expr in model.algebraic.items() if var.name == "Voltage [V]"
        ]
        assert len(voltage_eqs) == 1
        syms = [
            s
            for s in voltage_eqs[0].pre_order()
            if isinstance(s, pybamm.Variable) and s.name == "Voltage [V]"
        ]
        assert len(syms) > 0

    @pytest.mark.parametrize("model_cls", LI_ION_MODELS)
    def test_voltage_expression_matches_state(self, model_cls):
        model = model_cls(options=OPT_IN)
        sol = pybamm.Simulation(model).solve([0, 3600])

        v = sol["Voltage [V]"].entries
        v_expr = sol["Voltage expression [V]"].entries
        np.testing.assert_allclose(v, v_expr, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("model_cls", LI_ION_MODELS)
    def test_default_voltage_is_expression(self, model_cls):
        """Without the option, voltage remains a computed expression."""
        model = model_cls()
        algebraic_var_names = [var.name for var in model.algebraic.keys()]
        assert "Voltage [V]" not in algebraic_var_names
        assert not isinstance(model.variables["Voltage [V]"], pybamm.Variable)

    @pytest.mark.parametrize(
        "model_cls", [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe]
    )
    def test_opt_in_solvable_by_casadi_safe(self, model_cls):
        model = model_cls(options=OPT_IN)
        sim = pybamm.Simulation(model, solver=pybamm.CasadiSolver(mode="safe"))
        sol = sim.solve([0, 3600])
        v = sol["Voltage [V]"].entries
        assert v[0] > v[-1]  # voltage decreases during discharge

    def test_opt_in_rejects_scipy(self):
        """Promoting voltage makes SPM a DAE, which ODE solvers reject."""
        model = pybamm.lithium_ion.SPM(options=OPT_IN)
        sim = pybamm.Simulation(model, solver=pybamm.ScipySolver())
        with pytest.raises(pybamm.SolverError, match="Cannot use ODE solver"):
            sim.solve([0, 3600])


class TestExplicitPowerResistance:
    """Explicit power/resistance control needs voltage as a state: I = P/V
    (or I = V/R) is circular when V is an expression depending on I."""

    @pytest.mark.parametrize(
        "operating_mode", ["explicit power", "explicit resistance"]
    )
    def test_defaults_to_voltage_as_state(self, operating_mode):
        model = pybamm.lithium_ion.SPM({"operating mode": operating_mode})
        assert model.options["voltage as a state"] == "true"

    def test_explicit_power_solves(self):
        model = pybamm.lithium_ion.SPM({"operating mode": "explicit power"})
        params = model.default_parameter_values
        params["Power function [W]"] = 2.0
        sol = pybamm.Simulation(model, parameter_values=params).solve([0, 1800])
        power = sol["Voltage [V]"].entries * sol["Current [A]"].entries
        np.testing.assert_allclose(power, 2.0, rtol=1e-4)

    @pytest.mark.parametrize(
        "operating_mode", ["explicit power", "explicit resistance"]
    )
    def test_explicit_false_rejected(self, operating_mode):
        with pytest.raises(
            pybamm.OptionError,
            match=r"Cannot use.*operating mode.*'voltage as a state'.*'false'",
        ):
            pybamm.lithium_ion.SPM(
                options={
                    "voltage as a state": "false",
                    "operating mode": operating_mode,
                }
            )


class TestSurfaceFormConditionalDefaults:
    """SPM/SPMe promote surface form to 'algebraic' only when the
    explicit-current closure is unavailable."""

    @pytest.mark.parametrize(
        "model_cls", [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe]
    )
    def test_plain_models_keep_false_surface_form(self, model_cls):
        model = model_cls()
        assert model.options["surface form"] == "false"
        assert len(model.algebraic) == 0

    @pytest.mark.parametrize(
        "options",
        [
            {"intercalation kinetics": "asymmetric Butler-Volmer"},
            {"particle size": "distribution"},
        ],
    )
    def test_conditional_algebraic_surface_form(self, options):
        # No inverse kinetics for non-default kinetics; distributions
        # require a surface formulation
        model = pybamm.lithium_ion.SPM(options=options)
        assert model.options["surface form"] == "algebraic"

    def test_dfn_defaults_to_false_surface_form(self):
        model = pybamm.lithium_ion.DFN()
        assert model.options["surface form"] == "false"


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
