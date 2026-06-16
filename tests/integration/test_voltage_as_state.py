"""Integration tests for the 'voltage as a state' option.

Voltage is promoted to an algebraic state by default; these tests cover the
default behavior, the 'false' escape hatch that restores pure-ODE SPM/SPMe,
the conditional surface-form defaults, and the explicit power/resistance
operating modes.
"""

import numpy as np
import pytest

import pybamm

LEGACY_OPTIONS = {"voltage as a state": "false", "surface form": "false"}
LI_ION_MODELS = [
    pybamm.lithium_ion.SPM,
    pybamm.lithium_ion.SPMe,
    pybamm.lithium_ion.DFN,
]
REDUCED_MODELS = [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe]


class TestVoltageAsStateDefault:
    """By default, voltage is an algebraic state variable."""

    @pytest.mark.parametrize("model_cls", LI_ION_MODELS)
    def test_voltage_is_algebraic_state(self, model_cls):
        model = model_cls()
        algebraic_var_names = [var.name for var in model.algebraic.keys()]
        assert "Voltage [V]" in algebraic_var_names
        assert isinstance(model.variables["Voltage [V]"], pybamm.Variable)

    @pytest.mark.parametrize("model_cls", LI_ION_MODELS)
    def test_residuals_reference_variable(self, model_cls):
        """Algebraic residuals should contain the Variable, not the expression."""
        model = model_cls()
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
        model = model_cls()
        sol = pybamm.Simulation(model).solve([0, 3600])

        v = sol["Voltage [V]"].entries
        v_expr = sol["Voltage expression [V]"].entries
        np.testing.assert_allclose(v, v_expr, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("model_cls", LI_ION_MODELS)
    def test_false_excludes_algebraic_voltage(self, model_cls):
        """With the option disabled, voltage remains a computed expression."""
        model = model_cls(options=LEGACY_OPTIONS)
        algebraic_var_names = [var.name for var in model.algebraic.keys()]
        assert "Voltage [V]" not in algebraic_var_names
        assert not isinstance(model.variables["Voltage [V]"], pybamm.Variable)

    def test_dfn_false_still_has_algebraic_states(self):
        """DFN with the option disabled retains other algebraic states
        (electrode/electrolyte potentials), so it is not a pure ODE model."""
        model = pybamm.lithium_ion.DFN(options=LEGACY_OPTIONS)
        assert len(model.algebraic) > 0
        algebraic_var_names = [var.name for var in model.algebraic.keys()]
        assert "Voltage [V]" not in algebraic_var_names

    def test_dfn_false_rejects_scipy(self):
        model = pybamm.lithium_ion.DFN(options=LEGACY_OPTIONS)
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

    @pytest.mark.parametrize("model_cls", REDUCED_MODELS)
    def test_plain_models_keep_false_surface_form(self, model_cls):
        # Plain SPM/SPMe keep the explicit-current closure; voltage as a
        # state is the only algebraic equation
        model = model_cls()
        assert model.options["surface form"] == "false"
        assert len(model.algebraic) == 1

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


class TestLegacyOdeBehavior:
    """voltage-as-a-state='false' + surface form='false' remains the
    supported route to pure-ODE SPM/SPMe for ODE-only solvers."""

    @pytest.mark.parametrize("model_cls", REDUCED_MODELS)
    def test_legacy_solvable_by_scipy(self, model_cls):
        model = model_cls(options=LEGACY_OPTIONS)
        sim = pybamm.Simulation(model, solver=pybamm.ScipySolver())
        sol = sim.solve([0, 3600])
        v = sol["Voltage [V]"].entries
        assert v[0] > v[-1]  # voltage decreases during discharge

    @pytest.mark.parametrize("model_cls", REDUCED_MODELS)
    def test_legacy_solvable_by_casadi_safe(self, model_cls):
        model = model_cls(options=LEGACY_OPTIONS)
        sim = pybamm.Simulation(model, solver=pybamm.CasadiSolver(mode="safe"))
        sol = sim.solve([0, 3600])
        assert sol.termination == "final time"

    @pytest.mark.parametrize("model_cls", REDUCED_MODELS)
    def test_legacy_solvable_by_casadi_fast(self, model_cls):
        model = model_cls(options=LEGACY_OPTIONS)
        sim = pybamm.Simulation(model, solver=pybamm.CasadiSolver(mode="fast"))
        sol = sim.solve([0, 3600])
        assert sol.termination == "final time"


class TestDefaultDaeBehavior:
    """Default SPM/SPMe (voltage as an algebraic state) works with
    DAE-capable solvers and rejects ODE-only solvers."""

    @pytest.mark.parametrize("model_cls", LI_ION_MODELS)
    def test_default_solvable_by_idaklu(self, model_cls):
        model = model_cls()
        sim = pybamm.Simulation(model, solver=pybamm.IDAKLUSolver())
        sol = sim.solve([0, 3600])
        v = sol["Voltage [V]"].entries
        assert v[0] > v[-1]

    @pytest.mark.parametrize("model_cls", REDUCED_MODELS)
    def test_default_solvable_by_casadi_safe(self, model_cls):
        model = model_cls()
        sim = pybamm.Simulation(model, solver=pybamm.CasadiSolver(mode="safe"))
        sol = sim.solve([0, 3600])
        assert sol.termination == "final time"

    @pytest.mark.skipif(not pybamm.has_jax(), reason="jax or jaxlib is not installed")
    def test_default_solvable_by_jax_bdf(self):
        model = pybamm.lithium_ion.SPM()
        model.convert_to_format = "jax"
        model.events = []  # JaxSolver does not support terminate events
        sim = pybamm.Simulation(model, solver=pybamm.JaxSolver(method="BDF"))
        sol = sim.solve(np.linspace(0, 3600, 100))
        v = sol["Voltage [V]"].entries
        assert v[0] > v[-1]  # voltage decreases during discharge

    @pytest.mark.parametrize("model_cls", REDUCED_MODELS)
    def test_default_rejects_scipy(self, model_cls):
        model = model_cls()
        sim = pybamm.Simulation(model, solver=pybamm.ScipySolver())
        with pytest.raises(pybamm.SolverError, match="Cannot use ODE solver"):
            sim.solve([0, 3600])
