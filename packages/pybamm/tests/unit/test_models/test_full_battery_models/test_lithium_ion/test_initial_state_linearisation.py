#
# Tests for initial states that carry their derivatives with respect to inputs
#
import numpy as np
import pytest

import pybamm
from pybamm.expression_tree.operations.input_jacobian import compile_input_jacobian
from pybamm.models.full_battery_models.lithium_ion.initial_state_linearisation import (
    _PLACEHOLDER_PREFIX,
    _linearise_about_solution,
)

EPS_P = "Positive electrode active material volume fraction"
C_N_MAX = "Maximum concentration in negative electrode [mol.m-3]"
OCP_SHIFT = "Positive electrode OCP shift [V]"


def _standard():
    return pybamm.lithium_ion.SPM(), pybamm.ParameterValues("Chen2020")


def _hysteresis():
    model = pybamm.lithium_ion.SPM(
        {"open-circuit potential": ("current sigmoid", "single")}
    )
    parameter_values = pybamm.ParameterValues("Chen2020")
    ocp = parameter_values["Negative electrode OCP [V]"]
    parameter_values.update(
        {
            "Negative electrode lithiation OCP [V]": lambda sto: ocp(sto) - 0.01,
            "Negative electrode delithiation OCP [V]": lambda sto: ocp(sto) + 0.01,
        },
        check_already_exists=False,
    )
    return model, parameter_values


def _hysteresis_state():
    # Without a direction, a voltage start is solved on the OCP at this state
    model, parameter_values = _hysteresis()
    parameter_values.update(
        {"Initial hysteresis state in negative electrode": 0.5},
        check_already_exists=False,
    )
    return model, parameter_values


def _msmr():
    model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
    return model, model.default_parameter_values


def _half_cell():
    # The shift enters the equations for both an SOC and a voltage start
    model = pybamm.lithium_ion.SPM({"working electrode": "positive"})
    parameter_values = pybamm.ParameterValues("Xu2019")
    ocp = parameter_values["Positive electrode OCP [V]"]
    parameter_values.update(
        {
            "Positive electrode OCP [V]": lambda sto: (
                ocp(sto) + pybamm.Parameter(OCP_SHIFT)
            ),
            OCP_SHIFT: 0.0,
        },
        check_already_exists=False,
    )
    return model, parameter_values


def _composite():
    model = pybamm.lithium_ion.SPM(
        {"particle phases": ("2", "1"), "surface form": "algebraic"}
    )
    return model, pybamm.ParameterValues("Chen2020_composite")


def _concentrations(*names):
    return [
        f"{name}Initial concentration in {electrode} electrode [mol.m-3]"
        for name, electrode in names
    ]


# (model factory, inputs, initial-state parameters, set_initial_state kwargs)
VARIANTS = {
    "standard": (
        _standard,
        {EPS_P: 0.665, C_N_MAX: 33133},
        _concentrations(("", "negative"), ("", "positive")),
        {},
    ),
    "cell capacity": (
        _standard,
        {EPS_P: 0.665, "Nominal cell capacity [A.h]": 5.0},
        _concentrations(("", "negative"), ("", "positive")),
        {"known_value": "cell capacity"},
    ),
    "hysteresis": (
        _hysteresis,
        {EPS_P: 0.665},
        _concentrations(("", "negative"), ("", "positive")),
        {"direction": "discharge"},
    ),
    "hysteresis no direction": (
        _hysteresis,
        {EPS_P: 0.665},
        _concentrations(("", "negative"), ("", "positive")),
        {},
    ),
    "hysteresis state": (
        _hysteresis_state,
        {EPS_P: 0.665, "Initial hysteresis state in negative electrode": 0.5},
        _concentrations(("", "negative"), ("", "positive")),
        {},
    ),
    "msmr": (
        _msmr,
        {EPS_P: 0.665},
        [
            "Initial voltage in negative electrode [V]",
            "Initial voltage in positive electrode [V]",
        ],
        {},
    ),
    "half-cell": (
        _half_cell,
        {
            "Maximum concentration in positive electrode [mol.m-3]": 48390,
            "Open-circuit voltage at 100% SOC [V]": 4.2,
            OCP_SHIFT: 0.01,
        },
        _concentrations(("", "positive")),
        {},
    ),
    "composite": (
        _composite,
        {
            EPS_P: 0.665,
            "Secondary: Negative electrode active material volume fraction": 0.015,
        },
        _concentrations(
            ("Primary: ", "negative"), ("Secondary: ", "negative"), ("", "positive")
        ),
        {},
    ),
}


def _derivative(value, name, inputs):
    """Derivative of a parameter value with respect to an input, as the default
    backend computes it."""
    names = list(inputs)
    function = compile_input_jacobian([pybamm.convert_to_symbol(value)], names)
    _, jacobian = function(np.array([inputs[n] for n in names], dtype=float))
    return jacobian[0, names.index(name)]


def _set_initial_state(variant, initial_value, inputs, differentiable):
    make_model, _, _, kwargs = VARIANTS[variant]
    model, parameter_values = make_model()
    parameter_values.update({name: "[input]" for name in inputs})
    return pybamm.lithium_ion.set_initial_state(
        initial_value,
        parameter_values,
        param=model.param,
        options=model.options,
        inplace=False,
        inputs=inputs,
        tol=1e-12,
        differentiable=differentiable,
        **kwargs,
    )


class TestLinearisedInitialState:
    @pytest.mark.parametrize("initial_value", [0.4, "3.9 V"])
    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_value_and_derivative_match_electrode_soh(self, variant, initial_value):
        _, inputs, keys, _ = VARIANTS[variant]
        linearised = _set_initial_state(variant, initial_value, inputs, True)
        numeric = _set_initial_state(variant, initial_value, inputs, False)

        for key in keys:
            value = linearised[key]
            np.testing.assert_allclose(
                np.squeeze(pybamm.convert_to_symbol(value).evaluate(inputs=inputs)),
                numeric[key],
                rtol=1e-12,
            )
            for name in inputs:
                h = 1e-5 * inputs[name]
                plus = _set_initial_state(
                    variant, initial_value, {**inputs, name: inputs[name] + h}, False
                )[key]
                minus = _set_initial_state(
                    variant, initial_value, {**inputs, name: inputs[name] - h}, False
                )[key]
                finite_difference = (plus - minus) / (2 * h)
                np.testing.assert_allclose(
                    _derivative(value, name, inputs),
                    finite_difference,
                    rtol=1e-5,
                    atol=1e-9 * max(abs(numeric[key]), 1) / inputs[name],
                )

    @pytest.mark.parametrize("value", [np.array([0.665]), np.array([[0.665]])])
    def test_size_one_array_inputs(self, value):
        keys = _concentrations(("", "negative"), ("", "positive"))
        inputs = {EPS_P: 0.665, C_N_MAX: 33133.0}
        expected = _set_initial_state("standard", 0.4, inputs, True)
        result = _set_initial_state("standard", 0.4, {**inputs, EPS_P: value}, True)
        for key in keys:
            for name in inputs:
                np.testing.assert_allclose(
                    _derivative(result[key], name, inputs),
                    _derivative(expected[key], name, inputs),
                    rtol=1e-12,
                )

    @pytest.mark.parametrize("variant", ["standard", "cell capacity"])
    def test_derivative_only_for_requested_inputs(self, variant):
        _, inputs, keys, _ = VARIANTS[variant]
        full = _set_initial_state(variant, 0.4, inputs, True)
        for requested in inputs:
            # An unrequested input of size one but two dimensions is still accepted
            values = {
                name: value if name == requested else np.array([[value]])
                for name, value in inputs.items()
            }
            linearised = _set_initial_state(variant, 0.4, values, [requested])
            for key in keys:
                np.testing.assert_allclose(
                    pybamm.convert_to_symbol(linearised[key]).evaluate(inputs=inputs),
                    pybamm.convert_to_symbol(full[key]).evaluate(inputs=inputs),
                    rtol=1e-12,
                )
                for name in inputs:
                    expected = _derivative(full[key], name, inputs)
                    np.testing.assert_allclose(
                        _derivative(linearised[key], name, inputs),
                        expected if name == requested else 0,
                        rtol=1e-12,
                    )

    def test_numeric_without_inputs(self):
        model, parameter_values = _standard()
        result = pybamm.lithium_ion.set_initial_state(
            0.4,
            parameter_values,
            param=model.param,
            options=model.options,
            inplace=False,
            differentiable=True,
        )
        numeric = parameter_values.set_initial_state(0.4, inplace=False)
        key = "Initial concentration in negative electrode [mol.m-3]"
        assert isinstance(result[key], float)
        np.testing.assert_allclose(result[key], numeric[key], rtol=1e-12)

    def test_resets_hysteresis_state_from_a_direction(self):
        # An earlier call with a direction writes the initial hysteresis states
        model, parameter_values = _hysteresis()
        parameter_values.update({EPS_P: "[input]"})
        inputs = {EPS_P: 0.665}
        kwargs = {
            "param": model.param,
            "options": model.options,
            "inputs": inputs,
            "tol": 1e-12,
        }
        pybamm.lithium_ion.set_initial_state(
            0.5, parameter_values, direction="charge", inplace=True, **kwargs
        )
        linearised = pybamm.lithium_ion.set_initial_state(
            "3.9 V", parameter_values, inplace=False, differentiable=True, **kwargs
        )
        numeric = pybamm.lithium_ion.set_initial_state(
            "3.9 V", parameter_values, inplace=False, **kwargs
        )
        for key in _concentrations(("", "negative"), ("", "positive")):
            value = pybamm.convert_to_symbol(linearised[key]).evaluate(inputs=inputs)
            np.testing.assert_allclose(np.squeeze(value), numeric[key], rtol=1e-12)

    @pytest.mark.parametrize("initial_value", [0.4, "3.9 V"])
    def test_input_names(self, initial_value):
        model, parameter_values = _standard()
        ocp = parameter_values["Negative electrode OCP [V]"]
        parameter_values.update(
            {
                "Negative electrode OCP [V]": lambda sto: (
                    ocp(sto) + pybamm.InputParameter("OCP shift [V]")
                ),
                EPS_P: "[input]",
                "Current function [A]": "[input]",
                "Maximum concentration in positive electrode [mol.m-3]": "[input]",
            }
        )
        names = pybamm.lithium_ion.get_initial_state_input_names(
            initial_value, parameter_values, model.param, model.options
        )
        assert names == {
            "OCP shift [V]",
            EPS_P,
            "Maximum concentration in positive electrode [mol.m-3]",
        }

    @pytest.mark.parametrize(
        ("perturbation", "inputs", "match"),
        [
            # e.g. a solver that stopped at a bound short of the root
            (1e-2, 4.2, r"does not satisfy its equations"),
            (0, np.array([4.2, 4.2]), r"scalar input parameters"),
        ],
    )
    def test_undifferentiable_solution_raises(self, perturbation, inputs, match):
        name = "Open-circuit voltage at 100% SOC [V]"
        model, parameter_values = _standard()
        parameter_values.update({name: "[input]"})
        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            parameter_values, param=model.param
        )
        solution = esoh_solver._solve_initial_stoichiometries(0.4, inputs={name: 4.2})
        solution["x_100"] += perturbation
        with pytest.raises(pybamm.ModelError, match=match):
            pybamm.lithium_ion.set_initial_state(
                0.4,
                parameter_values,
                param=model.param,
                options=model.options,
                inplace=False,
                inputs={name: inputs},
                esoh_solver=_FixedSolution(solution),
                differentiable=True,
            )

    def test_singular_jacobian(self):
        # z_0 + z_1 = p leaves z_0 - z_1 undetermined, and no change in the unknowns
        # can balance a change in q
        z_0, z_1 = _unknowns(2)
        p, q = pybamm.InputParameter("p"), pybamm.InputParameter("q")
        inputs = {"p": 1.0, "q": 1.0}
        args = (
            ["z_0", "z_1"],
            [z_0 + z_1 - p, q - 1],
            {"sum": z_0 + z_1, "first": z_0},
            {"z_0": 0.2, "z_1": 0.8},
            inputs,
            1e-12,
        )
        linearised = _linearise_about_solution(*args, ["p"])
        np.testing.assert_allclose(
            _derivative(linearised["sum"], "p", inputs), 1, rtol=1e-12
        )
        # Nothing moves along the undetermined direction
        np.testing.assert_allclose(
            _derivative(linearised["first"], "p", inputs), 0.5, rtol=1e-12
        )
        with pytest.raises(pybamm.ModelError, match=r"\['q'\] is unbounded"):
            _linearise_about_solution(*args, ["q"])

    @pytest.mark.parametrize(
        ("residual", "z_star", "match"),
        [
            (pybamm.sqrt, 0.0, r"Jacobian .* is not finite"),
            (pybamm.log, -1.0, r"maximum residual nan"),
        ],
    )
    def test_non_finite_system_raises(self, residual, z_star, match):
        (z,) = _unknowns(1)
        p = pybamm.InputParameter("p")
        with pytest.raises(pybamm.ModelError, match=match):
            _linearise_about_solution(
                ["z"], [residual(z) - p], {"z": z}, {"z": z_star}, {"p": 0.0}, 1e-12
            )


def _unknowns(n):
    """Placeholders for the unknowns of a system passed to the linearisation."""
    return [pybamm.InputParameter(f"{_PLACEHOLDER_PREFIX}{i}") for i in range(n)]


class _FixedSolution:
    """Stands in for an electrode SOH solver, returning a fixed solution."""

    def __init__(self, solution):
        self.solution = solution

    def _solve_initial_stoichiometries(self, *args, **kwargs):
        return self.solution
