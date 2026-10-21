#
# Tests for initial states that carry their derivatives with respect to inputs
#
import casadi
import numpy as np
import pytest

import pybamm

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
    model = pybamm.lithium_ion.SPM({"particle phases": ("2", "1")})
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
    """Derivative of a parameter value with respect to an input, as the solver
    computes it."""
    p = casadi.MX.sym("p")
    expression = pybamm.convert_to_symbol(value).to_casadi(inputs={**inputs, name: p})
    jacobian = casadi.Function("jacobian", [p], [casadi.jacobian(expression, p)])
    return float(jacobian(inputs[name]))


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

    def test_derivative_only_for_requested_inputs(self):
        # An unrequested input of size one but two dimensions is still accepted
        inputs = {EPS_P: 0.665, C_N_MAX: 33133.0}
        linearised = _set_initial_state(
            "standard", 0.4, {**inputs, EPS_P: np.array([[0.665]])}, [C_N_MAX]
        )
        full = _set_initial_state("standard", 0.4, inputs, True)
        key = "Initial concentration in positive electrode [mol.m-3]"
        np.testing.assert_allclose(
            pybamm.convert_to_symbol(linearised[key]).evaluate(inputs=inputs),
            pybamm.convert_to_symbol(full[key]).evaluate(inputs=inputs),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            _derivative(linearised[key], C_N_MAX, inputs),
            _derivative(full[key], C_N_MAX, inputs),
            rtol=1e-12,
        )
        # The dependence on the unrequested input through the solution is dropped
        restricted, total = (
            _derivative(value[key], EPS_P, inputs) for value in [linearised, full]
        )
        assert abs(restricted - total) > 1e-3 * abs(total)

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


class _FixedSolution:
    """Stands in for an electrode SOH solver, returning a fixed solution."""

    def __init__(self, solution):
        self.solution = solution

    def _solve_initial_stoichiometries(self, *args, **kwargs):
        return self.solution
