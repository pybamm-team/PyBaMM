"""Helpers for the parameters that experiment steps drive (their controls)."""

from __future__ import annotations

import numbers

import pybamm

#: The parameter that battery steps drive.
CURRENT_CONTROL = "Current function [A]"

#: The parameter that a step's ``temperature`` sets, which steps cannot drive.
AMBIENT_TEMPERATURE_CONTROL = "Ambient temperature [K]"


def control_parameters(model):
    """The model's parameters that a step can drive, keyed by name."""
    return {
        parameter.name: parameter
        for parameter in model.parameters
        if isinstance(parameter, pybamm.Parameter | pybamm.FunctionParameter)
    }


def upsampling_variable_name(step, model, variables):
    """
    Name of the variable to upsample during ``step``: the step's own, else the
    model's ``upsampling_variable``, else the voltage, or ``None`` if ``variables``
    has none of them.
    """
    names = (
        step.upsampling_variable,
        getattr(model, "upsampling_variable", None),
        "Voltage [V]",
    )
    return next(
        (name for name in names if name is not None and name in variables), None
    )


def control_variable(variables, control):
    """The variable that a control becomes when it is adjusted."""
    if control == CURRENT_CONTROL:
        return variables["Current [A]"]
    return variables[control]


def default_control_value(model, parameter_values, control):
    """The value of a control in ``parameter_values``, or ``None`` if it has none."""
    try:
        return parameter_values.process_symbol(control_parameters(model)[control])
    except KeyError:
        return None


def adjusted_control_variable(control, default):
    """Variable of an adjusted control, scaled by its default value if constant."""
    scale = 1
    if default is not None and default.is_constant():
        scale = abs(float(default.evaluate())) or 1
    return pybamm.Variable(control, scale=scale)


def initial_control_guess(default, value=None):
    """
    Initial guess for an adjusted control: ``value`` if it is a number, else its
    default value if it is constant.
    """
    if isinstance(value, numbers.Number):
        return pybamm.Scalar(value)
    if default is not None and default.is_constant():
        return default
    return pybamm.Scalar(0)
