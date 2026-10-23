#
# Initial states that carry the derivatives of the electrode SOH solution with
# respect to input parameters
#
from __future__ import annotations

from collections.abc import Collection
from functools import lru_cache

import casadi
import numpy as np

import pybamm

from .electrode_soh import (
    _ElectrodeSOH,
    _ElectrodeSOHMSMR,
    _initial_soc_equations,
    _msmr_potential_equations,
    _soc_stoichiometries,
)
from .electrode_soh_composite import ElectrodeSOHComposite
from .electrode_soh_half_cell import (
    ElectrodeSOHHalfCell,
    _composite_soc_equations,
    _initial_voltage_equations,
)
from .util import _initial_state_parameters, _initial_state_variant, check_if_composite

_PLACEHOLDER_PREFIX = "__initial state unknown "


def _is_voltage(initial_value):
    return isinstance(initial_value, str) and initial_value.endswith("V")


def _standard_system(initial_value, param, options, direction, known_value):
    """Electrode SOH equations, plus the initial SOC equations for a voltage."""
    msmr = options.get("open-circuit potential") == "MSMR"
    soh_model = (_ElectrodeSOHMSMR if msmr else _ElectrodeSOH)(
        direction, param=param, known_value=known_value, options=options
    )
    capacities = {
        pybamm.InputParameter("Q_n"): param.n.Q_init,
        pybamm.InputParameter("Q_p"): param.p.Q_init,
        pybamm.InputParameter("Q_Li"): param.Q_Li_particles_init,
        pybamm.InputParameter("Q"): param.Q,
    }
    algebraic = {
        var: pybamm.replace(eqn, capacities) for var, eqn in soh_model.algebraic.items()
    }
    limits = tuple(
        pybamm.replace(soh_model.variables[name], capacities)
        for name in ["x_0", "x_100", "y_100", "y_0"]
    )

    if _is_voltage(initial_value):
        soc = pybamm.Variable("soc")
        V_init = float(initial_value[:-1])
        algebraic.update(
            _initial_soc_equations(soc, limits, V_init, param, options, direction)
        )
    else:
        soc = pybamm.Scalar(initial_value)
    x, y = _soc_stoichiometries(soc, *limits)
    if not msmr:
        return algebraic, {"x": x, "y": y}

    Un = pybamm.Variable("Un")
    Up = pybamm.Variable("Up")
    algebraic.update(_msmr_potential_equations(Un, Up, x, y, param))
    return algebraic, {"Un": Un, "Up": Up}


def _half_cell_system(initial_value, param, options, direction):
    """Equations for the initial stoichiometry of a half-cell working electrode."""
    is_composite = check_if_composite(options, "positive")
    x = pybamm.Variable("x")
    x_2 = pybamm.Variable("x_2") if is_composite else None
    outputs = {"x": x, "x_2": x_2} if is_composite else {"x": x}

    if _is_voltage(initial_value):
        V_init = float(initial_value[:-1])
        algebraic = _initial_voltage_equations(
            x, x_2, V_init, param, options, direction
        )
        return algebraic, outputs

    soh_model = ElectrodeSOHHalfCell(direction=direction, options=options)
    algebraic = dict(soh_model.algebraic)
    variables = soh_model.variables
    if not is_composite:
        x_0, x_100 = variables["x_0"], variables["x_100"]
        return algebraic, {"x": x_0 + initial_value * (x_100 - x_0)}

    limits = tuple(variables[name] for name in ["x_0", "x_100", "x_0_2", "x_100_2"])
    algebraic.update(
        _composite_soc_equations(
            x, x_2, initial_value, limits, param, options, direction
        )
    )
    return algebraic, outputs


def _composite_system(initial_value, param, options, direction):
    """Equations of :class:`ElectrodeSOHComposite` in terms of the parameters."""
    voltage = _is_voltage(initial_value)
    soh_model = ElectrodeSOHComposite(
        options,
        direction,
        initialization_method="voltage" if voltage else "SOC",
    )
    substitutions = {
        pybamm.InputParameter("Q_Li"): param.Q_Li_particles_init,
        pybamm.InputParameter("Q_n_1"): param.n.prim.Q_init,
        pybamm.InputParameter("Q_p_1"): param.p.prim.Q_init,
    }
    if check_if_composite(options, "negative"):
        substitutions[pybamm.InputParameter("Q_n_2")] = param.n.sec.Q_init
    if check_if_composite(options, "positive"):
        substitutions[pybamm.InputParameter("Q_p_2")] = param.p.sec.Q_init
    if voltage:
        substitutions[pybamm.InputParameter("V_init")] = float(initial_value[:-1])
    else:
        substitutions[pybamm.InputParameter("SOC_init")] = initial_value
    algebraic = {
        var: pybamm.replace(eqn, substitutions)
        for var, eqn in soh_model.algebraic.items()
    }
    outputs = {
        name: soh_model.variables[name]
        for name in ["x_init_1", "y_init_1", "x_init_2", "y_init_2"]
        if name in soh_model.variables
    }
    return algebraic, outputs


def _initial_state_system(initial_value, param, options, direction, known_value):
    """
    Equations whose solution sets the initial state, and the solved values that
    set it expressed in terms of their unknowns.
    """
    variant = _initial_state_variant(options)
    if variant == "half-cell":
        return _half_cell_system(initial_value, param, options, direction)
    if variant == "composite":
        return _composite_system(initial_value, param, options, direction)
    return _standard_system(initial_value, param, options, direction, known_value)


_SYSTEM_CACHE_SIZE = 16
_system_cache = {}


def _placeholder_system(initial_value, param, options, direction, known_value):
    """
    The initial-state equations with each unknown replaced by a placeholder input
    parameter, so that processing them leaves only input parameters.

    The same symbols are returned for repeated calls, so that processing them hits
    the parameter values' cache.

    Returns
    -------
    tuple
        The unknowns' names, the residuals and the initial-state parameter values
        keyed by parameter name.
    """
    key = (
        initial_value,
        id(param),
        repr(sorted(dict(options).items())),
        direction,
        known_value,
    )
    cached = _system_cache.get(key)
    # The cache holds param, so its id cannot be reused while the entry exists
    if cached is not None and cached[0] is param:
        return cached[1]

    algebraic, solved = _initial_state_system(
        initial_value, param, options, direction, known_value
    )
    outputs = {
        name: solved[key] * scale
        for name, (key, scale) in _initial_state_parameters(param, options).items()
    }
    placeholders = {
        var: pybamm.InputParameter(f"{_PLACEHOLDER_PREFIX}{i}")
        for i, var in enumerate(algebraic)
    }
    system = (
        [var.name for var in algebraic],
        [pybamm.replace(eqn, placeholders) for eqn in algebraic.values()],
        {name: pybamm.replace(value, placeholders) for name, value in outputs.items()},
    )
    if len(_system_cache) >= _SYSTEM_CACHE_SIZE:
        _system_cache.pop(next(iter(_system_cache)))
    _system_cache[key] = (param, system)
    return system


def _processed_system(
    initial_value, parameter_values, param, options, direction, known_value
):
    names, residuals, outputs = _placeholder_system(
        initial_value, param, options, direction, known_value
    )
    process = parameter_values.process_symbol
    return (
        names,
        [process(residual) for residual in residuals],
        {name: process(output) for name, output in outputs.items()},
    )


def _input_parameters(symbol):
    """The input parameters in a symbol, other than placeholders, keyed by name."""
    return {
        node.name: node
        for node in symbol.pre_order()
        if isinstance(node, pybamm.InputParameter)
        and not node.name.startswith(_PLACEHOLDER_PREFIX)
    }


@lru_cache(maxsize=16)
def _linearisation_function(residuals, outputs, input_names):
    """
    CasADi function of the unknowns and inputs returning the residuals and the
    outputs, each with its Jacobians with respect to both.
    """
    z = casadi.MX.sym("z", len(residuals))
    p = casadi.MX.sym("p", len(input_names))
    casadi_inputs = {name: p[k] for k, name in enumerate(input_names)}
    casadi_inputs.update(
        {f"{_PLACEHOLDER_PREFIX}{i}": z[i] for i in range(len(residuals))}
    )
    cache = {}
    results = []
    for symbols in [residuals, outputs]:
        value = casadi.vertcat(
            *[s.to_casadi(inputs=casadi_inputs, casadi_symbols=cache) for s in symbols]
        )
        results += [value, casadi.jacobian(value, z), casadi.jacobian(value, p)]
    return casadi.Function("linearisation", [z, p], results)


def get_initial_state_input_names(
    initial_value: float | str,
    parameter_values: pybamm.ParameterValues,
    param: pybamm.LithiumIonParameters,
    options: dict,
    direction: str | None = None,
    known_value: str = "cyclable lithium capacity",
) -> set[str]:
    """
    Names of the input parameters that the initial state set by
    :func:`set_initial_state` depends on.

    Parameters
    ----------
    initial_value : float or str
        Target initial SOC, or voltage as a string ending in "V".
    parameter_values : :class:`pybamm.ParameterValues`
        The parameter values.
    param : :class:`pybamm.LithiumIonParameters`
        The symbolic parameter set.
    options : dict-like
        Model options.
    direction : str, optional
        The OCV branch to use. Can be "charge" or "discharge".
    known_value : str, optional
        The known value needed to complete the electrode SOH model.

    Returns
    -------
    set of str
        The input parameter names.
    """
    _, residuals, outputs = _processed_system(
        initial_value, parameter_values, param, options, direction, known_value
    )
    names = set()
    for symbol in [*residuals, *outputs.values()]:
        names.update(_input_parameters(symbol))
    return names


def _scalar_input(name, value):
    """The value of an input in the electrode SOH equations, as a float."""
    value = np.asarray(value, dtype=float)
    if value.size != 1:
        raise pybamm.ModelError(
            "The initial state can only be differentiated with respect to scalar "
            f"input parameters, but '{name}' has {value.size} values."
        )
    return value.item()


def _linearise_about_solution(
    names, residuals, outputs, solution, inputs, tol, sensitivity_inputs=None
):
    """
    Linearise the outputs of an implicit system in the input parameters, about the
    solution at ``inputs``.

    For residuals :math:`G(z, p) = 0` solved by :math:`z^*` at inputs :math:`p^*`,
    each output :math:`f(z, p)` becomes :math:`f^* + f'(p - p^*)`, with
    :math:`f^* = f(z^*, p^*)` and :math:`f' = f_z z' + f_p`, where the implicit
    function theorem gives :math:`z' = -G_z^{-1} G_p` at :math:`(z^*, p^*)`.

    Parameters
    ----------
    names : list of str
        Name of each unknown, in the order of the placeholders.
    residuals : list of :class:`pybamm.Symbol`
        Processed residuals in terms of the placeholders and input parameters.
    outputs : dict
        Processed expressions in terms of the placeholders, keyed by name.
    solution : dict
        Value of each unknown at the solution, keyed by name.
    inputs : dict
        Input parameter values at which the system was solved.
    tol : float
        The tolerance the system was solved to.
    sensitivity_inputs : collection of str, optional
        The inputs to linearise in. Default is every input.

    Returns
    -------
    dict
        For each output, a float when it has no derivative with respect to these
        inputs, otherwise an expression in them.

    Raises
    ------
    :class:`pybamm.ModelError`
        If an input in the system has more than one value, or, when differentiating,
        the solution does not satisfy the residuals or their Jacobian with respect
        to the unknowns is singular.
    """
    z_star = np.array([float(np.squeeze(solution[name])) for name in names])
    parameters = {}
    for symbol in [*residuals, *outputs.values()]:
        parameters.update(_input_parameters(symbol))
    input_names = tuple(sorted(parameters))
    p_star = np.array([_scalar_input(name, inputs[name]) for name in input_names])
    function = _linearisation_function(
        tuple(residuals), tuple(outputs.values()), input_names
    )
    residual, residual_z, residual_p, value, value_z, value_p = (
        np.array(result) for result in function(z_star, p_star)
    )
    differentiated = [
        k
        for k, name in enumerate(input_names)
        if sensitivity_inputs is None or name in sensitivity_inputs
    ]

    derivative = np.zeros((len(outputs), len(differentiated)))
    if differentiated:
        # The solvers can stop at a bound short of a root, where there is no derivative
        max_residual = np.max(np.abs(residual))
        if max_residual > max(1e-5, 10 * tol):
            raise pybamm.ModelError(
                "Cannot differentiate the initial state with respect to the input "
                "parameters: the electrode SOH solution does not satisfy its "
                f"equations (maximum residual {max_residual:.2e}). Check that the "
                "electrode SOH problem is feasible for these inputs."
            )
        try:
            dz_dp = -np.linalg.solve(residual_z, residual_p[:, differentiated])
        except np.linalg.LinAlgError as error:
            raise pybamm.ModelError(
                "Cannot differentiate the initial state with respect to the input "
                "parameters: the electrode SOH equations are singular at the solution."
            ) from error
        derivative = value_z @ dz_dp + value_p[:, differentiated]

    linearised = {}
    for i, name in enumerate(outputs):
        terms = [
            pybamm.Scalar(derivative[i, j]) * (parameters[input_names[k]] - p_star[k])
            for j, k in enumerate(differentiated)
            if derivative[i, j] != 0
        ]
        output = value[i, 0].item()
        linearised[name] = sum(terms, pybamm.Scalar(output)) if terms else output
    return linearised


def _linearise_initial_state(
    solution: dict,
    initial_value: float | str,
    parameter_values: pybamm.ParameterValues,
    param: pybamm.LithiumIonParameters,
    options: dict,
    direction: str | None = None,
    known_value: str = "cyclable lithium capacity",
    tol: float = 1e-6,
    inputs: dict | None = None,
    sensitivity_inputs: Collection[str] | None = None,
) -> dict[str, float | pybamm.Symbol]:
    """
    Express the initial state in terms of the input parameters, with the value and
    first derivative of the solution at ``inputs``.

    Parameters
    ----------
    solution : dict
        The electrode SOH solution that sets the initial state, keyed by name.
    initial_value : float or str
        Target initial SOC, or voltage as a string ending in "V".
    parameter_values : :class:`pybamm.ParameterValues`
        The parameter values.
    param : :class:`pybamm.LithiumIonParameters`
        The symbolic parameter set.
    options : dict-like
        Model options.
    direction : str, optional
        The OCV branch to use. Can be "charge" or "discharge".
    known_value : str, optional
        The known value needed to complete the electrode SOH model.
    tol : float, optional
        The tolerance the electrode SOH was solved to. Default is 1e-6.
    inputs : dict, optional
        Input parameter values at which the electrode SOH was solved.
    sensitivity_inputs : collection of str, optional
        The inputs to differentiate with respect to. Default is every input.

    Returns
    -------
    dict
        The value of each initial-state parameter, a float or an expression in the
        inputs in ``sensitivity_inputs``.
    """
    names, residuals, outputs = _processed_system(
        initial_value, parameter_values, param, options, direction, known_value
    )
    return _linearise_about_solution(
        names, residuals, outputs, solution, inputs or {}, tol, sensitivity_inputs
    )
