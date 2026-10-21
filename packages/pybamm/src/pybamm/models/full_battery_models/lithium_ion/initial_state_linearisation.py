#
# Initial states that carry the derivatives of the electrode SOH solution with
# respect to input parameters
#
from __future__ import annotations

from functools import lru_cache

import casadi
import numpy as np

import pybamm

from .electrode_soh import ElectrodeSOHSolver, _ElectrodeSOH, _ElectrodeSOHMSMR
from .electrode_soh_composite import (
    ElectrodeSOHComposite,
    get_initial_stoichiometries_composite,
)
from .electrode_soh_half_cell import (
    ElectrodeSOHHalfCell,
    _solve_initial_stoichiometry_half_cell,
)
from .util import check_if_composite, get_initial_ocp, get_lithiation_delithiation

_PLACEHOLDER_PREFIX = "__initial state unknown "


def _is_voltage(initial_value):
    return isinstance(initial_value, str) and initial_value.endswith("V")


def _initial_state_variant(options):
    """Name the electrode SOH formulation used by :func:`set_initial_state`."""
    if options.get("open-circuit potential") == "MSMR":
        return "msmr"
    if options.get("working electrode") == "positive":
        return "half-cell"
    if check_if_composite(options, "positive") or check_if_composite(
        options, "negative"
    ):
        return "composite"
    return "standard"


def _soc_stoichiometries(soc, x_0, x_100, y_100, y_0):
    return x_0 + soc * (x_100 - x_0), y_0 - soc * (y_0 - y_100)


def _standard_system(initial_value, param, options, direction, known_value):
    """Electrode SOH equations, plus the initial SOC equation for a voltage."""
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
    limits = [
        pybamm.replace(soh_model.variables[name], capacities)
        for name in ["x_0", "x_100", "y_100", "y_0"]
    ]

    if _is_voltage(initial_value):
        V_init = float(initial_value[:-1])
        T_init = param.T_init
        soc = pybamm.Variable("soc")
        x, y = _soc_stoichiometries(soc, *limits)
        if msmr:
            Un = pybamm.Variable("Un(soc)")
            Up = pybamm.Variable("Up(soc)")
            algebraic[Un] = x - param.n.prim.x(Un, T_init)
            algebraic[Up] = y - param.p.prim.x(Up, T_init)
            algebraic[soc] = Up - Un - V_init
        else:
            Up = get_initial_ocp(
                param.p.prim, y, T_init, direction, "positive", options
            )
            Un = get_initial_ocp(
                param.n.prim, x, T_init, direction, "negative", options
            )
            algebraic[soc] = Up - Un - V_init
    else:
        x, y = _soc_stoichiometries(pybamm.Scalar(initial_value), *limits)

    if not msmr:
        return algebraic, {"x": x, "y": y}

    # MSMR initial states are electrode potentials at the reference temperature
    Un = pybamm.Variable("Un")
    Up = pybamm.Variable("Up")
    algebraic[Un] = param.n.prim.x(Un, param.T_ref) - x
    algebraic[Up] = param.p.prim.x(Up, param.T_ref) - y
    return algebraic, {"Un": Un, "Up": Up}


def _half_cell_system(initial_value, param, options, direction):
    """Equations for the initial stoichiometry of a half-cell working electrode."""
    is_composite = check_if_composite(options, "positive")
    lith_primary = get_lithiation_delithiation(
        direction, "positive", options, phase="primary"
    )
    lith_secondary = get_lithiation_delithiation(
        direction, "positive", options, phase="secondary"
    )
    x = pybamm.Variable("x")
    x_2 = pybamm.Variable("x_2")
    outputs = {"x": x, "x_2": x_2} if is_composite else {"x": x}

    if _is_voltage(initial_value):
        V_init = float(initial_value[:-1])
        T_init = param.T_init
        algebraic = {
            x: get_initial_ocp(
                param.p.prim, x, T_init, direction, "positive", options, "primary"
            )
            - V_init
        }
        if is_composite:
            algebraic[x_2] = (
                get_initial_ocp(
                    param.p.sec,
                    x_2,
                    T_init,
                    direction,
                    "positive",
                    options,
                    "secondary",
                )
                - V_init
            )
        return algebraic, outputs

    soh_model = ElectrodeSOHHalfCell(direction=direction, options=options)
    algebraic = dict(soh_model.algebraic)
    variables = soh_model.variables
    soc = pybamm.Scalar(initial_value)
    if not is_composite:
        x_0, x_100 = variables["x_0"], variables["x_100"]
        return algebraic, {"x": x_0 + soc * (x_100 - x_0)}

    Q_w = param.p.prim.Q_init
    Q_w_2 = param.p.sec.Q_init
    Q_min = variables["x_100"] * Q_w + variables["x_100_2"] * Q_w_2
    Q_max = variables["x_0"] * Q_w + variables["x_0_2"] * Q_w_2
    Q_now = x * Q_w + x_2 * Q_w_2
    algebraic[x] = param.p.prim.U(x, param.T_ref, lith_primary) - param.p.sec.U(
        x_2, param.T_ref, lith_secondary
    )
    algebraic[x_2] = (Q_now - Q_min) / (Q_max - Q_min) - soc
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
    Equations whose solution sets the initial state, and the initial-state values
    expressed in terms of their unknowns.
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
        The unknowns' names, the residuals and the outputs.
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

    algebraic, outputs = _initial_state_system(
        initial_value, param, options, direction, known_value
    )
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


def _input_names(symbol):
    return set(_input_parameters(symbol))


@lru_cache(maxsize=16)
def _residual_jacobians(residuals, input_names):
    """
    CasADi function of the unknowns and inputs returning the residuals and their
    Jacobians with respect to both.
    """
    z = casadi.MX.sym("z", len(residuals))
    p = casadi.MX.sym("p", len(input_names))
    casadi_inputs = {name: p[k] for k, name in enumerate(input_names)}
    casadi_inputs.update(
        {f"{_PLACEHOLDER_PREFIX}{i}": z[i] for i in range(len(residuals))}
    )
    cache = {}
    residual = casadi.vertcat(
        *[g.to_casadi(inputs=casadi_inputs, casadi_symbols=cache) for g in residuals]
    )
    return casadi.Function(
        "residual_jacobians",
        [z, p],
        [residual, casadi.jacobian(residual, z), casadi.jacobian(residual, p)],
    )


def get_initial_state_input_names(
    initial_value,
    parameter_values,
    param,
    options,
    direction=None,
    known_value="cyclable lithium capacity",
):
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
        names |= _input_names(symbol)
    # Stoichiometries are converted to concentrations with the maximum concentrations
    for phase in _stoichiometry_phases(param, options):
        names |= _input_names(parameter_values.process_symbol(phase.c_max))
    return names


def _stoichiometry_phases(param, options):
    """Phase parameters whose initial concentration is set from a stoichiometry."""
    variant = _initial_state_variant(options)
    if variant == "msmr":
        return []
    electrodes = ["positive"] if variant == "half-cell" else ["negative", "positive"]
    phases = []
    for electrode in electrodes:
        electrode_param = param.domain_params[electrode]
        phases.append(electrode_param.prim)
        if check_if_composite(options, electrode):
            phases.append(electrode_param.sec)
    return phases


def _linearise_about_solution(names, residuals, outputs, solution, inputs, tol):
    """
    Express the outputs of an implicit system as functions of the input parameters
    whose value and first derivative match the solved system at ``inputs``.

    For residuals :math:`G(z, p) = 0` solved by :math:`z^*` at inputs :math:`p^*`,
    each output :math:`f(z, p)` becomes :math:`f(z^* + z'(p - p^*), p)`, where the
    implicit function theorem gives
    :math:`z' = -(\\partial G / \\partial z)^{-1} \\partial G / \\partial p` at
    :math:`(z^*, p^*)`.

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

    Returns
    -------
    dict
        For each output, a float when it does not depend on any input parameter,
        otherwise an expression in the input parameters.

    Raises
    ------
    :class:`pybamm.ModelError`
        If an input in the residuals is not a scalar, the solution does not satisfy
        the residuals, or their Jacobian with respect to the unknowns is singular.
    """
    z_star = np.array([float(np.squeeze(solution[name])) for name in names])
    parameters = {}
    for residual in residuals:
        parameters.update(_input_parameters(residual))
    input_names = tuple(sorted(parameters))
    p_star = np.array([inputs[name] for name in input_names], dtype=float)
    if p_star.ndim != 1:
        raise pybamm.ModelError(
            "The initial state can only be differentiated with respect to scalar "
            "input parameters."
        )

    dz_dp = np.zeros((len(names), len(input_names)))
    if input_names:
        residual, jacobian_z, jacobian_p = (
            np.array(value)
            for value in _residual_jacobians(tuple(residuals), input_names)(
                z_star, p_star
            )
        )
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
            dz_dp = -np.linalg.solve(jacobian_z, jacobian_p)
        except np.linalg.LinAlgError as error:
            raise pybamm.ModelError(
                "Cannot differentiate the initial state with respect to the input "
                "parameters: the electrode SOH equations are singular at the solution."
            ) from error

    substitutions = {}
    for i in range(len(names)):
        value = pybamm.Scalar(z_star[i])
        for k, name in enumerate(input_names):
            if dz_dp[i, k] != 0:
                value = value + pybamm.Scalar(dz_dp[i, k]) * (
                    parameters[name] - p_star[k]
                )
        substitutions[pybamm.InputParameter(f"{_PLACEHOLDER_PREFIX}{i}")] = value

    linearised = {}
    for name, output in outputs.items():
        output = pybamm.replace(output, substitutions)
        if _input_names(output):
            linearised[name] = output
        else:
            linearised[name] = float(np.squeeze(output.evaluate()))
    return linearised


def get_linearised_initial_state(
    initial_value,
    parameter_values,
    param,
    options,
    direction=None,
    known_value="cyclable lithium capacity",
    tol=1e-6,
    inputs=None,
    esoh_solver=None,
):
    """
    Solve for the initial state and express it in terms of the input parameters,
    with the value and first derivative of the solution at ``inputs``.

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
    tol : float, optional
        The tolerance for the electrode SOH solvers. Default is 1e-6.
    inputs : dict, optional
        Input parameter values at which to solve.
    esoh_solver : optional
        A pre-built electrode SOH solver or simulation to reuse.

    Returns
    -------
    dict
        The initial state values keyed as in :func:`_initial_state_system`, each a
        float or an expression in the input parameters.
    """
    inputs = inputs or {}
    variant = _initial_state_variant(options)
    if variant == "half-cell":
        solution = _solve_initial_stoichiometry_half_cell(
            initial_value,
            parameter_values,
            options=options,
            tol=tol,
            inputs=inputs,
            direction=direction,
            esoh_sim=esoh_solver,
        )
    elif variant == "composite":
        solution = get_initial_stoichiometries_composite(
            initial_value,
            parameter_values,
            direction=direction,
            param=param,
            options=options,
            tol=tol,
            inputs=inputs,
            known_value=known_value,
            esoh_sim=esoh_solver,
        )
    else:
        if esoh_solver is None:
            esoh_solver = ElectrodeSOHSolver(
                parameter_values,
                direction=direction,
                param=param,
                known_value=known_value,
                options=options,
            )
        if variant == "msmr":
            solution = esoh_solver._solve_initial_ocps(
                initial_value, direction=direction, tol=tol, inputs=inputs
            )
        else:
            solution = esoh_solver._solve_initial_stoichiometries(
                initial_value, direction=direction, tol=tol, inputs=inputs
            )
    names, residuals, outputs = _processed_system(
        initial_value, parameter_values, param, options, direction, known_value
    )
    return _linearise_about_solution(names, residuals, outputs, solution, inputs, tol)
