#
# A model to calculate electrode-specific SOH, adapted for composite electrodes
#
from __future__ import annotations

import warnings
from typing import Any

import casadi
import numpy as np

import pybamm

from .electrode_soh import _ElectrodeSOH, get_esoh_default_solver
from .util import (
    check_if_composite,
    get_equilibrium_direction,
    get_lithiation_delithiation,
)

# `LithiumIonParameters.U` clips its stoichiometry and adds an asymptote, so every OCP
# diverges outside [0, 1] and both brackets straddle a root. The inner one is wider so
# an inversion can always reach the potential the outer solve asks it for.
_STOICH_LO, _STOICH_HI = -10.0, 10.0
_BRACKET_LO, _BRACKET_HI = -5.0, 5.0

_ABSTOL, _MAX_ITER = 1e-14, 200


def _get_primary_only_options(
    options: dict | pybamm.BatteryModelOptions | None,
) -> dict | None:
    """
    Create options dict with only primary phase OCP settings.

    When composite options have tuple OCP settings like
    (("single", "hysteresis"), "single"), the non-composite model needs only the
    primary (first) element for each electrode to avoid incorrectly detecting
    hysteresis from the secondary phase.

    Parameters
    ----------
    options : dict or pybamm.BatteryModelOptions or None
        Model options that may contain composite OCP settings.

    Returns
    -------
    dict or None
        Options dict with primary-only OCP settings, or None if input is None.

    Examples
    --------
    >>> opts = _get_primary_only_options(
    ...     {"open-circuit potential": (("single", "current sigmoid"), "single")}
    ... )
    >>> opts["open-circuit potential"]
    ('single', 'single')
    """
    if options is None:
        return None

    if not isinstance(options, pybamm.BatteryModelOptions):
        options = pybamm.BatteryModelOptions(options)

    options_dict = dict(options)

    ocp_option = options_dict.get("open-circuit potential")
    if ocp_option is None:
        return options_dict

    if isinstance(ocp_option, str):
        return options_dict

    if isinstance(ocp_option, tuple) and len(ocp_option) == 2:
        neg_ocp, pos_ocp = ocp_option

        if isinstance(neg_ocp, tuple):
            neg_ocp = neg_ocp[0]

        if isinstance(pos_ocp, tuple):
            pos_ocp = pos_ocp[0]

        options_dict["open-circuit potential"] = (neg_ocp, pos_ocp)

    return options_dict


def _initialization_method(initial_value: float | str) -> str:
    """``"voltage"`` for a string ending in ``"V"``, ``"SOC"`` for a float."""
    if isinstance(initial_value, str) and initial_value.endswith("V"):
        return "voltage"
    if isinstance(initial_value, float):
        return "SOC"
    raise ValueError(
        "Invalid initial value. Expected a float between 0 and 1 "
        "(for SOC) or a string ending in 'V' (for voltage), got "
        f"{initial_value!r} of type {type(initial_value).__name__}"
    )


def _get_stoich_variables(options):
    """Create stoichiometry variables for composite electrodes."""
    variables = {
        "x_100_1": pybamm.Variable("x_100_1", bounds=(0, 1)),
        "y_100_1": pybamm.Variable("y_100_1", bounds=(0, 1)),
        "x_0_1": pybamm.Variable("x_0_1", bounds=(0, 1)),
        "y_0_1": pybamm.Variable("y_0_1", bounds=(0, 1)),
        "x_init_1": pybamm.Variable("x_init_1", bounds=(0, 1)),
        "y_init_1": pybamm.Variable("y_init_1", bounds=(0, 1)),
    }
    is_positive_composite = check_if_composite(options, "positive")
    is_negative_composite = check_if_composite(options, "negative")
    if is_positive_composite:
        variables["y_100_2"] = pybamm.Variable("y_100_2", bounds=(0, 1))
        variables["y_0_2"] = pybamm.Variable("y_0_2", bounds=(0, 1))
        variables["y_init_2"] = pybamm.Variable("y_init_2", bounds=(0, 1))
    if is_negative_composite:
        variables["x_100_2"] = pybamm.Variable("x_100_2", bounds=(0, 1))
        variables["x_0_2"] = pybamm.Variable("x_0_2", bounds=(0, 1))
        variables["x_init_2"] = pybamm.Variable("x_init_2", bounds=(0, 1))
    return variables


def _get_initial_conditions(options, soc_init):
    """Get initial conditions for stoichiometry variables."""
    variables = _get_stoich_variables(options)
    ics = {}
    # Avoid exact boundary values for better numerical stability
    eps = 0.01
    for name, var in variables.items():
        if "100" in name and "x" in name:
            ics[var] = 0.85
        elif ("0" in name and "x" in name) or ("100" in name and "y" in name):
            ics[var] = 0.15
        elif "0" in name and "y" in name:
            ics[var] = 0.85
        elif "init" in name and "x" in name:
            ics[var] = pybamm.maximum(eps, pybamm.minimum(1 - eps, soc_init))
        elif "init" in name and "y" in name:
            ics[var] = pybamm.maximum(eps, pybamm.minimum(1 - eps, 1 - soc_init))
    return ics


def _get_direction(electrode):
    """Get direction multiplier for electrode capacity calculations."""
    if electrode == "positive":
        return pybamm.Scalar(-1)
    else:
        return pybamm.Scalar(1)


def _get_prefix(electrode):
    """Get stoichiometry variable prefix for electrode ('x' or 'y')."""
    if electrode == "positive":
        return "y"
    else:
        return "x"


def _get_electrode_capacity_equation(options, electrode):
    """
    Build equation for electrode capacity in composite electrodes.

    Returns Q = sum_i Q_i * (stoich_100_i - stoich_0_i) for all phases.
    """
    prefix = _get_prefix(electrode)
    e = electrode[0]
    i_am_composite = check_if_composite(options, electrode)
    stoich_variables = _get_stoich_variables(options)
    direction = _get_direction(electrode)
    Q_1 = pybamm.InputParameter(f"Q_{e}_1")
    Q = (
        direction
        * (stoich_variables[f"{prefix}_100_1"] - stoich_variables[f"{prefix}_0_1"])
        * Q_1
    )
    if i_am_composite:
        Q_2 = pybamm.InputParameter(f"Q_{e}_2")
        Q += (
            direction
            * (stoich_variables[f"{prefix}_100_2"] - stoich_variables[f"{prefix}_0_2"])
            * Q_2
        )
    return Q


def _get_cyclable_lithium_equation(options, soc="100"):
    """
    Build equation for total cyclable lithium in composite electrodes.

    Returns Q_Li = sum_i (Q_n_i * x_i + Q_p_i * y_i) for all phases at given SOC.
    """
    x_soc_1 = pybamm.Variable(f"x_{soc}_1", bounds=(0, 1))
    y_soc_1 = pybamm.Variable(f"y_{soc}_1", bounds=(0, 1))
    Q_n_1 = pybamm.InputParameter("Q_n_1")
    Q_p_1 = pybamm.InputParameter("Q_p_1")
    lithium_primary_phases = Q_n_1 * x_soc_1 + Q_p_1 * y_soc_1
    lithium_secondary_phases = 0.0
    is_positive_composite = check_if_composite(options, "positive")
    is_negative_composite = check_if_composite(options, "negative")
    if is_positive_composite:
        Q_p_2 = pybamm.InputParameter("Q_p_2")
        y_soc_2 = pybamm.Variable(f"y_{soc}_2", bounds=(0, 1))
        lithium_secondary_phases += Q_p_2 * y_soc_2
    if is_negative_composite:
        Q_n_2 = pybamm.InputParameter("Q_n_2")
        x_soc_2 = pybamm.Variable(f"x_{soc}_2", bounds=(0, 1))
        lithium_secondary_phases += Q_n_2 * x_soc_2
    return lithium_primary_phases + lithium_secondary_phases


def _solve_secondary_stoichiometry(
    primary_stoich: float,
    parameter_values: pybamm.ParameterValues,
    param: pybamm.LithiumIonParameters,
    electrode: str,
    direction: str | None,
    options: dict,
    T: float,
    tol: float = 1e-6,
) -> float:
    """
    Solve U_prim(z_1) = U_sec(z_2) to get z_2 given z_1.

    Parameters
    ----------
    primary_stoich : float
        The primary phase stoichiometry (x_1 or y_1)
    parameter_values : pybamm.ParameterValues
        The parameter values
    param : pybamm.LithiumIonParameters
        The parameter object
    electrode : str
        "negative" or "positive"
    direction : str
        "charge" or "discharge"
    options : dict
        Model options
    T : float
        Temperature
    tol : float
        Solver tolerance

    Returns
    -------
    float
        The secondary phase stoichiometry (x_2 or y_2)
    """
    model = pybamm.BaseModel()
    z_2 = pybamm.Variable("z_2", bounds=(0, 1))
    z_1 = pybamm.InputParameter("z_1")

    if electrode == "negative":
        lith_prim = get_lithiation_delithiation(
            direction, "negative", options, phase="primary"
        )
        lith_sec = get_lithiation_delithiation(
            direction, "negative", options, phase="secondary"
        )
        U_prim = param.n.prim.U(z_1, T, lith_prim)
        U_sec = param.n.sec.U(z_2, T, lith_sec)
    else:
        lith_prim = get_lithiation_delithiation(
            direction, "positive", options, phase="primary"
        )
        lith_sec = get_lithiation_delithiation(
            direction, "positive", options, phase="secondary"
        )
        U_prim = param.p.prim.U(z_1, T, lith_prim)
        U_sec = param.p.sec.U(z_2, T, lith_sec)

    model.algebraic[z_2] = U_prim - U_sec
    model.initial_conditions[z_2] = primary_stoich
    model.variables["z_2"] = z_2

    sim = pybamm.Simulation(
        model, parameter_values=parameter_values, solver=get_esoh_default_solver(tol)
    )
    sol = sim.solve([0], inputs={"z_1": primary_stoich})
    return sol["z_2"].data[0]


class ElectrodeSOHComposite(pybamm.BaseModel):
    """Model to calculate electrode-specific SOH for a cell with composite electrodes,
    adapted from :footcite:t:`Mohtat2019`. This model is mainly for internal use, to
    calculate summary variables in a simulation.

    Subscript 1 indicates primary phase and subscript 2 indicates secondary phase.

    The model calculates stoichiometries at three states:

    - 100% SOC (x_100, y_100): Limit state reached via charging
    - 0% SOC (x_0, y_0): Limit state reached via discharging
    - Initial SOC (x_init, y_init): Dynamic state, uses specified direction

    Stoichiometry limits (_100 and _0 variables) are evaluated in a
    temperature-independent manner (at the reference temperature, ignoring entropy
    effects), so that the SOC range is not temperature-dependent. In the presence of
    a hysteresis model, equilibration in the charging direction is assumed for 100%
    SOC (charging OCP branch for each material), and equilibration in the discharging
    direction is assumed for 0% SOC (discharging OCP branch for each material). The
    initial stoichiometries (_init variables) use the specified direction to account
    for hysteresis during charge/discharge.

    Parameters
    ----------
    options : dict
        Model options including particle phases and OCP settings
    direction : str, optional
        "charge" or "discharge" - only affects initial stoichiometry calculation
    name : str, optional
        Model name (default: "ElectrodeSOH model")
    initialization_method : str, optional
        "voltage" or "SOC" (default: "voltage")
    """

    def __init__(
        self,
        options,
        direction=None,
        name="ElectrodeSOH model",
        initialization_method="voltage",
    ):
        pybamm.citations.register("Mohtat2019")
        super().__init__(name)
        param = pybamm.LithiumIonParameters(options)
        neg_composite = check_if_composite(options, "negative")
        pos_composite = check_if_composite(options, "positive")

        Q_Li = pybamm.InputParameter("Q_Li")
        Q_n = [pybamm.InputParameter("Q_n_1")]
        Q_p = [pybamm.InputParameter("Q_p_1")]
        if neg_composite:
            Q_n.append(pybamm.InputParameter("Q_n_2"))
        if pos_composite:
            Q_p.append(pybamm.InputParameter("Q_p_2"))

        T_ref = param.T_ref
        T_init = param.T_init if initialization_method == "voltage" else T_ref

        def ocps(electrode, soc):
            """(open-circuit potential, branch, name) per phase, at an SOC level."""
            side = param.n if electrode == "negative" else param.p
            phases = ["primary"]
            if check_if_composite(options, electrode):
                phases.append("secondary")
            out = []
            for phase in phases:
                U = (side.prim if phase == "primary" else side.sec).U
                if soc == "init":
                    branch = get_lithiation_delithiation(
                        direction, electrode, options, phase=phase
                    )
                else:
                    branch = get_lithiation_delithiation(
                        get_equilibrium_direction(soc, electrode, options, phase),
                        electrode,
                        options,
                        phase=phase,
                    )
                out.append((U, branch, f"{electrode} {phase} {soc}"))
            return out

        def invert(U, branch, target, T, name):
            """Stoichiometry at a given potential, by Brent."""
            sto = pybamm._BrentUnknown(name)
            return pybamm._Brent(
                U(sto, T, branch) - target,
                sto,
                (_STOICH_LO, _STOICH_HI),
                abstol=_ABSTOL,
                max_iter=_MAX_ITER,
            )

        def bracket(pairs, offsets, T):
            """Potentials for which every inversion in `pairs` brackets."""
            los = [
                U(pybamm.Scalar(_BRACKET_HI), T, b) - off
                for (U, b, _), off in zip(pairs, offsets, strict=True)
            ]
            his = [
                U(pybamm.Scalar(_BRACKET_LO), T, b) - off
                for (U, b, _), off in zip(pairs, offsets, strict=True)
            ]
            lo, hi = los[0], his[0]
            for value in los[1:]:
                lo = pybamm.maximum(lo, value)
            for value in his[1:]:
                hi = pybamm.minimum(hi, value)
            return lo, hi

        def limits(soc, V_target, closure):
            """A limit state: one unknown, the shared negative potential."""
            neg, pos = ocps("negative", soc), ocps("positive", soc)

            # built twice: once against the unknown potential, once against the solved
            def states(U_n, suffix):
                return (
                    [invert(U, b, U_n, T_ref, f"{n} {suffix}") for U, b, n in neg],
                    [
                        invert(U, b, U_n + V_target, T_ref, f"{n} {suffix}")
                        for U, b, n in pos
                    ],
                )

            U_n = pybamm._BrentUnknown(f"negative potential {soc}")
            lo, hi = bracket(neg + pos, [0] * len(neg) + [V_target] * len(pos), T_ref)
            solved = pybamm._Brent(
                closure(*states(U_n, "guess")),
                U_n,
                (lo, hi),
                abstol=_ABSTOL,
                max_iter=_MAX_ITER,
            )
            return states(solved, "solved")

        def total_lithium(x, y):
            return sum(q * s for q, s in zip(Q_n + Q_p, x + y, strict=True))

        x_100, y_100 = limits(
            "100", param.ocp_soc_100, lambda x, y: Q_Li - total_lithium(x, y)
        )
        x_0, y_0 = limits(
            "0",
            param.ocp_soc_0,
            lambda x, y: (
                -sum(q * (a - b) for q, a, b in zip(Q_p, y_100, y, strict=True))
                - sum(q * (a - b) for q, a, b in zip(Q_n, x_100, x, strict=True))
            ),
        )

        neg_init, pos_init = ocps("negative", "init"), ocps("positive", "init")

        def init_states(x_init_1, suffix):
            U_n = neg_init[0][0](x_init_1, T_init, neg_init[0][1])
            x = [x_init_1] + [
                invert(U, b, U_n, T_init, f"{n} {suffix}") for U, b, n in neg_init[1:]
            ]
            if initialization_method == "voltage":
                U_p = U_n + pybamm.InputParameter("V_init")
                return x, [
                    invert(U, b, U_p, T_init, f"{n} {suffix}") for U, b, n in pos_init
                ]
            # No cell-voltage relation: lithium conservation fixes the positive side.
            remaining = Q_Li - sum(q * s for q, s in zip(Q_n, x, strict=True))
            if not pos_composite:
                return x, [remaining / Q_p[0]]
            U_p = pybamm._BrentUnknown(f"positive potential {suffix}")
            lo, hi = bracket(pos_init, [0] * len(pos_init), T_init)
            solved = pybamm._Brent(
                sum(
                    q * invert(U, b, U_p, T_init, f"{n} {suffix} guess")
                    for q, (U, b, n) in zip(Q_p, pos_init, strict=True)
                )
                - remaining,
                U_p,
                (lo, hi),
                abstol=_ABSTOL,
                max_iter=_MAX_ITER,
            )
            return x, [
                invert(U, b, solved, T_init, f"{n} {suffix} solved")
                for U, b, n in pos_init
            ]

        unknown = pybamm._BrentUnknown("x_init_1")
        x_guess, y_guess = init_states(unknown, "guess")
        if initialization_method == "voltage":
            closure = total_lithium(x_guess, y_guess) - Q_Li
        elif initialization_method == "SOC":

            def charge(x):
                return sum(q * s for q, s in zip(Q_n, x, strict=True))

            closure = (charge(x_guess) - charge(x_0)) / (
                charge(x_100) - charge(x_0)
            ) - pybamm.InputParameter("SOC_init")
        else:
            raise pybamm.OptionError(
                f"Invalid initialization method '{initialization_method}', "
                "expected 'voltage' or 'SOC'"
            )

        # Solve for x_init_1 rather than the potential: the residual is sensitive to
        # it, and a non-physical target still has an exact answer.
        x_init_1 = pybamm._Brent(
            closure,
            unknown,
            (_BRACKET_LO, _BRACKET_HI),
            abstol=_ABSTOL,
            max_iter=_MAX_ITER,
        )
        x_init, y_init = init_states(x_init_1, "solved")

        # the stoichiometries are expressions, so the solver needs a state of its own;
        # the placeholder is kept out of `variables`
        placeholder = pybamm.Variable("ESOH placeholder")
        self.algebraic = {placeholder: placeholder}
        self.initial_conditions = {placeholder: pybamm.Scalar(0)}

        for index, value in enumerate(x_100):
            self.variables[f"x_100_{index + 1}"] = value
        for index, value in enumerate(y_100):
            self.variables[f"y_100_{index + 1}"] = value
        for index, value in enumerate(x_0):
            self.variables[f"x_0_{index + 1}"] = value
        for index, value in enumerate(y_0):
            self.variables[f"y_0_{index + 1}"] = value
        for index, value in enumerate(x_init):
            self.variables[f"x_init_{index + 1}"] = value
        for index, value in enumerate(y_init):
            self.variables[f"y_init_{index + 1}"] = value

        # set by `_esoh_evaluator`, keyed on the parameter values
        self._evaluator: tuple | None = None

    @property
    def default_solver(self):
        return get_esoh_default_solver()

    @staticmethod
    def solve_split(
        initial_value: float | str,
        parameter_values: pybamm.ParameterValues,
        direction: str | None = None,
        param: pybamm.LithiumIonParameters | None = None,
        options: dict | None = None,
        tol: float = 1e-6,
        inputs: dict | None = None,
    ) -> dict[str, float]:
        """
        Split solve approach for composite electrode SOH.

        Step 1: Solve for primary stoichiometries using non-composite model
        Step 2: Solve U_prim(z_1) = U_sec(z_2) for secondary stoichiometries

        The equilibrium stoichiometries (x_100, x_0, y_100, y_0) are calculated
        using direction=None (equilibrium branch). Only initial stoichiometries
        (x_init, y_init) use the specified direction to account for hysteresis during
        charge/discharge.

        Parameters
        ----------
        initial_value : float or str
            Target initial value. If float (0-1), interpreted as SOC.
            If string ending in 'V', interpreted as voltage.
        parameter_values : pybamm.ParameterValues
            Parameter values for the simulation
        direction : str, optional
            "charge" or "discharge" for hysteresis direction (only affects
            initial stoichiometries, not equilibrium values)
        param : pybamm.LithiumIonParameters, optional
            Parameter object
        options : dict, optional
            Model options
        tol : float, optional
            Solver tolerance (default 1e-6)
        inputs : dict, optional
            Additional inputs

        Returns
        -------
        dict
            Dictionary of stoichiometry values
        """
        inputs = inputs or {}
        param = param or pybamm.LithiumIonParameters(options)

        is_positive_composite = check_if_composite(options, "positive")
        is_negative_composite = check_if_composite(options, "negative")

        Q_n_1 = parameter_values.evaluate(param.n.prim.Q_init, inputs=inputs)
        Q_p_1 = parameter_values.evaluate(param.p.prim.Q_init, inputs=inputs)
        Qs = {"Q_n_1": Q_n_1, "Q_p_1": Q_p_1}
        if is_positive_composite:
            Q_p_2 = parameter_values.evaluate(param.p.sec.Q_init, inputs=inputs)
            Qs["Q_p_2"] = Q_p_2
        if is_negative_composite:
            Q_n_2 = parameter_values.evaluate(param.n.sec.Q_init, inputs=inputs)
            Qs["Q_n_2"] = Q_n_2

        Q_Li = parameter_values.evaluate(param.Q_Li_particles_init, inputs=inputs)

        if isinstance(initial_value, str) and initial_value.endswith("V"):
            V_init = float(initial_value[:-1])
            initialization_method = "voltage"
        elif isinstance(initial_value, float):
            initialization_method = "SOC"
            if initial_value > 1:
                warnings.warn(
                    message=f"Initial SoC {initial_value} is greater than 1",
                    category=UserWarning,
                    stacklevel=2,
                )
            elif initial_value < 0:
                warnings.warn(
                    message=f"Initial SoC {initial_value} is less than 0",
                    category=UserWarning,
                    stacklevel=2,
                )

        else:
            raise ValueError(
                "Invalid initial value. Expected a float (for SoC, "
                "1.0 for 100%) or a string ending in 'V' (for voltage), got "
                f"{initial_value!r} of type {type(initial_value).__name__}"
            )

        Q_n_total = Q_n_1 + (Qs.get("Q_n_2", 0))
        Q_p_total = Q_p_1 + (Qs.get("Q_p_2", 0))

        primary_options = _get_primary_only_options(options)
        # _ElectrodeSOH uses get_equilibrium_direction internally for the equilibrium
        # stoichiometries (x_0, x_100, y_0, y_100), consistent with the full solve
        primary_model = _ElectrodeSOH(
            direction=direction,
            param=param,
            solve_for=["x_0", "x_100"],
            known_value="cyclable lithium capacity",
            options=primary_options,
        )

        primary_inputs = {**inputs, "Q_n": Q_n_total, "Q_p": Q_p_total, "Q_Li": Q_Li}
        primary_sim = pybamm.Simulation(
            primary_model,
            parameter_values=parameter_values,
            solver=get_esoh_default_solver(tol),
        )

        primary_sol = primary_sim.solve([0], inputs=primary_inputs)
        x_100_1 = primary_sol["x_100"].data[0]
        x_0_1 = primary_sol["x_0"].data[0]
        y_100_1 = primary_sol["y_100"].data[0]
        y_0_1 = primary_sol["y_0"].data[0]

        T_ref = parameter_values["Reference temperature [K]"]
        result = {
            "x_100_1": x_100_1,
            "x_0_1": x_0_1,
            "y_100_1": y_100_1,
            "y_0_1": y_0_1,
        }

        if is_negative_composite:
            result["x_100_2"] = _solve_secondary_stoichiometry(
                x_100_1,
                parameter_values,
                param,
                "negative",
                "charge",  # 100% SOC is reached via charging
                options,
                T_ref,
                tol,
            )
            result["x_0_2"] = _solve_secondary_stoichiometry(
                x_0_1,
                parameter_values,
                param,
                "negative",
                "discharge",  # 0% SOC is reached via discharging
                options,
                T_ref,
                tol,
            )

        if is_positive_composite:
            result["y_100_2"] = _solve_secondary_stoichiometry(
                y_100_1,
                parameter_values,
                param,
                "positive",
                "charge",  # 100% SOC is reached via charging
                options,
                T_ref,
                tol,
            )
            result["y_0_2"] = _solve_secondary_stoichiometry(
                y_0_1,
                parameter_values,
                param,
                "positive",
                "discharge",  # 0% SOC is reached via discharging
                options,
                T_ref,
                tol,
            )

        if initialization_method == "voltage":
            T_init = parameter_values["Initial temperature [K]"]
            soc_model = pybamm.BaseModel()
            x_init = pybamm.Variable("x_init", bounds=(0, 1))
            y_init = y_0_1 + (x_init - x_0_1) / (x_100_1 - x_0_1) * (y_100_1 - y_0_1)
            lith_pos = get_lithiation_delithiation(
                direction, "positive", options, phase="primary"
            )
            lith_neg = get_lithiation_delithiation(
                direction, "negative", options, phase="primary"
            )
            Up = param.p.prim.U(y_init, T_init, lith_pos)
            Un = param.n.prim.U(x_init, T_init, lith_neg)
            soc_model.algebraic[x_init] = Up - Un - V_init
            soc_model.initial_conditions[x_init] = (x_0_1 + x_100_1) / 2
            soc_model.variables["x_init"] = x_init
            soc_model.variables["y_init"] = y_init
            soc_sim = pybamm.Simulation(
                soc_model,
                parameter_values=parameter_values,
                solver=get_esoh_default_solver(tol),
            )
            soc_sol = soc_sim.solve([0], inputs=inputs)
            x_init_1 = soc_sol["x_init"].data[0]
            y_init_1 = soc_sol["y_init"].data[0]
        else:  # SOC initialization
            soc = initial_value
            x_init_1 = x_0_1 + soc * (x_100_1 - x_0_1)
            y_init_1 = y_0_1 + soc * (y_100_1 - y_0_1)

        result["x_init_1"] = x_init_1
        result["y_init_1"] = y_init_1

        T_for_init = (
            parameter_values["Initial temperature [K]"]
            if initialization_method == "voltage"
            else T_ref
        )
        if is_negative_composite:
            result["x_init_2"] = _solve_secondary_stoichiometry(
                x_init_1,
                parameter_values,
                param,
                "negative",
                direction,
                options,
                T_for_init,
                tol,
            )
        if is_positive_composite:
            result["y_init_2"] = _solve_secondary_stoichiometry(
                y_init_1,
                parameter_values,
                param,
                "positive",
                direction,
                options,
                T_for_init,
                tol,
            )

        return result

    @staticmethod
    def solve_full(
        initial_value: float | str,
        parameter_values: pybamm.ParameterValues,
        direction: str | None = None,
        param: pybamm.LithiumIonParameters | None = None,
        options: dict | None = None,
        tol: float = 1e-6,
        inputs: dict | None = None,
        initial_conditions: dict[str, float] | None = None,
        esoh_sim: pybamm.Simulation | None = None,
    ) -> dict[str, float]:
        """
        Full solve approach: solve all stoichiometries simultaneously.

        Uses the full ElectrodeSOHComposite algebraic model to solve for all
        stoichiometries at once. The equilibrium stoichiometries (x_100, x_0,
        y_100, y_0) are calculated using direction=None (equilibrium branch).
        Only initial stoichiometries (x_init, y_init) use the specified direction to
        account for hysteresis.

        Parameters
        ----------
        initial_value : float or str
            Target initial value. If float (0-1), interpreted as SOC.
            If string ending in 'V', interpreted as voltage.
        parameter_values : pybamm.ParameterValues
            Parameter values for the simulation
        direction : str, optional
            "charge" or "discharge" for hysteresis direction (only affects
            initial stoichiometries, not equilibrium values)
        param : pybamm.LithiumIonParameters, optional
            Parameter object
        options : dict, optional
            Model options
        tol : float, optional
            Solver tolerance (default 1e-6)
        inputs : dict, optional
            Additional inputs
        initial_conditions : dict, optional
            Accepted and ignored. Each stoichiometry is found by a bracketed
            rootfind, which needs a bracket rather than a starting guess.
        esoh_sim : :class:`pybamm.Simulation`, optional
            A pre-built simulation wrapping an :class:`ElectrodeSOHComposite` model.
            Passing one back reuses its compiled evaluator across calls.

        Returns
        -------
        dict
            Dictionary of stoichiometry values
        """
        inputs = inputs or {}
        param = param or pybamm.LithiumIonParameters(options)

        is_positive_composite = check_if_composite(options, "positive")
        is_negative_composite = check_if_composite(options, "negative")

        Q_n_1 = parameter_values.evaluate(param.n.prim.Q_init, inputs=inputs)
        Q_p_1 = parameter_values.evaluate(param.p.prim.Q_init, inputs=inputs)
        Qs = {"Q_n_1": Q_n_1, "Q_p_1": Q_p_1}
        if is_positive_composite:
            Q_p_2 = parameter_values.evaluate(param.p.sec.Q_init, inputs=inputs)
            Qs["Q_p_2"] = Q_p_2
        if is_negative_composite:
            Q_n_2 = parameter_values.evaluate(param.n.sec.Q_init, inputs=inputs)
            Qs["Q_n_2"] = Q_n_2

        Q_Li = parameter_values.evaluate(param.Q_Li_particles_init, inputs=inputs)

        initialization_method = _initialization_method(initial_value)
        if initialization_method == "voltage":
            V_init = float(initial_value[:-1])
        elif initial_value > 1:
            warnings.warn(
                message=f"Initial SoC {initial_value} is greater than 1",
                category=UserWarning,
                stacklevel=2,
            )
        elif initial_value < 0:
            warnings.warn(
                message=f"Initial SoC {initial_value} is less than 0",
                category=UserWarning,
                stacklevel=2,
            )

        all_inputs = {**inputs, **Qs, "Q_Li": Q_Li}
        if initialization_method == "voltage":
            all_inputs["V_init"] = V_init
        else:
            all_inputs["SOC_init"] = initial_value

        if esoh_sim is None:
            model = ElectrodeSOHComposite(
                options, direction, initialization_method=initialization_method
            )
        else:
            model = esoh_sim.model

        names, input_names, function = _esoh_evaluator(model, parameter_values)
        try:
            values = np.asarray(
                function(*[all_inputs[name] for name in input_names])
            ).reshape(-1)
        except RuntimeError as error:
            # the native rootfinder reports failure as RuntimeError; callers, and the
            # fallback in get_initial_stoichiometries_composite, expect a SolverError
            raise pybamm.SolverError(
                f"Composite electrode SOH solve failed: {error}"
            ) from error
        if not np.all(np.isfinite(values)):
            raise pybamm.SolverError(
                "Composite electrode SOH solve returned a non-finite stoichiometry"
            )
        return dict(zip(names, values, strict=True))


def _unique_nodes(roots):
    """Every symbol reachable from ``roots``, visited once.

    Unlike ``pre_order``, which re-yields a shared node once per path to it.
    """
    seen: set = set()
    nodes = []
    stack = list(roots)
    while stack:
        symbol = stack.pop()
        if id(symbol) in seen:
            continue
        seen.add(id(symbol))
        nodes.append(symbol)
        stack.extend(symbol.children)
    return nodes


def _parameter_fingerprint(parameter_values, names):
    """A comparable summary of the parameters in ``names``.

    Numbers compare by value and everything else by identity, so replacing an OCP
    function counts as a change but re-reading the same one does not.
    """
    return tuple(
        (name, value)
        if isinstance(value := parameter_values[name], (int, float))
        else (name, id(value))
        for name in sorted(names)
        if name in parameter_values
    )


def _esoh_evaluator(model, parameter_values):
    """Map capacities and target straight to stoichiometries.

    The model defines each stoichiometry as an expression, not a state, so one CasADi
    function replaces a solve. Cached on ``model`` per parameter set.

    Returns
    -------
    tuple
        ``(names, input_names, function)``, where ``function`` maps the inputs in
        ``input_names`` order to the stoichiometries in ``names`` order.
    """
    cached = model._evaluator
    if cached is not None:
        baked, fingerprint, names, input_names, function = cached
        if _parameter_fingerprint(parameter_values, baked) == fingerprint:
            return names, input_names, function

    names = sorted(model.variables)
    nodes = _unique_nodes(model.variables[name] for name in names)
    input_names = sorted(
        {s.name for s in nodes if isinstance(s, pybamm.InputParameter)}
    )
    # Only the parameters the graph substitutes can invalidate it. The capacities
    # arrive as InputParameter, so a caller ageing a cell does not rebuild anything.
    baked = frozenset(
        s.name
        for s in nodes
        if isinstance(s, pybamm.Parameter | pybamm.FunctionParameter)
    )
    symbols = {name: casadi.MX.sym(name) for name in input_names}
    time = casadi.MX.sym("t")
    state = casadi.MX.sym("y", 1)
    # one conversion cache: the variables are views of a single shared graph
    converted: dict = {}
    expressions = [
        parameter_values.process_symbol(model.variables[name]).to_casadi(
            time, state, inputs=symbols, casadi_symbols=converted
        )
        for name in names
    ]
    function = casadi.Function(
        "electrode_soh_composite",
        list(symbols.values()),
        [casadi.vertcat(*expressions)],
    )
    model._evaluator = (
        baked,
        _parameter_fingerprint(parameter_values, baked),
        names,
        input_names,
        function,
    )
    return names, input_names, function


def get_initial_stoichiometries_composite(
    initial_value: float | str,
    parameter_values: pybamm.ParameterValues,
    direction: str | None = None,
    param: pybamm.LithiumIonParameters | None = None,
    options: dict | None = None,
    tol: float = 1e-6,
    inputs: dict | None = None,
    known_value: str = "cyclable lithium capacity",
    try_split_solve: bool = True,
    esoh_sim: pybamm.Simulation | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """
    Get the stoichiometries for composite electrodes from parameter values.

    Calculates stoichiometries at three states:
    - 100% SOC (x_100, y_100): Equilibrium state (direction=None)
    - 0% SOC (x_0, y_0): Equilibrium state (direction=None)
    - Initial SOC (x_init, y_init): Dynamic state (uses specified direction)

    For electrode models with OCP hysteresis, the equilibrium stoichiometries use
    the charging OCP branch at 100% SOC and the discharging OCP branch at 0% SOC.
    For models without hysteresis there is only one OCP curve, so the direction
    does not affect the equilibrium stoichiometries.

    Parameters
    ----------
    initial_value : float or str
        Target initial value.
        If float between 0 and 1, interpreted as SOC.
        If string ending in 'V' (e.g., "4 V"), interpreted as voltage,
        must be between V_min and V_max.
    parameter_values : pybamm.ParameterValues
        Parameter values for the simulation
    direction : str, optional
        The OCV branch to use for initial stoichiometries. Can be "charge" or
        "discharge". Only affects x_init/y_init, not equilibrium values.
    param : pybamm.LithiumIonParameters, optional
        Parameter object
    options : dict, optional
        Model options
    tol : float, optional
        The tolerance for the solver used to compute the initial stoichiometries.
        A lower value results in higher precision but may increase computation time.
        Default is 1e-6.
    inputs : dict, optional
        A dictionary of input parameters passed to the model.
    known_value : str, optional
        The known value needed to complete the electrode SOH model.
        Can be "cyclable lithium capacity".
    try_split_solve : bool, optional
        Whether to use the split solve method to improve robustness. Default is True.
        When True, if the full solve fails:
        1. Run split solve to get approximate stoichiometries
        2. Use these as initial conditions and retry full solve
        3. If retry succeeds, return full solve results
        4. If retry fails, raise error

    Returns
    -------
    dict
        Dictionary of stoichiometry values for all phases at 0%, 100%, and initial SOC
    """
    inputs = inputs or {}
    param = param or pybamm.LithiumIonParameters(options)

    if known_value != "cyclable lithium capacity":
        raise ValueError(
            "Only `cyclable lithium capacity` is supported for composite electrodes"
        )

    # A value that is neither an SOC nor a voltage is the caller's mistake, not a
    # solve that failed, so it must not reach the fallback.
    _initialization_method(initial_value)

    try:
        return ElectrodeSOHComposite.solve_full(
            initial_value,
            parameter_values,
            direction=direction,
            param=param,
            options=options,
            tol=tol,
            inputs=inputs,
            esoh_sim=esoh_sim,
        )
    except (pybamm.SolverError, ValueError, RuntimeError) as first_error:
        if not try_split_solve:
            raise pybamm.SolverError(
                f"Failed to solve composite electrode SOH: {first_error}"
            ) from first_error
        # The split solve reaches the answer a different way, one electrode at a
        # time, so its result is the fallback rather than a guess to re-solve from.
        try:
            return ElectrodeSOHComposite.solve_split(
                initial_value,
                parameter_values,
                direction=direction,
                param=param,
                options=options,
                tol=tol,
                inputs=inputs,
            )
        except (pybamm.SolverError, ValueError, RuntimeError) as split_error:
            raise pybamm.SolverError(
                f"Failed to solve composite electrode SOH. "
                f"Full solve error: {first_error}. "
                f"Split solve error: {split_error}"
            ) from split_error
