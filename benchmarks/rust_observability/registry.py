"""Scenario definitions for the Rust-vs-CasADi observability suite.

Scenarios are declared once here so the runners and the report agree on what was
measured; ``get_*_scenarios`` is the only way to obtain them, and it validates the
names a caller asked for.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

import pybamm

DEFAULT_OUTPUT_POINTS = 1000
DEFAULT_PROTOCOLS = ("cc_discharge",)

# Chen2020 gives both particle diffusivities as scalars, so the inference lane
# can swap them for inputs like for like.
BASE_PARAMETER_SET = "Chen2020"

CC_DURATION_S = 3600.0
CHARGE_C_RATE = 2.0
# 2C from empty tops out at ~1150 s; the margin keeps the event inside the grid
# as the inference lane perturbs the inputs.
CHARGE_DURATION_S = 1800.0
TRIANGLE_AMPLITUDE_A = 5.0
TRIANGLE_PERIOD_S = 600.0
TRIANGLE_DURATION_S = 1800.0
PULSE_AMPLITUDE_A = 5.0
PULSE_ON_S = 60.0
PULSE_REST_S = 120.0
PULSE_RAMP_S = 2.0
PULSE_DURATION_S = 1800.0
EXPERIMENT_PERIOD = "10 seconds"

# Every entry must be live on every model in the matrix. No maximum concentrations:
# initial concentration is absolute here, so fitting one moves the stoichiometry past 1.
INFERENCE_INPUTS = {
    "Negative particle diffusivity [m2.s-1]": "D_n",
    "Positive particle diffusivity [m2.s-1]": "D_p",
    "Negative electrode active material volume fraction": "eps_n",
    "Positive electrode active material volume fraction": "eps_p",
}

# Active material and pore volume sum to 1 in the base set, so each porosity has
# to track its electrode's fitted fraction or the geometry goes infeasible.
INFERENCE_COMPLEMENTS = {
    "Negative electrode porosity": "eps_n",
    "Positive electrode porosity": "eps_p",
}

# Sampling half-width per parameter: 20% on a volume fraction swings porosity
# 0.39 to 0.165 about a nominal 0.25, which stalls DFN and saturates SPMe on charge.
INFERENCE_SPREADS = {
    "D_n": 0.2,
    "D_p": 0.2,
    "eps_n": 0.05,
    "eps_p": 0.05,
}


@dataclass(frozen=True)
class ArtifactScenario:
    """One model whose compiled artifacts are timed, operation by operation."""

    name: str
    operations: tuple[str, ...]
    atol: float
    rtol: float


@dataclass(frozen=True)
class SolvePlan:
    """How one protocol issues its solve.

    Parameters
    ----------
    t_interp : numpy.ndarray or None
        Output grid. ``None`` when the protocol determines its own grid.
    t_eval : numpy.ndarray or None
        Breakpoints passed as ``t_eval``. ``None`` means ``[t0, tf]``.
    experiment : pybamm.Experiment or None
        When set, the solve goes through the experiment path.
    """

    t_interp: np.ndarray | None
    t_eval: np.ndarray | None
    experiment: Any | None = None

    @property
    def requested_points(self) -> int:
        """Output points asked for; ``0`` when the protocol decides the grid."""
        return 0 if self.t_interp is None else self.t_interp.size


@dataclass(frozen=True)
class Protocol:
    """One operating condition, independent of which model runs it."""

    build_parameter_values: Callable[[Any], Any]
    initial_soc: float | None
    build_plan: Callable[[int], SolvePlan]


def _base_parameter_values():
    """The shared base parameter set, at its own constant-current default."""
    return pybamm.ParameterValues(BASE_PARAMETER_SET)


def _grid_plan(duration: float) -> Callable[[int], SolvePlan]:
    """A plan builder issuing ``output_points`` samples evenly over ``duration``."""

    def build(output_points: int) -> SolvePlan:
        return SolvePlan(
            t_interp=np.linspace(0.0, duration, output_points), t_eval=None
        )

    return build


def _charge_parameter_values():
    parameter_values = _base_parameter_values()
    # 1C never reaches the voltage cutoff within the window; 2C terminates on it
    # for all three models.
    parameter_values["Current function [A]"] = -CHARGE_C_RATE * float(
        parameter_values["Nominal cell capacity [A.h]"]
    )
    return parameter_values


def _triangle_breakpoints() -> tuple[np.ndarray, np.ndarray]:
    """Vertices of the triangle wave, which linear interpolation reproduces exactly."""
    vertices = np.unique(
        np.concatenate(
            [
                [0.0],
                np.arange(
                    TRIANGLE_PERIOD_S / 4.0,
                    TRIANGLE_DURATION_S,
                    TRIANGLE_PERIOD_S / 2.0,
                ),
                [TRIANGLE_DURATION_S],
            ]
        )
    )
    values = (
        TRIANGLE_AMPLITUDE_A
        * (2.0 / np.pi)
        * np.arcsin(np.sin(2.0 * np.pi * vertices / TRIANGLE_PERIOD_S))
    )
    return vertices, values


def _pulse_breakpoints() -> tuple[np.ndarray, np.ndarray]:
    """Edges of a ramped pulse/rest train; finite ramps keep the profile Lipschitz."""
    times = [0.0]
    values = [0.0]
    start = 0.0
    period = PULSE_ON_S + PULSE_REST_S
    while start < PULSE_DURATION_S:
        times.extend(
            [
                start + PULSE_RAMP_S,
                start + PULSE_ON_S,
                start + PULSE_ON_S + PULSE_RAMP_S,
                start + period,
            ]
        )
        values.extend([PULSE_AMPLITUDE_A, PULSE_AMPLITUDE_A, 0.0, 0.0])
        start += period
    times_arr = np.asarray(times, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64)
    keep = times_arr <= PULSE_DURATION_S
    return times_arr[keep], values_arr[keep]


def _interpolant_parameter_values(breakpoints):
    times, values = breakpoints
    parameter_values = _base_parameter_values()
    parameter_values["Current function [A]"] = pybamm.Interpolant(
        times, values, pybamm.t, interpolator="linear"
    )
    return parameter_values


def _drive_cycle_parameter_values():
    return _interpolant_parameter_values(_triangle_breakpoints())


def _pulse_train_parameter_values():
    return _interpolant_parameter_values(_pulse_breakpoints())


def _interpolant_plan(breakpoints, duration: float, output_points: int) -> SolvePlan:
    times, _ = breakpoints
    grid = np.linspace(0.0, duration, output_points)
    # Every breakpoint must appear in t_eval or PyBaMM warns about resolution.
    return SolvePlan(t_interp=grid, t_eval=times)


def _drive_cycle_plan(output_points: int) -> SolvePlan:
    return _interpolant_plan(
        _triangle_breakpoints(), TRIANGLE_DURATION_S, output_points
    )


def _pulse_train_plan(output_points: int) -> SolvePlan:
    return _interpolant_plan(_pulse_breakpoints(), PULSE_DURATION_S, output_points)


def _experiment_plan(output_points: int) -> SolvePlan:
    del output_points  # The step period fixes the grid, identically on every backend.
    experiment = pybamm.Experiment(
        [
            "Discharge at 1C for 10 minutes",
            "Rest for 5 minutes",
            "Charge at 1C for 10 minutes",
            "Rest for 5 minutes",
        ],
        period=EXPERIMENT_PERIOD,
    )
    return SolvePlan(t_interp=None, t_eval=None, experiment=experiment)


_PROTOCOLS = {
    "cc_discharge": Protocol(
        build_parameter_values=_base_parameter_values,
        initial_soc=None,
        build_plan=_grid_plan(CC_DURATION_S),
    ),
    "cc_charge": Protocol(
        build_parameter_values=_charge_parameter_values,
        initial_soc=0.0,
        build_plan=_grid_plan(CHARGE_DURATION_S),
    ),
    "drive_cycle": Protocol(
        build_parameter_values=_drive_cycle_parameter_values,
        initial_soc=0.5,
        build_plan=_drive_cycle_plan,
    ),
    "pulse_train": Protocol(
        build_parameter_values=_pulse_train_parameter_values,
        initial_soc=0.5,
        build_plan=_pulse_train_plan,
    ),
    "experiment": Protocol(
        build_parameter_values=_base_parameter_values,
        initial_soc=None,
        build_plan=_experiment_plan,
    ),
}

_MODELS = {
    "SPM": pybamm.lithium_ion.SPM,
    "SPMe": pybamm.lithium_ion.SPMe,
    "DFN": pybamm.lithium_ion.DFN,
}


@dataclass(frozen=True)
class SolverScenario:
    """One model run under one protocol, plus the output variable to compare on."""

    name: str
    protocol: str
    model_factory: Callable[..., Any]
    model_options: dict[str, Any]
    parameter_values_builder: Callable[[], Any]
    initial_soc: float | None
    plan: SolvePlan
    observed_output: str
    atol: float
    rtol: float


def filter_names(available: list[str], selected: list[str] | None) -> list[str]:
    """Narrow ``available`` to ``selected``, keeping registry order.

    Raises
    ------
    ValueError
        If a requested name is not registered, so a typo fails the run instead of
        silently measuring less.
    """
    if not selected:
        return list(available)
    available_set = set(available)
    unknown = sorted(set(selected) - available_set)
    if unknown:
        raise ValueError(f"Unknown names requested: {', '.join(unknown)}")
    return [name for name in available if name in set(selected)]


_ARTIFACT_SCENARIOS = {
    "toy_expr": ArtifactScenario(
        name="toy_expr",
        operations=(
            "eval",
            "jacobian_y",
            "jacobian_p",
            "jvp",
            "eval_trajectory",
        ),
        atol=1e-12,
        rtol=1e-9,
    ),
    "spm_residual": ArtifactScenario(
        name="spm_residual",
        operations=(
            "eval",
            "jacobian_y",
            "jacobian_p",
            "jvp",
            "eval_trajectory",
        ),
        atol=1e-9,
        rtol=1e-7,
    ),
    "spme_residual": ArtifactScenario(
        name="spme_residual",
        operations=(
            "eval",
            "jacobian_y",
            "jacobian_p",
            "jvp",
            "eval_trajectory",
        ),
        atol=1e-9,
        rtol=1e-7,
    ),
    "dfn_residual": ArtifactScenario(
        name="dfn_residual",
        operations=(
            "eval",
            "jacobian_y",
            "jacobian_p",
            "jvp",
            "eval_trajectory",
        ),
        atol=1e-9,
        rtol=1e-7,
    ),
}


def get_artifact_scenarios(selected: list[str] | None = None) -> list[ArtifactScenario]:
    """The registered artifact scenarios, or just ``selected`` in registry order."""
    names = filter_names(list(_ARTIFACT_SCENARIOS), selected)
    return [_ARTIFACT_SCENARIOS[name] for name in names]


def get_protocol_names() -> list[str]:
    """Every registered protocol name, in registry order."""
    return list(_PROTOCOLS)


def _inference_parameter_values_for(protocol_builder):
    """Wrap a protocol's builder so the fitted parameters come back symbolic.

    Wrapping rather than replacing keeps the protocol's own control law, which
    is the whole reason a protocol row is worth timing.
    """

    def build():
        parameter_values = protocol_builder()
        for pybamm_name, input_name in INFERENCE_INPUTS.items():
            parameter_values[pybamm_name] = pybamm.InputParameter(input_name)
        for pybamm_name, input_name in INFERENCE_COMPLEMENTS.items():
            parameter_values[pybamm_name] = 1 - pybamm.InputParameter(input_name)
        return parameter_values

    return build


def inference_nominal_values() -> dict[str, float]:
    """Nominal value of each fitted parameter, read from the base set."""
    parameter_values = _base_parameter_values()
    return {
        input_name: float(parameter_values[pybamm_name])
        for pybamm_name, input_name in INFERENCE_INPUTS.items()
    }


def get_inference_scenarios(
    selected: list[str] | None = None,
    protocols: list[str] | None = None,
    *,
    output_points: int = DEFAULT_OUTPUT_POINTS,
) -> list[SolverScenario]:
    """Solver scenarios with the fitted parameters swapped for inputs."""
    return [
        replace(
            scenario,
            parameter_values_builder=_inference_parameter_values_for(
                scenario.parameter_values_builder
            ),
        )
        for scenario in get_solver_scenarios(
            selected, protocols, output_points=output_points
        )
    ]


def get_solver_scenarios(
    selected: list[str] | None = None,
    protocols: list[str] | None = None,
    *,
    output_points: int = DEFAULT_OUTPUT_POINTS,
) -> list[SolverScenario]:
    """The model × protocol cross product, model-major, in registry order.

    Parameters
    ----------
    selected : list of str, optional
        Model names; all registered models when omitted.
    protocols : list of str, optional
        Protocol names. ``None`` selects :data:`DEFAULT_PROTOCOLS`, keeping a
        bare run identical to the constant-current baseline; an empty list
        selects every protocol, as an empty ``selected`` does for models.
    output_points : int
        Output grid size for protocols that own their grid.

    Raises
    ------
    ValueError
        If ``output_points`` is below 2, or a model or protocol is unknown.
    """
    if output_points < 2:
        raise ValueError("output_points must be at least 2")
    model_names = filter_names(list(_MODELS), selected)
    protocol_names = filter_names(
        list(_PROTOCOLS),
        list(DEFAULT_PROTOCOLS) if protocols is None else protocols,
    )
    return [
        SolverScenario(
            name=model_name,
            protocol=protocol_name,
            model_factory=_MODELS[model_name],
            model_options={},
            parameter_values_builder=_PROTOCOLS[protocol_name].build_parameter_values,
            initial_soc=_PROTOCOLS[protocol_name].initial_soc,
            plan=_PROTOCOLS[protocol_name].build_plan(output_points),
            observed_output="Voltage [V]",
            atol=1e-6,
            rtol=1e-6,
        )
        for model_name in model_names
        for protocol_name in protocol_names
    ]
