"""Measurement lanes for the Rust-vs-CasADi observability suite.

Each lane runs the same scenario across the backend matrix, samples the phase
timings, and pairs every candidate against the CasADi baseline so a regression
shows up as a comparison status rather than as a raw number a reader has to judge.
Backend order is shuffled per scenario, because a fixed order lets machine warm-up
flatter whichever backend runs last.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import random
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

import pybamm
from benchmarks.rust_observability.registry import (
    DEFAULT_OUTPUT_POINTS,
    INFERENCE_INPUTS,
    INFERENCE_SPREADS,
    ArtifactScenario,
    SolverScenario,
    get_solver_scenarios,
    inference_nominal_values,
)

# The row every candidate's raw difference is reported against.
_BASELINE_CASE = ("casadi_idaklu", False)

# Two backends at one tolerance differ by their mutual error, not their accuracy;
# same-integrator rows also cancel the error they share. Hence a converged oracle.
_REFERENCE_BACKEND = "casadi_idaklu"
DEFAULT_REFERENCE_TOLERANCE = 1e-10

# Tolerance bounds one step's local error; global error accumulates it over
# thousands. Measured: casadi_idaklu itself lands ~18 tolerance units out on states.
_REFERENCE_ACCURACY_HEADROOM = 100.0

# A reference must stay clear of what it judges; two decades holds its own error
# to ~1% of the candidate's.
_MIN_REFERENCE_DECADES = 2

_BACKEND_CASES = (
    _BASELINE_CASE,
    ("casadi_idaklu", True),
    ("casadi_idaklu_aot", False),
    ("casadi_idaklu_aot", True),
    ("rust_idaklu", False),
    ("rust_idaklu", True),
    ("rust_diffsol", False),
    ("rust_diffsol", True),
)


def backend_cases(include_aot: bool) -> tuple[tuple[str, bool], ...]:
    """The candidate backend matrix, with the AOT rows optional."""
    return tuple(
        case for case in _BACKEND_CASES if include_aot or case[0] != "casadi_idaklu_aot"
    )


def resolve_reference_tolerance(scenarios, tolerance: float | None) -> float | None:
    """Validate the converged reference's tolerance, or ``None`` when disabled.

    Parameters
    ----------
    scenarios : list of SolverScenario
        The scenarios the reference will be solved for.
    tolerance : float or None
        Requested reference tolerance; ``0`` or ``None`` disables the reference
        and reverts every lane to comparing against the baseline backend.

    Returns
    -------
    float or None
        The tolerance to solve the reference at, or ``None`` when disabled.

    Raises
    ------
    ValueError
        If the tolerance is negative, or sits within
        :data:`_MIN_REFERENCE_DECADES` of the loosest scenario it would judge --
        a reference no better than the candidate measures the gap between two
        approximations, which is the artifact the reference exists to remove.
    """
    if not tolerance:
        return None
    if tolerance < 0.0:
        raise ValueError("reference_tolerance must be non-negative")
    loosest = min(
        (min(scenario.atol, scenario.rtol) for scenario in scenarios), default=1.0
    )
    ceiling = loosest * 10.0**-_MIN_REFERENCE_DECADES
    if tolerance > ceiling:
        raise ValueError(
            f"reference_tolerance {tolerance:g} is not at least "
            f"{_MIN_REFERENCE_DECADES} decades tighter than the scenario "
            f"tolerance {loosest:g}"
        )
    return tolerance


def _reference_ladder(scenario, tolerance: float) -> list[float]:
    """Reference tolerances to try, loosening a decade at a time.

    A converged solve is not reachable everywhere -- DFN under a ramping current
    fails IDA's error test at ``t = 0`` below 1e-9 -- and the loosest reference
    that still clears the scenario by :data:`_MIN_REFERENCE_DECADES` is worth
    more than no reference at all.
    """
    ceiling = min(scenario.atol, scenario.rtol) * 10.0**-_MIN_REFERENCE_DECADES
    rungs = math.floor(math.log10(ceiling / tolerance) + 1e-9)
    # Rebuilt from the exponent rather than multiplied up, so a decade stays a decade.
    exponent = math.log10(tolerance)
    return [10.0 ** (exponent + step) for step in range(max(int(rungs), 0) + 1)]


def _resolve_reference(scenario, tolerance: float | None, solve, *, what: str):
    """Run ``solve(attempt)`` at the tightest reference tolerance that converges.

    Returns the tolerance used and the solve's value, or ``(None, None)`` when
    nothing in the ladder converges, so the caller falls back to the baseline.
    """
    if tolerance is None:
        return None, None
    failure = None
    for attempt in _reference_ladder(scenario, tolerance):
        try:
            return attempt, solve(attempt)
        except Exception as exc:  # noqa: BLE001
            failure = exc
    pybamm.logger.warning(
        f"{scenario.name}/{scenario.protocol}: no converged {what} from "
        f"{tolerance:g} upwards ({_short_reason(failure)}); comparing against "
        "the baseline backend instead."
    )
    return None, None


def _reference_tolerances(scenario) -> tuple[float, float]:
    """Allowance for judging a value against the converged reference."""
    return (
        _REFERENCE_ACCURACY_HEADROOM * scenario.atol,
        _REFERENCE_ACCURACY_HEADROOM * scenario.rtol,
    )


def _reference_solve(
    scenario,
    tolerance: float,
    *,
    inputs: dict[str, float] | None = None,
    parameter_values=None,
    sensitivity_parameters: list[str] | None = None,
):
    """One converged solve of ``scenario``, the answer every row is judged against.

    Always full state and never ``output_variables``-restricted, so one solve
    serves the state, output and sensitivity comparisons alike.
    """
    solver = _make_solver(
        _REFERENCE_BACKEND, atol=tolerance, rtol=tolerance, output_variables=None
    )
    simulation = _build_simulation(
        scenario, solver, _REFERENCE_BACKEND, parameter_values=parameter_values
    )
    _build_and_time(simulation, scenario, inputs=inputs)
    extra: dict = {}
    if inputs is not None:
        extra["inputs"] = inputs
    if sensitivity_parameters:
        extra["calculate_sensitivities"] = list(sensitivity_parameters)
    return simulation.solve(**_solve_kwargs(scenario, extra))


def _worst_comparison(summaries) -> ComparisonSummary | None:
    """The summary that came closest to, or furthest past, its own tolerance."""
    present = [summary for summary in summaries if summary is not None]
    if not present:
        return None
    return max(present, key=lambda summary: summary.max_normalized_error)


@dataclass(frozen=True)
class PhaseTiming:
    """Milliseconds attributed to each phase of one measured run."""

    build_ms: float = 0.0
    prepare_ms: float = 0.0
    cold_startup_ms: float = 0.0
    run_ms: float = 0.0
    set_up_ms: float = 0.0
    warm_set_up_ms: float = 0.0
    solve_ms: float = 0.0
    wall_solve_ms: float = 0.0
    integration_ms: float = 0.0
    observe_ms: float = 0.0
    e2e_ms: float = 0.0


@dataclass(frozen=True)
class ComparisonSummary:
    """Worst-case agreement between a candidate backend and the baseline.

    ``max_normalized_error`` is the absolute difference over the tolerance the
    status is judged against, so it crosses 1.0 exactly when the status flips.
    Rank comparisons by it rather than by ``max_abs_diff``, whose admissible
    size varies with the baseline magnitude.
    """

    max_abs_diff: float
    max_rel_diff: float
    max_normalized_error: float

    @property
    def status(self) -> str:
        """``"pass"`` while the normalised error is inside the tolerance."""
        return "pass" if self.max_normalized_error <= 1.0 else "warn"


# Shapes that cannot be lined up at all: reported as an infinite miss rather than
# raised, so one unsupported backend does not abandon the rest of the suite.
_UNCOMPARABLE = ComparisonSummary(
    max_abs_diff=float("inf"),
    max_rel_diff=float("inf"),
    max_normalized_error=float("inf"),
)


@dataclass(frozen=True)
class TimingSamples:
    """Per-repeat samples behind the reported medians, kept for spread checks."""

    warm_set_up_ms: tuple[float, ...] = ()
    solve_ms: tuple[float, ...] = ()
    wall_solve_ms: tuple[float, ...] = ()
    integration_ms: tuple[float, ...] = ()
    observe_ms: tuple[float, ...] = ()
    e2e_ms: tuple[float, ...] = ()


@dataclass(frozen=True)
class AotProfile:
    """What CasADi's ahead-of-time path cost, split by cold and warm disk cache."""

    fresh_cache_statuses: tuple[str, ...]
    disk_cache_statuses: tuple[str, ...]
    codegen_ms: float
    compiler_ms: float
    fresh_load_ms: float
    disk_load_ms: float
    fresh_total_ms: float
    disk_total_ms: float
    disk_prepare_ms: float
    disk_cold_startup_ms: float
    library_size_bytes: int
    verified: bool


@dataclass(frozen=True)
class TrajectorySummary:
    """How far two backends agreed on the time grid they actually returned.

    Adaptive steppers stop at different points, so timings are only comparable
    where the grids overlap; ``coverage`` is that overlap.
    """

    baseline_points: int
    candidate_points: int
    common_points: int
    coverage: float
    max_time_diff: float
    final_time_diff: float
    baseline_termination: str
    candidate_termination: str
    status: str


@dataclass(frozen=True)
class JacobianTelemetry:
    """Compile-time Jacobian metrics read back from the Rust model."""

    strategy: str
    n_colors: int
    nnz: int
    n_dense_rows: int
    dense_row_entries: int
    dense_row_tape_instructions: int
    split_eval_primal_instructions: int | None
    split_eval_total_instructions: int | None
    split_eval_raw_instructions: int | None
    branch_block_lens: tuple[int, ...]


@dataclass(frozen=True)
class ArtifactResult:
    """One artifact operation timed on baseline, candidate and the AOT backend."""

    scenario: str
    operation: str
    baseline_backend: str
    candidate_backend: str
    baseline_timings: PhaseTiming
    candidate_timings: PhaseTiming
    comparison: ComparisonSummary
    baseline_run_samples_ms: tuple[float, ...] = ()
    candidate_run_samples_ms: tuple[float, ...] = ()
    aot_run_samples_ms: tuple[float, ...] = ()
    # Third backend: AOT-compiled CasADi (same kernels as baseline, lowered to a
    # native shared lib). ``aot_comparison`` is AOT vs the CasADi VM baseline.
    aot_backend: str | None = None
    aot_timings: PhaseTiming | None = None
    aot_comparison: ComparisonSummary | None = None

    @property
    def status(self) -> str:
        """Agreement status of this operation against the baseline."""
        return self.comparison.status


def _comparison_status(supported: bool, summaries) -> str:
    """Worst status across the comparisons that ran.

    ``"unsupported"`` when the backend could not run the scenario at all and
    ``"baseline"`` when nothing was compared, so neither reads as a pass.
    """
    if not supported:
        return "unsupported"
    statuses = [summary.status for summary in summaries if summary is not None]
    if not statuses:
        return "baseline"
    return "pass" if all(status == "pass" for status in statuses) else "warn"


@dataclass(frozen=True)
class SolverResult:
    """One scenario solved on every backend, with agreement on states and outputs.

    The comparisons are against the converged reference when
    ``reference_tolerance`` is set, and against the baseline backend otherwise.
    ``baseline_delta`` carries the raw cross-backend difference either way.
    """

    scenario: str
    backend: str
    timings: PhaseTiming
    requested_output_points: int
    protocol: str = "cc_discharge"
    timing_samples: TimingSamples = TimingSamples()
    state_comparison: ComparisonSummary | None = None
    output_comparison: ComparisonSummary | None = None
    trajectory_comparison: TrajectorySummary | None = None
    reference_tolerance: float | None = None
    baseline_delta: ComparisonSummary | None = None
    jacobian_telemetry: JacobianTelemetry | None = None
    aot_profile: AotProfile | None = None
    supported: bool = True
    reason: str | None = None

    @property
    def status(self) -> str:
        """Worst status across the state, output and trajectory comparisons."""
        return _comparison_status(
            self.supported,
            (
                self.state_comparison,
                self.output_comparison,
                self.trajectory_comparison,
            ),
        )


@dataclass(frozen=True)
class SensitivityResult:
    """One scenario solved with forward sensitivities on every backend.

    Same comparison target convention as :class:`SolverResult`.
    """

    scenario: str
    backend: str
    timings: PhaseTiming
    requested_output_points: int
    protocol: str = "cc_discharge"
    timing_samples: TimingSamples = TimingSamples()
    # The parameters actually differentiated, which a protocol can narrow.
    sensitivity_parameters: tuple[str, ...] = ()
    state_sens_comparison: ComparisonSummary | None = None
    output_sens_comparison: ComparisonSummary | None = None
    trajectory_comparison: TrajectorySummary | None = None
    reference_tolerance: float | None = None
    baseline_delta: ComparisonSummary | None = None
    aot_profile: AotProfile | None = None
    supported: bool = True
    reason: str | None = None

    @property
    def status(self) -> str:
        """Worst status across the sensitivity and trajectory comparisons."""
        return _comparison_status(
            self.supported,
            (
                self.state_sens_comparison,
                self.output_sens_comparison,
                self.trajectory_comparison,
            ),
        )


@dataclass(frozen=True)
class InferenceResult:
    """One backend's cost per likelihood evaluation under changing inputs.

    Same comparison target convention as :class:`SolverResult`.
    """

    scenario: str
    protocol: str
    backend: str
    build_ms: float
    setup_ms: float
    aot_cache_status: str
    eval_samples_ms: tuple[float, ...]
    solve_samples_ms: tuple[float, ...]
    observe_samples_ms: tuple[float, ...]
    requested_output_points: int
    cold_observe_ms: float = 0.0
    output_comparison: ComparisonSummary | None = None
    sensitivity_comparison: ComparisonSummary | None = None
    trajectory_comparison: TrajectorySummary | None = None
    reference_tolerance: float | None = None
    baseline_delta: ComparisonSummary | None = None
    supported: bool = True
    reason: str | None = None

    @property
    def eval_median_ms(self) -> float:
        """Median wall-clock cost of one likelihood evaluation."""
        return float(np.median(self.eval_samples_ms))

    @property
    def eval_p10_ms(self) -> float:
        """10th-percentile evaluation cost, the fast end of the spread."""
        return float(np.percentile(self.eval_samples_ms, 10))

    @property
    def eval_p90_ms(self) -> float:
        """90th-percentile evaluation cost, the slow end of the spread."""
        return float(np.percentile(self.eval_samples_ms, 90))

    @property
    def solve_median_ms(self) -> float:
        """Median time inside ``Simulation.solve``."""
        return float(np.median(self.solve_samples_ms))

    @property
    def observe_median_ms(self) -> float:
        """Median time materialising the observed output."""
        return float(np.median(self.observe_samples_ms))

    @property
    def status(self) -> str:
        """Worst status across the value, gradient and trajectory comparisons."""
        return _comparison_status(
            self.supported,
            (
                self.output_comparison,
                self.sensitivity_comparison,
                self.trajectory_comparison,
            ),
        )


@dataclass
class _PreparedArtifactCase:
    baseline_backend: str
    candidate_backend: str
    baseline_timings: PhaseTiming
    candidate_timings: PhaseTiming
    # operation -> (baseline call, candidate call, aot call, materializer). Kernel
    # calls return native output (timed); the materializer densifies it (untimed).
    operations: dict[str, tuple[callable, callable, callable, callable]]
    atol: float
    rtol: float
    aot_backend: str = "casadi_aot"
    aot_timings: PhaseTiming = PhaseTiming()


def summarize_diff(
    baseline,
    candidate,
    *,
    atol: float,
    rtol: float,
) -> ComparisonSummary:
    """Compare two result arrays under the scenario's own tolerances.

    A shape mismatch is reported as infinite difference rather than raised, so one
    unsupported backend does not abandon the rest of the suite.
    """
    baseline_arr = _as_dense(baseline)
    candidate_arr = _as_dense(candidate)
    if baseline_arr.shape != candidate_arr.shape:
        return _UNCOMPARABLE
    abs_diff = np.abs(candidate_arr - baseline_arr)
    denom = np.maximum(np.abs(baseline_arr), atol)
    rel_diff = abs_diff / denom
    # The np.allclose tolerance, so the reported error and the status agree.
    normalized = abs_diff / (atol + rtol * np.abs(baseline_arr))
    max_normalized = float(normalized.max(initial=0.0))
    return ComparisonSummary(
        max_abs_diff=float(np.round(abs_diff.max(initial=0.0), 15)),
        max_rel_diff=float(np.round(rel_diff.max(initial=0.0), 15)),
        max_normalized_error=max_normalized,
    )


def run_artifact_lane(
    scenarios: list[ArtifactScenario],
    *,
    repeats: int,
    warmup: int,
) -> list[ArtifactResult]:
    """Time each scenario's compiled artifacts, one result per operation.

    Artifacts are prepared once per scenario and then called ``repeats`` times
    after ``warmup`` discarded calls, so what is measured is steady-state
    evaluation rather than compilation.
    """
    _validate_counts(repeats, warmup)
    results: list[ArtifactResult] = []
    for scenario in scenarios:
        prepared = _prepare_artifact_case(scenario)
        for operation in scenario.operations:
            (
                baseline_callable,
                candidate_callable,
                aot_callable,
                materialize,
            ) = prepared.operations[operation]
            baseline_run_ms, baseline_value, baseline_samples = _time_callable(
                baseline_callable, repeats=repeats, warmup=warmup
            )
            candidate_run_ms, candidate_value, candidate_samples = _time_callable(
                candidate_callable, repeats=repeats, warmup=warmup
            )
            aot_run_ms, aot_value, aot_samples = _time_callable(
                aot_callable, repeats=repeats, warmup=warmup
            )
            # Densify outside the timed region so run timings reflect kernel cost,
            # not format marshalling.
            comparison = summarize_diff(
                materialize(baseline_value),
                materialize(candidate_value),
                atol=prepared.atol,
                rtol=prepared.rtol,
            )
            aot_comparison = summarize_diff(
                materialize(baseline_value),
                materialize(aot_value),
                atol=prepared.atol,
                rtol=prepared.rtol,
            )
            results.append(
                ArtifactResult(
                    scenario=scenario.name,
                    operation=operation,
                    baseline_backend=prepared.baseline_backend,
                    candidate_backend=prepared.candidate_backend,
                    baseline_timings=replace(
                        prepared.baseline_timings,
                        run_ms=baseline_run_ms,
                    ),
                    candidate_timings=replace(
                        prepared.candidate_timings,
                        run_ms=candidate_run_ms,
                    ),
                    comparison=comparison,
                    baseline_run_samples_ms=baseline_samples,
                    candidate_run_samples_ms=candidate_samples,
                    aot_run_samples_ms=aot_samples,
                    aot_backend=prepared.aot_backend,
                    aot_timings=replace(prepared.aot_timings, run_ms=aot_run_ms),
                    aot_comparison=aot_comparison,
                )
            )
    return results


def _compare_solver_pair(
    target_solution,
    target_output,
    solution,
    output,
    scenario,
    *,
    output_only: bool,
    atol: float,
    rtol: float,
):
    """State, output and trajectory agreement of one solve against a target solve."""
    trajectory = summarize_trajectory(
        target_solution, solution, atol=scenario.atol, rtol=scenario.rtol
    )
    common_points = trajectory.common_points
    state = None
    if not output_only:
        target_y, candidate_y = _align_time_axis(
            target_solution.y, solution.y, common_points=common_points
        )
        state = summarize_diff(target_y, candidate_y, atol=atol, rtol=rtol)
    target_out, candidate_out = _align_time_axis(
        target_output, output, common_points=common_points
    )
    values = summarize_diff(target_out, candidate_out, atol=atol, rtol=rtol)
    return state, values, trajectory


def _solver_reference(scenario, tolerance: float | None):
    """The converged solve, its observed output, and the tolerance that produced them."""

    def solve(attempt):
        solution = _reference_solve(scenario, attempt)
        return solution, _extract_output(solution, scenario.observed_output)

    return _resolve_reference(scenario, tolerance, solve, what="reference")


def run_solver_lane(
    scenarios: list[SolverScenario],
    *,
    repeats: int,
    warmup: int,
    include_aot: bool = True,
    backend_order_seed: int = 0,
    reference_tolerance: float | None = DEFAULT_REFERENCE_TOLERANCE,
) -> list[SolverResult]:
    """Solve each scenario on every backend and compare states and outputs.

    Every row, the baseline included, is judged against one converged
    ``casadi_idaklu`` solve at ``reference_tolerance``; ``None`` falls back to
    comparing against the baseline backend, leaving that row ungated.

    ``backend_order_seed`` seeds the per-scenario backend shuffle; any fixed
    value gives a reproducible order. The baseline is shuffled in with the
    candidates and every comparison is computed after execution, so no backend
    is systematically measured on a colder machine.
    """
    _validate_counts(repeats, warmup)
    reference_tolerance = resolve_reference_tolerance(scenarios, reference_tolerance)
    results: list[SolverResult] = []
    for scenario in scenarios:
        measured = _execute_backend_cases(
            scenario,
            "solver",
            repeats=repeats,
            warmup=warmup,
            include_aot=include_aot,
            backend_order_seed=backend_order_seed,
        )
        _, baseline_solution, baseline_output = measured[_BASELINE_CASE]
        reference_used, reference = _solver_reference(scenario, reference_tolerance)
        reference_atol, reference_rtol = _reference_tolerances(scenario)

        for case, (result, solution, output) in measured.items():
            output_only = case[1]
            if not result.supported:
                results.append(result)
                continue
            if reference is None and case == _BASELINE_CASE:
                results.append(result)
                continue

            if reference is None:
                target, target_output = baseline_solution, baseline_output
                atol, rtol = scenario.atol, scenario.rtol
            else:
                target, target_output = reference
                atol, rtol = reference_atol, reference_rtol
            state_comparison, output_comparison, trajectory_comparison = (
                _compare_solver_pair(
                    target,
                    target_output,
                    solution,
                    output,
                    scenario,
                    output_only=output_only,
                    atol=atol,
                    rtol=rtol,
                )
            )
            baseline_delta = None
            if reference is not None and case != _BASELINE_CASE:
                baseline_state, baseline_values, _ = _compare_solver_pair(
                    baseline_solution,
                    baseline_output,
                    solution,
                    output,
                    scenario,
                    output_only=output_only,
                    atol=scenario.atol,
                    rtol=scenario.rtol,
                )
                baseline_delta = _worst_comparison((baseline_state, baseline_values))
            results.append(
                replace(
                    result,
                    state_comparison=state_comparison,
                    output_comparison=output_comparison,
                    trajectory_comparison=trajectory_comparison,
                    reference_tolerance=reference_used,
                    baseline_delta=baseline_delta,
                )
            )
    return results


def _compare_sensitivity_pair(
    target_solution,
    target_state,
    target_output,
    solution,
    state_sens,
    output_sens,
    scenario,
    *,
    output_only: bool,
    atol: float | None,
    rtol: float,
):
    """State- and output-sensitivity agreement of one solve against a target solve.

    ``atol`` of ``None`` takes each block's near-zero floor from that block's own
    peak instead: sensitivity magnitudes run from ``dV/dI`` at 1e-2 to ``dc/dp``
    at 1e4, so one shared absolute floor is either meaningless or dominant.
    """
    trajectory = summarize_trajectory(
        target_solution, solution, atol=scenario.atol, rtol=scenario.rtol
    )
    align = {
        "baseline_points": target_solution.t.size,
        "candidate_points": solution.t.size,
        "common_points": trajectory.common_points,
    }

    def compare(target, candidate):
        target_rows, candidate_rows = _align_rows(target, candidate, **align)
        floor = atol
        if floor is None:
            floor = rtol * float(np.abs(_as_dense(target_rows)).max(initial=0.0))
        return summarize_diff(target_rows, candidate_rows, atol=floor, rtol=rtol)

    state = None
    if not output_only and state_sens is not None and target_state is not None:
        state = compare(target_state, state_sens)
    return state, compare(target_output, output_sens), trajectory


def _sensitivity_reference(scenario, tolerance: float | None):
    """The converged sensitivity solve and its blocks, with the tolerance used."""
    parameter_values, inputs = _build_sensitivity_parameters(scenario)

    def solve(attempt):
        solution = _reference_solve(
            scenario,
            attempt,
            inputs=inputs,
            parameter_values=parameter_values,
            sensitivity_parameters=sorted(inputs),
        )
        state_sens, output_sens = _extract_sensitivities(
            solution, scenario.observed_output, output_only=False
        )
        return solution, state_sens, output_sens

    return _resolve_reference(scenario, tolerance, solve, what="sensitivity reference")


def run_sensitivity_lane(
    scenarios: list[SolverScenario],
    *,
    repeats: int,
    warmup: int,
    include_aot: bool = False,
    backend_order_seed: int = 0,
    reference_tolerance: float | None = DEFAULT_REFERENCE_TOLERANCE,
) -> list[SensitivityResult]:
    """Solve each scenario with forward sensitivities and compare them.

    Compares the state- and output-sensitivity blocks plus the trajectory, not the
    state and output values the solver lane checks, since a backend can integrate
    correctly and still get ``dy/dp`` wrong. Same converged-reference convention as
    :func:`run_solver_lane`, judged at the looser gradient tolerance because
    forward sensitivities are not error-controlled to the state tolerance.
    """
    _validate_counts(repeats, warmup)
    reference_tolerance = resolve_reference_tolerance(scenarios, reference_tolerance)
    results: list[SensitivityResult] = []
    for scenario in scenarios:
        measured = _execute_backend_cases(
            scenario,
            "sensitivity",
            repeats=repeats,
            warmup=warmup,
            include_aot=include_aot,
            backend_order_seed=backend_order_seed,
        )
        _, baseline_state, baseline_output, baseline_solution = measured[_BASELINE_CASE]
        reference_used, reference = _sensitivity_reference(
            scenario, reference_tolerance
        )
        _, gradient_rtol = _sensitivity_tolerances(scenario)

        for case, (result, state_sens, output_sens, solution) in measured.items():
            output_only = case[1]
            if not result.supported:
                results.append(result)
                continue
            if reference is None and case == _BASELINE_CASE:
                results.append(result)
                continue

            # A baseline fallback does not make gradients error-controlled.
            target = (
                (baseline_solution, baseline_state, baseline_output)
                if reference is None
                else reference
            )
            atol, rtol = None, gradient_rtol
            state_sens_comparison, output_sens_comparison, trajectory_comparison = (
                _compare_sensitivity_pair(
                    *target,
                    solution,
                    state_sens,
                    output_sens,
                    scenario,
                    output_only=output_only,
                    atol=atol,
                    rtol=rtol,
                )
            )
            baseline_delta = None
            if reference is not None and case != _BASELINE_CASE:
                delta_state, delta_output, _ = _compare_sensitivity_pair(
                    baseline_solution,
                    baseline_state,
                    baseline_output,
                    solution,
                    state_sens,
                    output_sens,
                    scenario,
                    output_only=output_only,
                    atol=scenario.atol,
                    rtol=scenario.rtol,
                )
                baseline_delta = _worst_comparison((delta_state, delta_output))
            results.append(
                replace(
                    result,
                    state_sens_comparison=state_sens_comparison,
                    output_sens_comparison=output_sens_comparison,
                    trajectory_comparison=trajectory_comparison,
                    reference_tolerance=reference_used,
                    baseline_delta=baseline_delta,
                )
            )
    return results


def _run_backend_case(
    lane: str,
    scenario: SolverScenario,
    *,
    backend: str,
    repeats: int,
    warmup: int,
    output_only: bool,
) -> tuple:
    """Measure one ``(backend, output_only)`` case on one scenario."""
    if backend == "casadi_idaklu_aot":
        return _run_profiled_aot_backend(
            lane,
            scenario,
            repeats=repeats,
            warmup=warmup,
            output_only=output_only,
            output_points=scenario.plan.requested_points or DEFAULT_OUTPUT_POINTS,
        )
    runner = _run_solver_backend if lane == "solver" else _run_sensitivity_backend
    return runner(
        scenario,
        backend=backend,
        repeats=repeats,
        warmup=warmup,
        output_only=output_only,
    )


def _unsupported_lane_result(
    lane: str, scenario: SolverScenario, backend: str, output_only: bool, reason: str
) -> tuple:
    """A fixed-arity failure record, so the comparison pass unpacks uniformly."""
    factory = SolverResult if lane == "solver" else SensitivityResult
    result = factory(
        scenario=scenario.name,
        protocol=scenario.protocol,
        backend=_solver_row_name(backend, output_only),
        timings=PhaseTiming(),
        requested_output_points=scenario.plan.requested_points,
        supported=False,
        reason=reason,
    )
    return (result, None, None) if lane == "solver" else (result, None, None, None)


def _execute_backend_cases(
    scenario: SolverScenario,
    lane: str,
    *,
    repeats: int,
    warmup: int,
    include_aot: bool,
    backend_order_seed: int,
) -> dict[tuple[str, bool], tuple]:
    """Measure every backend case for one scenario, in shuffled execution order.

    The baseline is shuffled in with the candidates rather than pinned first, so
    machine warm-up cannot systematically favour one backend. The returned
    mapping is re-keyed to start at the baseline, leaving the comparison pass
    independent of the order things ran in.
    """
    cases = _shuffled_backend_cases(
        backend_order_seed,
        f"{lane}:{scenario.name}:{scenario.protocol}",
        include_aot=include_aot,
    )
    measured: dict[tuple[str, bool], tuple] = {}
    for backend, output_only in cases:
        try:
            measured[(backend, output_only)] = _run_backend_case(
                lane,
                scenario,
                backend=backend,
                repeats=repeats,
                warmup=warmup,
                output_only=output_only,
            )
        except Exception as exc:
            # Only the baseline is load-bearing; anything else degrades to a
            # visible row rather than discarding a whole run's timings.
            if (backend, output_only) == _BASELINE_CASE:
                raise
            measured[(backend, output_only)] = _unsupported_lane_result(
                lane, scenario, backend, output_only, _short_reason(exc)
            )
    return {_BASELINE_CASE: measured.pop(_BASELINE_CASE), **measured}


def sample_input_vectors(
    nominal: dict[str, float],
    count: int,
    *,
    seed: int,
    spread: float | dict[str, float] = 0.2,
) -> list[dict[str, float]]:
    """Log-uniform input vectors within ``spread`` of ``nominal``.

    Drawn once per scenario and shared across backends, so repeat *i* uses the
    same inputs everywhere and cross-backend comparison stays exact.

    Parameters
    ----------
    nominal : dict of str to float
        Nominal value per input name.
    count : int
        Number of vectors to draw.
    seed : int
        Seed for the generator, making the sequence reproducible.
    spread : float or dict of str to float
        Fractional half-width of the sampling interval, either shared by every
        parameter or given per parameter. One width cannot suit parameters of
        different natures: what is routine for a diffusivity takes a bounded
        volume fraction somewhere its model cannot be solved.

    Returns
    -------
    list of dict
        One input dictionary per draw, in draw order.

    Raises
    ------
    KeyError
        If a per-parameter mapping omits a name in ``nominal``, so a new fitted
        parameter cannot silently inherit someone else's width.
    """
    rng = np.random.default_rng(seed)
    names = list(nominal)
    widths = np.array(
        [spread if isinstance(spread, float | int) else spread[name] for name in names],
        dtype=np.float64,
    )
    low = np.log(1.0 - widths)
    high = np.log(1.0 + widths)
    factors = np.exp(rng.uniform(low, high, size=(count, len(names))))
    return [
        {
            name: float(nominal[name] * factor)
            for name, factor in zip(names, row, strict=True)
        }
        for row in factors
    ]


def run_inference_lane(
    scenarios: list[SolverScenario],
    *,
    repeats: int,
    warmup: int,
    seed: int = 0,
    sensitivities: bool = False,
    include_aot: bool = False,
    backend_order_seed: int = 0,
    reference_tolerance: float | None = DEFAULT_REFERENCE_TOLERANCE,
) -> list[InferenceResult]:
    """Time one likelihood evaluation per repeat, with inputs changing each time.

    Unlike the solver lane, the fitted parameters stay symbolic and every timed
    repeat uses a different input vector, and observation goes through the
    interpolating call interface rather than the raw entries. Same
    converged-reference convention as :func:`run_solver_lane`, one reference
    solve per measured draw.
    """
    _validate_counts(repeats, warmup)
    reference_tolerance = resolve_reference_tolerance(scenarios, reference_tolerance)
    nominal = inference_nominal_values()
    results: list[InferenceResult] = []
    for scenario in scenarios:
        vectors = sample_input_vectors(
            nominal, warmup + repeats, seed=seed, spread=INFERENCE_SPREADS
        )
        cases = _shuffled_backend_cases(
            backend_order_seed,
            f"inference:{scenario.name}:{scenario.protocol}",
            include_aot=include_aot,
        )
        measured: dict[tuple[str, bool], tuple] = {}
        failed_scenario: str | None = None
        try:
            grid = _shared_observation_grid(scenario, vectors[0])
        except Exception as exc:  # noqa: BLE001
            failed_scenario = _short_reason(exc)

        for backend, output_only in cases:
            if failed_scenario is not None:
                measured[(backend, output_only)] = _unsupported_inference_case(
                    scenario, backend, output_only, failed_scenario
                )
                continue
            try:
                measured[(backend, output_only)] = _run_inference_backend(
                    scenario,
                    backend=backend,
                    output_only=output_only,
                    vectors=vectors,
                    warmup=warmup,
                    sensitivities=sensitivities,
                    grid=grid,
                )
            except Exception as exc:  # noqa: BLE001
                reason = _short_reason(exc)
                # Without a baseline there is nothing to compare against, so the
                # scenario degrades whole rather than aborting the suite.
                if (backend, output_only) == _BASELINE_CASE:
                    failed_scenario = reason
                    measured = {
                        case: _unsupported_inference_case(scenario, *case, reason)
                        for case in measured
                    }
                measured[(backend, output_only)] = _unsupported_inference_case(
                    scenario, backend, output_only, reason
                )

        # Re-keyed to start at the baseline, so the reported order is independent
        # of the shuffled execution order.
        measured = {_BASELINE_CASE: measured.pop(_BASELINE_CASE), **measured}
        _, baseline_repeats = measured[_BASELINE_CASE]
        reference_used, reference_repeats = (
            (None, None)
            if failed_scenario is not None
            else _inference_reference(
                scenario,
                vectors,
                warmup=warmup,
                grid=grid,
                sensitivities=sensitivities,
                tolerance=reference_tolerance,
            )
        )
        sensitivity_atol, sensitivity_rtol = _sensitivity_tolerances(scenario)
        reference_atol, reference_rtol = _reference_tolerances(scenario)
        for case, (result, candidate_repeats) in measured.items():
            if not result.supported:
                results.append(result)
                continue
            if reference_repeats is None and case == _BASELINE_CASE:
                results.append(result)
                continue

            if reference_repeats is None:
                target_repeats = baseline_repeats
                value_atol, value_rtol = scenario.atol, scenario.rtol
            else:
                target_repeats = reference_repeats
                value_atol, value_rtol = reference_atol, reference_rtol
            baseline_delta = None
            if reference_repeats is not None and case != _BASELINE_CASE:
                baseline_delta = _worst_comparison(
                    (
                        _worst_repeat_comparison(
                            baseline_repeats,
                            candidate_repeats,
                            select=_observed_values,
                            atol=scenario.atol,
                            rtol=scenario.rtol,
                        ),
                        _worst_repeat_comparison(
                            baseline_repeats,
                            candidate_repeats,
                            select=_observed_sensitivities,
                            atol=sensitivity_atol,
                            rtol=sensitivity_rtol,
                        ),
                    )
                )
            results.append(
                replace(
                    result,
                    output_comparison=_worst_repeat_comparison(
                        target_repeats,
                        candidate_repeats,
                        select=_observed_values,
                        atol=value_atol,
                        rtol=value_rtol,
                    ),
                    sensitivity_comparison=_worst_repeat_comparison(
                        target_repeats,
                        candidate_repeats,
                        select=_observed_sensitivities,
                        atol=sensitivity_atol,
                        rtol=sensitivity_rtol,
                    ),
                    trajectory_comparison=_worst_repeat_trajectory(
                        target_repeats, candidate_repeats, scenario
                    ),
                    reference_tolerance=reference_used,
                    baseline_delta=baseline_delta,
                )
            )
    return results


def _unsupported_inference_case(
    scenario, backend: str, output_only: bool, reason: str
) -> tuple:
    """An in-row failure paired with empty repeats, so the comparison pass unpacks
    uniformly and one bad backend never abandons the rest of the suite."""
    return (
        InferenceResult(
            scenario=scenario.name,
            protocol=scenario.protocol,
            backend=_solver_row_name(backend, output_only),
            build_ms=0.0,
            setup_ms=0.0,
            aot_cache_status="-",
            eval_samples_ms=(),
            solve_samples_ms=(),
            observe_samples_ms=(),
            requested_output_points=scenario.plan.requested_points,
            supported=False,
            reason=reason,
        ),
        (),
    )


@dataclass(frozen=True)
class RepeatObservation:
    """What one likelihood evaluation produced, for cross-backend comparison.

    ``times`` are the timestamps the values were read at, stopping at the
    solution's own end, so a candidate that terminated early carries fewer.
    Sensitivities come back on the solution's own grid rather than the
    observation grid, hence the second time axis.
    """

    values: np.ndarray
    times: np.ndarray
    sensitivities: np.ndarray | None
    sensitivity_times: np.ndarray | None
    final_time: float
    termination: str


def _observed_values(repeat: RepeatObservation):
    return repeat.values, repeat.times


def _observed_sensitivities(repeat: RepeatObservation):
    if repeat.sensitivities is None:
        return None
    return repeat.sensitivities, repeat.sensitivity_times


def _comparable_length(baseline_times, candidate_times, *, endpoint_gap: float) -> int:
    """Points that both repeats can be judged on.

    The shared prefix, minus any point inside the window where the two solutions'
    endpoints disagree: a moved event time leaves both trajectories racing to a
    cutoff at different moments, so points there measure the endpoint gap rather
    than trajectory agreement. Sizing the window from the measured gap keeps it at
    zero when the terminations coincide, and independent of ``--output-points``.
    """
    length = min(baseline_times.size, candidate_times.size)
    if length == 0 or endpoint_gap <= 0.0:
        return length
    shared_times = baseline_times[:length]
    cutoff = float(shared_times[-1]) - endpoint_gap
    return int(np.searchsorted(shared_times, cutoff, side="right"))


def _trim_to_points(values: np.ndarray, times: np.ndarray, points: int) -> np.ndarray:
    """Trim a time-outer block to ``points`` timepoints, whole rows at a time."""
    if times.size == 0 or values.shape[0] % times.size:
        return values
    return values[: points * (values.shape[0] // times.size)]


def _worst_repeat_comparison(
    baseline_repeats, candidate_repeats, *, select, atol: float, rtol: float
):
    """Worst agreement across every repeat, not just the last.

    Ranked by tolerance-normalised error, so a repeat that breached tolerance can
    never be masked by one whose absolute difference is larger but still allowed.
    Returns ``None`` when ``select`` yields nothing to compare.
    """
    summaries = []
    for baseline, candidate in zip(baseline_repeats, candidate_repeats, strict=True):
        selected = (select(baseline), select(candidate))
        if any(item is None for item in selected):
            return None
        (base_values, base_times), (cand_values, cand_times) = selected
        length = _comparable_length(
            base_times,
            cand_times,
            endpoint_gap=abs(baseline.final_time - candidate.final_time),
        )
        if length <= 0:
            summaries.append(_UNCOMPARABLE)
            continue
        summaries.append(
            summarize_diff(
                _trim_to_points(base_values, base_times, length),
                _trim_to_points(cand_values, cand_times, length),
                atol=atol,
                rtol=rtol,
            )
        )
    return _worst_comparison(summaries) or _UNCOMPARABLE


def _worst_repeat_trajectory(baseline_repeats, candidate_repeats, scenario):
    """Worst trajectory agreement across repeats: coverage, endpoint, termination.

    Both backends are read on the same grid, truncated at their own solution's
    end, so a shortfall in observed points *is* an early termination.
    """
    summaries = [
        _repeat_trajectory(baseline, candidate, scenario)
        for baseline, candidate in zip(baseline_repeats, candidate_repeats, strict=True)
    ]
    if not summaries:
        return None
    return min(
        summaries, key=lambda summary: (summary.status == "pass", summary.coverage)
    )


def _repeat_trajectory(
    baseline: RepeatObservation, candidate: RepeatObservation, scenario
) -> TrajectorySummary:
    """Coverage and termination of one inference repeat against another.

    Both read the same shared observation grid, truncated at each solution's
    own end, so the time axes agree exactly over their common prefix and only
    the lengths and the final time carry information.
    """
    summary = _summarize_time_axes(
        np.asarray(baseline.times, dtype=np.float64).reshape(-1),
        np.asarray(candidate.times, dtype=np.float64).reshape(-1),
        baseline.termination,
        candidate.termination,
        atol=scenario.atol,
        rtol=scenario.rtol,
    )
    return replace(
        summary, final_time_diff=abs(baseline.final_time - candidate.final_time)
    )


def _observation_grid(scenario) -> np.ndarray:
    """Experimental timestamps to observe at, offset off the solver's own grid.

    The first interval is skipped. It straddles the initial transient, which no
    two-point interpolant can represent from its endpoints alone -- a converged
    reference needs hundreds of internal steps inside it at low output-point
    counts. Reading there scores the output grid's resolution, not the backend:
    a Hermite interpolant given correct data lands further out than a linear
    chord, so the cell ranks backends by which interpolant they happen to use.
    """
    plan = scenario.plan
    if plan.t_interp is not None:
        # Midpoints, so observation genuinely interpolates rather than hitting nodes.
        return 0.5 * (plan.t_interp[1:-1] + plan.t_interp[2:])
    return np.array([], dtype=np.float64)


@contextmanager
def _aot_compile_events(backend: str):
    """Capture the AOT cache telemetry a solve emits, for AOT backends only."""
    if backend != "casadi_idaklu_aot":
        yield None
        return
    from pybamm.codegen.compilation import _capture_aot_compile_events

    with _capture_aot_compile_events() as events:
        yield events


def _summarize_cache_statuses(events) -> str:
    """Collapse per-kernel cache statuses into one cell.

    Reports what the compiler actually did (``miss``/``disk``/``memory``) so a
    cheap warm ``Setup`` is never misread as a fresh compile.
    """
    if not events:
        return "-"
    statuses = sorted({event.cache_status for event in events})
    return statuses[0] if len(statuses) == 1 else "+".join(statuses)


def _scaled_sensitivities(solution, name: str, *, inputs: dict[str, float]):
    """``p . d(var)/dp`` for every fitted parameter, one column each.

    The fitted parameters span eighteen orders of magnitude (a diffusivity at
    1e-14 beside a concentration at 1e4), so raw ``d/dp`` columns are not
    comparable under one tolerance. Scaling by the parameter gives the derivative
    with respect to a fractional change: dimensionless, the same size as the
    variable itself, and the quantity a log-space fitting loop actually consumes.
    Columns are taken by name rather than from the stacked ``"all"`` block so the
    scale factor cannot be paired with the wrong parameter.
    """
    sensitivities = solution[name].sensitivities
    return np.column_stack(
        [
            np.asarray(sensitivities[input_name], dtype=np.float64).reshape(-1)
            * inputs[input_name]
            for input_name in sorted(inputs)
        ]
    )


# A gate placed on the noise floor tracks it instead of discriminating against it.
_SENSITIVITY_NOISE_HEADROOM = 10.0


def _sensitivity_tolerances(scenario) -> tuple[float, float]:
    """Tolerances for judging a gradient, looser than for the state.

    A solver error-controls the state to ``(atol, rtol)``; the forward
    sensitivities integrated alongside it are not, and degrade to roughly the
    square root of it. Approaching an event the scaled gradient is also
    near-singular -- 2e3 against order 1 mid-run -- and the worst measured
    cross-backend agreement there is ~1e-3 relative, right at that square root.
    The gate therefore sits a decade above it: still three orders below the
    order-one relative miss a broken chain rule or an unseeded ``dy0/dp`` gives.
    """
    return (
        _SENSITIVITY_NOISE_HEADROOM * math.sqrt(scenario.atol),
        _SENSITIVITY_NOISE_HEADROOM * math.sqrt(scenario.rtol),
    )


def _shared_observation_grid(scenario, inputs: dict[str, float]) -> np.ndarray:
    """The timestamps every backend in this scenario is read at.

    Declared grids come straight from the protocol. A period-driven protocol has
    none, so one throwaway baseline solve establishes it; probing here rather than
    inside a measured run keeps the grid independent of the shuffled backend order.
    """
    declared = _observation_grid(scenario)
    if declared.size:
        return declared
    solver = _make_solver(
        "casadi_idaklu", atol=scenario.atol, rtol=scenario.rtol, output_variables=None
    )
    simulation = _build_simulation(scenario, solver, "casadi_idaklu")
    _build_and_time(simulation, scenario, inputs=inputs)
    probe = simulation.solve(**_solve_kwargs(scenario, {"inputs": inputs}))
    return np.asarray(probe.t, dtype=np.float64)


def _observe_inference(
    solution, inputs: dict[str, float], *, grid, scenario, sensitivities: bool
) -> RepeatObservation:
    """Read one solve on the shared grid, the way an inference loop would."""
    # The interpolating call interface, which is what an inference loop uses.
    times = grid[grid <= solution.t[-1]]
    values = np.asarray(
        solution[scenario.observed_output](times), dtype=np.float64
    ).reshape(-1)
    gradient = sensitivity_times = None
    if sensitivities:
        gradient = _scaled_sensitivities(
            solution, scenario.observed_output, inputs=inputs
        )
        sensitivity_times = np.asarray(solution.t, dtype=np.float64)
    return RepeatObservation(
        values=values,
        times=times,
        sensitivities=gradient,
        sensitivity_times=sensitivity_times,
        final_time=float(solution.t[-1]),
        termination=str(solution.termination),
    )


def _reference_solve_kwargs(scenario, extra: dict | None = None) -> dict:
    """Solve arguments for a reference, storing the solver's own steps.

    A reference restricted to the candidate's output grid carries that grid's
    interpolation error, and the gate would charge it to any candidate whose
    interpolant is *better* than the reference's -- ranking interpolants rather
    than checking correctness. Dropping ``t_interp`` stores every internal step,
    so an off-node read is judged against a trajectory dense enough to resolve it.
    """
    kwargs = _solve_kwargs(scenario, extra)
    if kwargs.get("t_interp") is not None:
        kwargs["t_interp"] = None
    return kwargs


def _inference_reference(
    scenario,
    vectors,
    *,
    warmup: int,
    grid,
    sensitivities: bool,
    tolerance: float | None,
):
    """One converged observation per measured draw, with the tolerance used.

    Solved outside every timed region, so the reference never lands in a sample.
    Built from ``vectors[0]`` like every backend is: the lane holds ``y0`` at the
    first draw, and a reference resolved from a different one starts the cell at
    a different state of charge. A draw that will not converge fails the whole
    ladder rung -- the comparison needs a reference for every draw, not most.
    """
    measured = list(vectors[warmup:])
    # The draw that failed the last rung decides the next one too, so trying it
    # first costs a doomed rung one converged solve instead of all of them.
    decides_the_rung = 0

    def solve(attempt):
        nonlocal decides_the_rung
        solver = _make_solver(
            _REFERENCE_BACKEND, atol=attempt, rtol=attempt, output_variables=None
        )
        simulation = _build_simulation(scenario, solver, _REFERENCE_BACKEND)
        _build_and_time(simulation, scenario, inputs=vectors[0])
        observations: dict[int, object] = {}
        order = sorted(range(len(measured)), key=lambda i: i != decides_the_rung)
        for index in order:
            inputs = measured[index]
            extra: dict = {"inputs": inputs}
            if sensitivities:
                extra["calculate_sensitivities"] = sorted(INFERENCE_INPUTS.values())
            try:
                solution = simulation.solve(**_reference_solve_kwargs(scenario, extra))
            except Exception:
                decides_the_rung = index
                raise
            observations[index] = _observe_inference(
                solution,
                inputs,
                grid=grid,
                scenario=scenario,
                sensitivities=sensitivities,
            )
        return [observations[index] for index in range(len(measured))]

    return _resolve_reference(scenario, tolerance, solve, what="reference")


def _run_inference_backend(
    scenario,
    *,
    backend: str,
    output_only: bool,
    vectors: list[dict[str, float]],
    warmup: int,
    sensitivities: bool,
    grid: np.ndarray,
):
    """Solve one scenario once per input vector and time each evaluation.

    ``grid`` is the shared observation grid, so every backend is read at the same
    timestamps. Cold observation is forced and timed before the warmup loop,
    keeping lazy variable compilation out of the per-evaluation samples whatever
    ``warmup`` is set to.
    """
    output_variables = [scenario.observed_output] if output_only else None
    solver = _make_solver(
        backend,
        atol=scenario.atol,
        rtol=scenario.rtol,
        output_variables=output_variables,
    )
    simulation = _build_simulation(scenario, solver, backend)
    build_ms = _build_and_time(simulation, scenario, inputs=vectors[0])

    def solve_extra(inputs):
        extra: dict = {"inputs": inputs}
        if sensitivities:
            extra["calculate_sensitivities"] = sorted(INFERENCE_INPUTS.values())
        return extra

    with _aot_compile_events(backend) as events:
        cold_solution = simulation.solve(
            **_solve_kwargs(scenario, solve_extra(vectors[0]))
        )
    cache_status = _summarize_cache_statuses(events)
    setup_ms = _time_to_ms(cold_solution.set_up_time)

    def observe(solution, inputs) -> tuple[float, RepeatObservation]:
        start = perf_counter()
        observation = _observe_inference(
            solution, inputs, grid=grid, scenario=scenario, sensitivities=sensitivities
        )
        return (perf_counter() - start) * 1000.0, observation

    cold_observe_ms, _ = observe(cold_solution, vectors[0])

    def evaluate(inputs):
        solve_start = perf_counter()
        solution = simulation.solve(**_solve_kwargs(scenario, solve_extra(inputs)))
        solve_ms = (perf_counter() - solve_start) * 1000.0
        observe_ms, observation = observe(solution, inputs)
        return solve_ms, observe_ms, observation

    for inputs in vectors[:warmup]:
        evaluate(inputs)

    solve_samples: list[float] = []
    observe_samples: list[float] = []
    eval_samples: list[float] = []
    observations: list[RepeatObservation] = []
    for inputs in vectors[warmup:]:
        eval_start = perf_counter()
        solve_ms, observe_ms, observation = evaluate(inputs)
        eval_samples.append((perf_counter() - eval_start) * 1000.0)
        solve_samples.append(solve_ms)
        observe_samples.append(observe_ms)
        observations.append(observation)

    result = InferenceResult(
        scenario=scenario.name,
        protocol=scenario.protocol,
        backend=_solver_row_name(backend, output_only),
        build_ms=build_ms,
        setup_ms=setup_ms,
        cold_observe_ms=cold_observe_ms,
        aot_cache_status=cache_status,
        eval_samples_ms=tuple(eval_samples),
        solve_samples_ms=tuple(solve_samples),
        observe_samples_ms=tuple(observe_samples),
        requested_output_points=scenario.plan.requested_points,
    )
    return result, observations


def _prepare_artifact_case(scenario: ArtifactScenario) -> _PreparedArtifactCase:
    if scenario.name == "toy_expr":
        return _prepare_toy_expr_case(scenario)
    if scenario.name in {"spm_residual", "spme_residual", "dfn_residual"}:
        model_factory = {
            "spm_residual": pybamm.lithium_ion.SPM,
            "spme_residual": pybamm.lithium_ion.SPMe,
            "dfn_residual": pybamm.lithium_ion.DFN,
        }[scenario.name]
        return _prepare_model_residual_case(scenario, model_factory)
    raise ValueError(f"Unsupported artifact scenario: {scenario.name}")


def _prepare_aot_kernels(functions, n_traj_cols):
    """AOT-compile the CasADi kernels to a native shared lib and return the
    externals, the mapped eval, and the (compile-dominated) prep time.

    Raises if compilation silently fell back to the VM so the AOT row can never
    be mislabelled as native code."""
    from pybamm.codegen.compilation import aot_compile

    start = perf_counter()
    cf, cjy, cjp, cjvp = aot_compile(list(functions))
    cf_map = cf.map(n_traj_cols)
    prepare_ms = (perf_counter() - start) * 1000.0
    for fn in (cf, cjy, cjp, cjvp):
        if fn.class_name() != "External":
            raise RuntimeError(
                "AOT compilation fell back to the CasADi VM (compiler missing "
                "or failed); refusing to report a mislabelled AOT row."
            )
    return cf, cjy, cjp, cjvp, cf_map, prepare_ms


def _prepare_toy_expr_case(scenario: ArtifactScenario) -> _PreparedArtifactCase:
    import casadi

    from pybamm.rust import ExprGraph

    build_start = perf_counter()
    expr, n_states, input_names = _toy_expr()
    t = 0.7
    y = np.array([0.3, 1.2])
    p = np.array([2.5, -0.8])
    v = np.array([0.6, -1.1])
    ts = np.linspace(0.0, 2.0, 100)
    y_traj = np.vstack([np.linspace(0.1, 1.0, 100), np.linspace(-0.5, 1.5, 100)])
    build_ms = (perf_counter() - build_start) * 1000.0

    baseline_prepare_start = perf_counter()
    t_sym = casadi.MX.sym("t")
    y_sym = casadi.MX.sym("y", n_states)
    y_dot_sym = casadi.MX.sym("y_dot", n_states)
    p_syms = {name: casadi.MX.sym(name) for name in input_names}
    cexpr = expr.to_casadi(
        t_sym,
        y_sym,
        y_dot_sym,
        p_syms,
        {"t": t_sym, "y": y_sym, "y_dot": y_dot_sym, "inputs": p_syms},
    )
    p_stacked = casadi.vertcat(*p_syms.values())
    v_sym = casadi.MX.sym("v", n_states)
    cf = casadi.Function("f", [t_sym, y_sym, p_stacked], [cexpr])
    cjy = casadi.Function(
        "jy",
        [t_sym, y_sym, p_stacked],
        [casadi.jacobian(cexpr, y_sym)],
    )
    cjp = casadi.Function(
        "jp", [t_sym, y_sym, p_stacked], [casadi.jacobian(cexpr, p_stacked)]
    )
    cjvp = casadi.Function(
        "jvp", [t_sym, y_sym, p_stacked, v_sym], [casadi.jtimes(cexpr, y_sym, v_sym)]
    )
    cf_map = cf.map(ts.size)
    ts_row, p_tiled = _trajectory_inputs(ts, p)
    baseline_prepare_ms = (perf_counter() - baseline_prepare_start) * 1000.0

    candidate_prepare_start = perf_counter()
    graph = ExprGraph()
    rust_expr = expr.to_rust(graph, {})
    rust_function = graph.compile(rust_expr, name="toy_expr", n_states=n_states)
    rust_jacobian_y = rust_function.jacobian()
    rust_jacobian_p = rust_function.jacobian(wrt="p")
    rust_function.jvp(t, y, p, v)
    candidate_prepare_ms = (perf_counter() - candidate_prepare_start) * 1000.0

    aot_cf, aot_cjy, aot_cjp, aot_cjvp, aot_cf_map, aot_prepare_ms = (
        _prepare_aot_kernels((cf, cjy, cjp, cjvp), ts.size)
    )

    operations = {
        "eval": (
            lambda: cf(t, y, p),
            lambda: rust_function(t, y, p),
            lambda: aot_cf(t, y, p),
            _as_vec,
        ),
        "jacobian_y": (
            lambda: cjy(t, y, p),
            lambda: rust_jacobian_y(t, y, p),
            lambda: aot_cjy(t, y, p),
            _as_dense,
        ),
        "jacobian_p": (
            lambda: cjp(t, y, p),
            lambda: rust_jacobian_p(t, y, p),
            lambda: aot_cjp(t, y, p),
            _as_dense,
        ),
        "jvp": (
            lambda: cjvp(t, y, p, v),
            lambda: rust_function.jvp(t, y, p, v),
            lambda: aot_cjvp(t, y, p, v),
            _as_vec,
        ),
        "eval_trajectory": (
            lambda: cf_map(ts_row, y_traj, p_tiled),
            lambda: rust_function.eval_trajectory(ts, y_traj, p),
            lambda: aot_cf_map(ts_row, y_traj, p_tiled),
            _as_dense,
        ),
    }
    return _PreparedArtifactCase(
        baseline_backend="casadi",
        candidate_backend="rust",
        baseline_timings=PhaseTiming(
            build_ms=build_ms,
            prepare_ms=baseline_prepare_ms,
        ),
        candidate_timings=PhaseTiming(
            build_ms=build_ms,
            prepare_ms=candidate_prepare_ms,
        ),
        operations=operations,
        atol=scenario.atol,
        rtol=scenario.rtol,
        aot_timings=PhaseTiming(build_ms=build_ms, prepare_ms=aot_prepare_ms),
    )


def _prepare_model_residual_case(
    scenario: ArtifactScenario,
    model_factory,
) -> _PreparedArtifactCase:
    import casadi

    from pybamm.rust import ExprGraph

    build_start = perf_counter()
    model = model_factory()
    model.events = []
    parameter_values = pybamm.ParameterValues("Chen2020")
    parameter_values["Current function [A]"] = pybamm.InputParameter("I")
    simulation = pybamm.Simulation(
        model,
        parameter_values=parameter_values,
        var_pts=_make_var_pts(model, 10),
    )
    simulation.build()
    built = simulation.built_model
    full_symbol = _full_residual_symbol(built)
    y = np.asarray(
        built.concatenated_initial_conditions.evaluate(), dtype=np.float64
    ).reshape(-1)
    p = np.array([0.5], dtype=np.float64)
    v = np.random.default_rng(0).standard_normal(y.size)
    ts = np.linspace(0.0, 10.0, 100)
    y_traj = np.tile(y[:, None], (1, ts.size)) * np.linspace(1.0, 1.01, ts.size)
    build_ms = (perf_counter() - build_start) * 1000.0

    baseline_prepare_start = perf_counter()
    n_states = built.len_rhs_and_alg
    t_sym = casadi.MX.sym("t")
    y_sym = casadi.MX.sym("y", n_states)
    y_dot_sym = casadi.MX.sym("y_dot", n_states)
    p_syms = {"I": casadi.MX.sym("I")}
    cexpr = full_symbol.to_casadi(
        t_sym,
        y_sym,
        y_dot_sym,
        p_syms,
        {"t": t_sym, "y": y_sym, "y_dot": y_dot_sym, "inputs": p_syms},
    )
    p_stacked = casadi.vertcat(*p_syms.values())
    v_sym = casadi.MX.sym("v", n_states)
    cf = casadi.Function("f", [t_sym, y_sym, p_stacked], [cexpr])
    cjy = casadi.Function(
        "jy",
        [t_sym, y_sym, p_stacked],
        [casadi.jacobian(cexpr, y_sym)],
    )
    cjp = casadi.Function(
        "jp", [t_sym, y_sym, p_stacked], [casadi.jacobian(cexpr, p_stacked)]
    )
    cjvp = casadi.Function(
        "jvp", [t_sym, y_sym, p_stacked, v_sym], [casadi.jtimes(cexpr, y_sym, v_sym)]
    )
    cf_map = cf.map(ts.size)
    ts_row, p_tiled = _trajectory_inputs(ts, p)
    baseline_prepare_ms = (perf_counter() - baseline_prepare_start) * 1000.0

    candidate_prepare_start = perf_counter()
    graph = ExprGraph()
    rust_expr = full_symbol.to_rust(graph, {})
    rust_function = graph.compile(rust_expr, name=scenario.name, n_states=n_states)
    rust_jacobian_y = rust_function.jacobian()
    rust_jacobian_p = rust_function.jacobian(wrt="p")
    rust_function.jvp(0.0, y, p, v)
    candidate_prepare_ms = (perf_counter() - candidate_prepare_start) * 1000.0

    aot_cf, aot_cjy, aot_cjp, aot_cjvp, aot_cf_map, aot_prepare_ms = (
        _prepare_aot_kernels((cf, cjy, cjp, cjvp), ts.size)
    )

    operations = {
        "eval": (
            lambda: cf(0.0, y, p),
            lambda: rust_function(0.0, y, p),
            lambda: aot_cf(0.0, y, p),
            _as_vec,
        ),
        "jacobian_y": (
            lambda: cjy(0.0, y, p),
            lambda: rust_jacobian_y(0.0, y, p),
            lambda: aot_cjy(0.0, y, p),
            _as_dense,
        ),
        "jacobian_p": (
            lambda: cjp(0.0, y, p),
            lambda: rust_jacobian_p(0.0, y, p),
            lambda: aot_cjp(0.0, y, p),
            _as_dense,
        ),
        "jvp": (
            lambda: cjvp(0.0, y, p, v),
            lambda: rust_function.jvp(0.0, y, p, v),
            lambda: aot_cjvp(0.0, y, p, v),
            _as_vec,
        ),
        "eval_trajectory": (
            lambda: cf_map(ts_row, y_traj, p_tiled),
            lambda: rust_function.eval_trajectory(ts, y_traj, p),
            lambda: aot_cf_map(ts_row, y_traj, p_tiled),
            _as_dense,
        ),
    }
    return _PreparedArtifactCase(
        baseline_backend="casadi",
        candidate_backend="rust",
        baseline_timings=PhaseTiming(
            build_ms=build_ms,
            prepare_ms=baseline_prepare_ms,
        ),
        candidate_timings=PhaseTiming(
            build_ms=build_ms,
            prepare_ms=candidate_prepare_ms,
        ),
        operations=operations,
        atol=scenario.atol,
        rtol=scenario.rtol,
        aot_timings=PhaseTiming(build_ms=build_ms, prepare_ms=aot_prepare_ms),
    )


def _toy_expr():
    y0 = pybamm.StateVector(slice(0, 1))
    y1 = pybamm.StateVector(slice(1, 2))
    a = pybamm.InputParameter("a")
    b = pybamm.InputParameter("b")
    expr = pybamm.NumpyConcatenation(
        a * y0 * y1 + pybamm.t,
        pybamm.sin(y0) * b + pybamm.exp(-y1),
    )
    return expr, 2, ("a", "b")


def _build_simulation(scenario, solver, backend, *, parameter_values=None):
    """Construct (but do not build) a simulation for one scenario."""
    model = scenario.model_factory(options=scenario.model_options)
    model.convert_to_format = _backend_convert_to_format(backend)
    if parameter_values is None:
        parameter_values = scenario.parameter_values_builder()
    kwargs = {"parameter_values": parameter_values, "solver": solver}
    if scenario.plan.experiment is not None:
        kwargs["experiment"] = scenario.plan.experiment
    return pybamm.Simulation(model, **kwargs)


def _build_and_time(simulation, scenario, inputs: dict | None = None) -> float:
    """Build the simulation once, outside the experiment path, and time it.

    Pre-building an experiment-attached simulation reparameterises the same
    model that ``Simulation.solve`` then parameterises again per step, which
    trips PyBaMM's reparameterised-model guard. The experiment path therefore
    builds lazily on its first ``solve`` call instead, contributing 0 ms here.

    Parameters
    ----------
    simulation : pybamm.Simulation
        The constructed, unbuilt simulation.
    scenario : SolverScenario
        Supplies the protocol's ``initial_soc`` and solve plan.
    inputs : dict, optional
        Values for any symbolic parameters. Required when ``initial_soc`` is
        set and the parameter set carries ``InputParameter``s, because mapping
        SOC to concentrations runs an ElectrodeSOH solve that must evaluate
        them. The initial state is fixed at these values for every repeat, so
        the warm path stays warm.
    """
    build_start = perf_counter()
    if scenario.plan.experiment is None:
        # initial_soc is applied here, once, never per solve: passing it to
        # Simulation.solve re-runs set_initial_state on every call.
        simulation.build(initial_soc=scenario.initial_soc, inputs=inputs)
    return (perf_counter() - build_start) * 1000.0


def _solve_kwargs(scenario, extra: dict | None = None) -> dict:
    """Time arguments for one protocol's solve, plus any caller extras.

    The experiment path takes its times from the experiment itself, so it
    contributes no time arguments at all.
    """
    kwargs: dict = {}
    plan = scenario.plan
    if plan.experiment is None:
        kwargs["t_eval"] = (
            [float(plan.t_interp[0]), float(plan.t_interp[-1])]
            if plan.t_eval is None
            else plan.t_eval
        )
        kwargs["t_interp"] = plan.t_interp
    kwargs.update(extra or {})
    return kwargs


def _run_solver_backend(
    scenario: SolverScenario,
    *,
    backend: str,
    repeats: int,
    warmup: int,
    output_only: bool,
):
    output_variables = [scenario.observed_output] if output_only else None
    solver = _make_solver(
        backend,
        atol=scenario.atol,
        rtol=scenario.rtol,
        output_variables=output_variables,
    )
    simulation = _build_simulation(scenario, solver, backend)

    cold_startup_start = perf_counter()
    build_ms = _build_and_time(simulation, scenario)

    solve_kwargs = _solve_kwargs(scenario)

    cold_solution = simulation.solve(**solve_kwargs)
    cold_observe_start = perf_counter()
    last_output = _extract_output(cold_solution, scenario.observed_output)
    cold_observe_ms = (perf_counter() - cold_observe_start) * 1000.0
    cold_startup_ms = (perf_counter() - cold_startup_start) * 1000.0

    for _ in range(warmup):
        warm_solution = simulation.solve(**solve_kwargs)
        _extract_output(warm_solution, scenario.observed_output)

    warm_set_up_samples = []
    solve_samples = []
    wall_solve_samples = []
    integration_samples = []
    observe_samples = []
    e2e_samples = []
    last_solution = cold_solution

    for _ in range(repeats):
        e2e_start = perf_counter()
        wall_solve_start = perf_counter()
        solution = simulation.solve(**solve_kwargs)
        wall_solve_samples.append((perf_counter() - wall_solve_start) * 1000.0)
        warm_set_up_samples.append(_time_to_ms(solution.set_up_time))
        solve_samples.append(_time_to_ms(solution.solve_time))
        integration_samples.append(_time_to_ms(solution.integration_time))
        observe_start = perf_counter()
        output = _extract_output(solution, scenario.observed_output)
        observe_samples.append((perf_counter() - observe_start) * 1000.0)
        e2e_samples.append((perf_counter() - e2e_start) * 1000.0)
        last_solution = solution
        last_output = output

    samples = TimingSamples(
        warm_set_up_ms=tuple(warm_set_up_samples),
        solve_ms=tuple(solve_samples),
        wall_solve_ms=tuple(wall_solve_samples),
        integration_ms=tuple(integration_samples),
        observe_ms=tuple(observe_samples),
        e2e_ms=tuple(e2e_samples),
    )
    result = SolverResult(
        scenario=scenario.name,
        protocol=scenario.protocol,
        backend=_solver_row_name(backend, output_only),
        timings=_summarize_timing_samples(
            samples,
            build_ms=build_ms,
            cold_set_up_ms=_time_to_ms(cold_solution.set_up_time),
            cold_observe_ms=cold_observe_ms,
            cold_startup_ms=cold_startup_ms,
        ),
        requested_output_points=scenario.plan.requested_points,
        timing_samples=samples,
        jacobian_telemetry=_get_jacobian_telemetry(solver, backend),
    )
    return result, last_solution, last_output


SENSITIVITY_INPUTS = {
    "Current function [A]": "I",
    "Positive electrode active material volume fraction": "eps_p",
}


def _build_sensitivity_parameters(scenario):
    """Swap the sensitivity parameters for input parameters on top of the protocol's values."""
    parameter_values = scenario.parameter_values_builder()
    inputs = {}
    for pybamm_name, input_name in SENSITIVITY_INPUTS.items():
        if (
            pybamm_name == "Current function [A]"
            and scenario.plan.experiment is not None
        ):
            # An Experiment's own control law supersedes "Current function [A]",
            # so it would be a dead, unreferenced sensitivity input.
            continue
        value = parameter_values[pybamm_name]
        if isinstance(value, pybamm.Symbol):
            # A protocol has made this parameter time-varying; it cannot be an input.
            continue
        inputs[input_name] = float(value)
        parameter_values[pybamm_name] = pybamm.InputParameter(input_name)
    return parameter_values, inputs


def _extract_sensitivities(solution, name: str, *, output_only: bool):
    """Materialize the stacked ``"all"`` sensitivity blocks.

    Output sensitivities ``d(var)/dp`` are always available (chain rule); full
    state sensitivities ``dy/dp`` only when the solve was not output-restricted.
    """
    output_sens = np.asarray(solution[name].sensitivities["all"], dtype=np.float64)
    if output_only:
        return None, output_sens
    state_sens = np.asarray(solution.sensitivities["all"], dtype=np.float64)
    return state_sens, output_sens


def _run_sensitivity_backend(
    scenario: SolverScenario,
    *,
    backend: str,
    repeats: int,
    warmup: int,
    output_only: bool,
):
    parameter_values, inputs = _build_sensitivity_parameters(scenario)
    output_variables = [scenario.observed_output] if output_only else None
    solver = _make_solver(
        backend,
        atol=scenario.atol,
        rtol=scenario.rtol,
        output_variables=output_variables,
    )
    simulation = _build_simulation(
        scenario, solver, backend, parameter_values=parameter_values
    )

    cold_startup_start = perf_counter()
    build_ms = _build_and_time(simulation, scenario, inputs=inputs)

    solve_kwargs = _solve_kwargs(
        scenario,
        {"inputs": inputs, "calculate_sensitivities": sorted(inputs)},
    )

    cold_solution = simulation.solve(**solve_kwargs)
    cold_observe_start = perf_counter()
    last_state_sens, last_output_sens = _extract_sensitivities(
        cold_solution, scenario.observed_output, output_only=output_only
    )
    cold_observe_ms = (perf_counter() - cold_observe_start) * 1000.0
    cold_startup_ms = (perf_counter() - cold_startup_start) * 1000.0

    for _ in range(warmup):
        warm_solution = simulation.solve(**solve_kwargs)
        _extract_sensitivities(
            warm_solution, scenario.observed_output, output_only=output_only
        )

    warm_set_up_samples = []
    solve_samples = []
    wall_solve_samples = []
    integration_samples = []
    observe_samples = []
    e2e_samples = []
    last_solution = cold_solution
    for _ in range(repeats):
        e2e_start = perf_counter()
        wall_solve_start = perf_counter()
        solution = simulation.solve(**solve_kwargs)
        wall_solve_samples.append((perf_counter() - wall_solve_start) * 1000.0)
        warm_set_up_samples.append(_time_to_ms(solution.set_up_time))
        solve_samples.append(_time_to_ms(solution.solve_time))
        integration_samples.append(_time_to_ms(solution.integration_time))
        observe_start = perf_counter()
        state_sens, output_sens = _extract_sensitivities(
            solution, scenario.observed_output, output_only=output_only
        )
        observe_samples.append((perf_counter() - observe_start) * 1000.0)
        e2e_samples.append((perf_counter() - e2e_start) * 1000.0)
        last_state_sens = state_sens
        last_output_sens = output_sens
        last_solution = solution

    samples = TimingSamples(
        warm_set_up_ms=tuple(warm_set_up_samples),
        solve_ms=tuple(solve_samples),
        wall_solve_ms=tuple(wall_solve_samples),
        integration_ms=tuple(integration_samples),
        observe_ms=tuple(observe_samples),
        e2e_ms=tuple(e2e_samples),
    )
    result = SensitivityResult(
        scenario=scenario.name,
        protocol=scenario.protocol,
        backend=_solver_row_name(backend, output_only),
        timings=_summarize_timing_samples(
            samples,
            build_ms=build_ms,
            cold_set_up_ms=_time_to_ms(cold_solution.set_up_time),
            cold_observe_ms=cold_observe_ms,
            cold_startup_ms=cold_startup_ms,
        ),
        requested_output_points=scenario.plan.requested_points,
        timing_samples=samples,
        sensitivity_parameters=tuple(sorted(inputs)),
    )
    return result, last_state_sens, last_output_sens, last_solution


@contextmanager
def _aot_cache_environment(cache_dir: str):
    previous = os.environ.get("PYBAMM_CASADI_AOT_CACHE")
    os.environ["PYBAMM_CASADI_AOT_CACHE"] = cache_dir
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PYBAMM_CASADI_AOT_CACHE", None)
        else:
            os.environ["PYBAMM_CASADI_AOT_CACHE"] = previous


def _run_aot_once(
    lane: str,
    scenario: SolverScenario,
    *,
    repeats: int,
    warmup: int,
    output_only: bool,
    cache_dir: str,
):
    from pybamm.codegen.compilation import _CACHE, _capture_aot_compile_events

    _CACHE.clear()
    try:
        with _aot_cache_environment(cache_dir), _capture_aot_compile_events() as events:
            if lane == "solver":
                values = _run_solver_backend(
                    scenario,
                    backend="casadi_idaklu_aot",
                    repeats=repeats,
                    warmup=warmup,
                    output_only=output_only,
                )
            else:
                values = _run_sensitivity_backend(
                    scenario,
                    backend="casadi_idaklu_aot",
                    repeats=repeats,
                    warmup=warmup,
                    output_only=output_only,
                )
        return values, tuple(events)
    finally:
        _CACHE.clear()


def _aot_worker_payload(
    lane, scenario, *, output_only, cache_dir, output_points
) -> tuple:
    """Name-only payload for the AOT disk worker.

    A ``SolverScenario`` pickles fine today, but its protocol builders are
    references to module-level functions, which would break silently if one
    ever became a closure or lambda. Names are simpler and immune to that.
    """
    return (
        lane,
        scenario.name,
        scenario.protocol,
        output_points,
        output_only,
        cache_dir,
    )


def _run_aot_disk_worker(payload):
    lane, model_name, protocol_name, output_points, output_only, cache_dir = payload
    scenario = get_solver_scenarios(
        [model_name], [protocol_name], output_points=output_points
    )[0]
    values, events = _run_aot_once(
        lane,
        scenario,
        repeats=1,
        warmup=0,
        output_only=output_only,
        cache_dir=cache_dir,
    )
    return values[0].timings, events


def _summarize_aot_profile(fresh_events, disk_events, disk_timings) -> AotProfile:
    fresh_statuses = tuple(event.cache_status for event in fresh_events)
    disk_statuses = tuple(event.cache_status for event in disk_events)
    fresh_keys = tuple(event.cache_key for event in fresh_events)
    disk_keys = tuple(event.cache_key for event in disk_events)
    verified = bool(fresh_events) and (
        all(status == "miss" for status in fresh_statuses)
        and all(status == "disk" for status in disk_statuses)
        and fresh_keys == disk_keys
    )
    profile = AotProfile(
        fresh_cache_statuses=fresh_statuses,
        disk_cache_statuses=disk_statuses,
        codegen_ms=sum(event.codegen_ms for event in fresh_events),
        compiler_ms=sum(event.compiler_ms for event in fresh_events),
        fresh_load_ms=sum(event.load_ms for event in fresh_events),
        disk_load_ms=sum(event.load_ms for event in disk_events),
        fresh_total_ms=sum(event.total_ms for event in fresh_events),
        disk_total_ms=sum(event.total_ms for event in disk_events),
        disk_prepare_ms=disk_timings.prepare_ms,
        disk_cold_startup_ms=disk_timings.cold_startup_ms,
        library_size_bytes=sum(event.library_size_bytes or 0 for event in fresh_events),
        verified=verified,
    )
    if not verified:
        raise RuntimeError(
            "AOT profile was not a verified cache miss followed by a fresh-process "
            f"disk hit: fresh={fresh_statuses}, disk={disk_statuses}"
        )
    return profile


def _run_profiled_aot_backend(
    lane: str,
    scenario: SolverScenario,
    *,
    repeats: int,
    warmup: int,
    output_only: bool,
    output_points: int,
):
    with tempfile.TemporaryDirectory(prefix="pybamm-aot-benchmark-") as cache_dir:
        fresh_values, fresh_events = _run_aot_once(
            lane,
            scenario,
            repeats=repeats,
            warmup=warmup,
            output_only=output_only,
            cache_dir=cache_dir,
        )
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=1) as pool:
            disk_timings, disk_events = pool.apply(
                _run_aot_disk_worker,
                (
                    _aot_worker_payload(
                        lane,
                        scenario,
                        output_only=output_only,
                        cache_dir=cache_dir,
                        output_points=output_points,
                    ),
                ),
            )
        profile = _summarize_aot_profile(
            fresh_events,
            disk_events,
            disk_timings,
        )
    result = replace(fresh_values[0], aot_profile=profile)
    return result, *fresh_values[1:]


def _backend_convert_to_format(backend: str) -> str:
    """Map a bench backend name to the ``model.convert_to_format`` that selects
    it."""
    if backend in ("casadi_idaklu", "casadi_idaklu_aot"):
        return "casadi"
    if backend in ("rust_idaklu", "rust_diffsol"):
        return "rust"
    raise ValueError(f"Unsupported solver backend: {backend}")


def _make_solver(backend: str, *, atol: float, rtol: float, output_variables=None):
    if backend == "casadi_idaklu_aot":
        return pybamm.IDAKLUSolver(
            atol=atol,
            rtol=rtol,
            output_variables=output_variables,
            options={"compile": True},
        )
    if backend in ("casadi_idaklu", "rust_idaklu"):
        return pybamm.IDAKLUSolver(
            atol=atol,
            rtol=rtol,
            output_variables=output_variables,
        )
    if backend == "rust_diffsol":
        return pybamm.DiffsolSolver(
            atol=atol,
            rtol=rtol,
            output_variables=output_variables,
        )
    raise ValueError(f"Unsupported solver backend: {backend}")


def _solver_row_name(backend: str, output_only: bool) -> str:
    return f"{backend}_out" if output_only else backend


def _short_reason(exc: Exception) -> str:
    reason = " ".join(str(exc).split())
    if not reason:
        reason = exc.__class__.__name__
    return f"{exc.__class__.__name__}: {reason}"


def _make_var_pts(model, npts: int) -> dict:
    var_pts = {}
    for key, value in model.default_var_pts.items():
        if isinstance(value, (int, float)) and value > 1 and key not in {"y", "z"}:
            var_pts[key] = npts
        else:
            var_pts[key] = value
    return var_pts


def _full_residual_symbol(built_model):
    if built_model.len_alg > 0:
        return pybamm.numpy_concatenation(
            built_model.concatenated_rhs,
            built_model.concatenated_algebraic,
        )
    return built_model.concatenated_rhs


def _extract_output(solution, name: str) -> np.ndarray:
    """Read the observed variable off the solver's own stored grid.

    Deliberately not the interpolating call interface: the solver lane measures
    what materialising the stored trajectory costs, and the inference lane
    measures interpolated reads. Between them both doors are covered.
    """
    return np.asarray(solution[name].data, dtype=np.float64)


def summarize_trajectory(
    baseline_solution,
    candidate_solution,
    *,
    atol: float,
    rtol: float,
) -> TrajectorySummary:
    """Compare time coverage and termination before numerical parity."""
    return _summarize_time_axes(
        np.asarray(baseline_solution.t, dtype=np.float64).reshape(-1),
        np.asarray(candidate_solution.t, dtype=np.float64).reshape(-1),
        str(baseline_solution.termination),
        str(candidate_solution.termination),
        atol=atol,
        rtol=rtol,
    )


def _summarize_time_axes(
    baseline_t: np.ndarray,
    candidate_t: np.ndarray,
    baseline_termination: str,
    candidate_termination: str,
    *,
    atol: float,
    rtol: float,
) -> TrajectorySummary:
    """The coverage and termination rules every lane classifies against.

    Shared so the solver and inference lanes cannot come to different verdicts
    about the same divergence.
    """
    baseline_points = baseline_t.size
    candidate_points = candidate_t.size
    minimum_points = min(baseline_points, candidate_points)
    span = max(
        abs(float(baseline_t[-1] - baseline_t[0])) if baseline_points else 0.0,
        abs(float(candidate_t[-1] - candidate_t[0])) if candidate_points else 0.0,
    )
    time_atol = max(atol, span * rtol)

    common_points = 0
    if minimum_points:
        matches = np.isclose(
            baseline_t[:minimum_points],
            candidate_t[:minimum_points],
            atol=time_atol,
            rtol=0.0,
        )
        mismatch = np.flatnonzero(~matches)
        common_points = int(mismatch[0]) if mismatch.size else minimum_points

    if common_points:
        max_time_diff = float(
            np.max(np.abs(baseline_t[:common_points] - candidate_t[:common_points]))
        )
    else:
        max_time_diff = float("inf")
    final_time_diff = (
        abs(float(baseline_t[-1] - candidate_t[-1]))
        if baseline_points and candidate_points
        else float("inf")
    )
    coverage = common_points / max(baseline_points, candidate_points, 1)
    only_terminal_sample_differs = (
        common_points >= minimum_points - 1
        and abs(baseline_points - candidate_points) <= 1
    )
    status = (
        "pass"
        if baseline_termination == candidate_termination
        and only_terminal_sample_differs
        and final_time_diff <= time_atol
        else "warn"
    )
    return TrajectorySummary(
        baseline_points=baseline_points,
        candidate_points=candidate_points,
        common_points=common_points,
        coverage=coverage,
        max_time_diff=max_time_diff,
        final_time_diff=final_time_diff,
        baseline_termination=baseline_termination,
        candidate_termination=candidate_termination,
        status=status,
    )


def _align_time_axis(baseline, candidate, *, common_points: int):
    """Trim time-last arrays to a trajectory-validated common prefix."""
    base = _as_dense(baseline)
    cand = _as_dense(candidate)
    return base[..., :common_points], cand[..., :common_points]


def _align_rows(
    baseline,
    candidate,
    *,
    baseline_points: int,
    candidate_points: int,
    common_points: int,
):
    """Trim fused time-outer sensitivity rows on timepoint boundaries."""
    base = _as_dense(baseline)
    cand = _as_dense(candidate)
    if baseline_points == 0 or candidate_points == 0:
        return base[:0], cand[:0]
    if base.shape[0] % baseline_points or cand.shape[0] % candidate_points:
        return base, cand
    base_rows_per_point = base.shape[0] // baseline_points
    candidate_rows_per_point = cand.shape[0] // candidate_points
    return (
        base[: common_points * base_rows_per_point],
        cand[: common_points * candidate_rows_per_point],
    )


def _get_jacobian_telemetry(solver, backend: str) -> JacobianTelemetry | None:
    if backend == "rust_idaklu":
        # ``_setup`` is only assigned inside set_up(); the experiment path solves
        # through a per-step copy of this solver, so it may never run here.
        model = getattr(solver, "_setup", {}).get("rust_model")
    elif backend == "rust_diffsol":
        model = getattr(solver, "_rust_model", None)
    else:
        return None
    if model is None:
        return None
    stats = model.jacobian_stats()
    return JacobianTelemetry(
        strategy=stats["strategy"],
        n_colors=int(stats["n_colors"]),
        nnz=int(stats["nnz"]),
        n_dense_rows=int(stats.get("n_dense_rows", 0)),
        dense_row_entries=int(stats.get("dense_row_entries", 0)),
        dense_row_tape_instructions=int(stats.get("dense_row_tape_instructions", 0)),
        split_eval_primal_instructions=stats["split_eval_primal_instructions"],
        split_eval_total_instructions=stats["split_eval_total_instructions"],
        split_eval_raw_instructions=stats.get("split_eval_raw_instructions"),
        branch_block_lens=tuple(stats.get("branch_block_lens", ())),
    )


def _summarize_timing_samples(
    samples: TimingSamples,
    *,
    build_ms: float,
    cold_set_up_ms: float,
    cold_observe_ms: float,
    cold_startup_ms: float,
) -> PhaseTiming:
    return PhaseTiming(
        build_ms=build_ms,
        prepare_ms=cold_set_up_ms + cold_observe_ms,
        cold_startup_ms=cold_startup_ms,
        set_up_ms=cold_set_up_ms,
        warm_set_up_ms=float(np.median(samples.warm_set_up_ms)),
        solve_ms=float(np.median(samples.solve_ms)),
        wall_solve_ms=float(np.median(samples.wall_solve_ms)),
        integration_ms=float(np.median(samples.integration_ms)),
        observe_ms=float(np.median(samples.observe_ms)),
        e2e_ms=float(np.median(samples.e2e_ms)),
    )


def _as_dense(value) -> np.ndarray:
    """Materialize a backend's native output (CasADi DM, Rust sparse, ndarray) as dense float64."""
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float64)


def _as_vec(value) -> np.ndarray:
    """Materialize a vector-valued output as a flat float64 array."""
    return _as_dense(value).reshape(-1)


def _trajectory_inputs(ts: np.ndarray, p: np.ndarray):
    """Shape the trajectory inputs for a CasADi mapped function call.

    ``Function.map(N)`` maps every input over N columns, so the scalar time
    becomes a (1, N) row and the constant parameter vector is tiled to (n_p, N).
    """
    ts_row = np.asarray(ts, dtype=np.float64).reshape(1, -1)
    p_tiled = np.tile(np.asarray(p, dtype=np.float64).reshape(-1, 1), (1, ts.size))
    return ts_row, p_tiled


def _time_callable(
    callable_, *, repeats: int, warmup: int, min_batch_seconds: float = 0.02
):
    """Median per-call time (ms) using auto-calibrated batching.

    A single perf_counter() span around one sub-microsecond call is dominated by
    timer resolution and Python dispatch. Instead we size an inner batch so each
    timed span exceeds ``min_batch_seconds``, then take the median per-call time
    across ``repeats`` such batches. ``repeats`` therefore counts batches, not
    individual calls.
    """
    for _ in range(warmup):
        callable_()

    batch = 1
    result = None
    while True:
        start = perf_counter()
        for _ in range(batch):
            result = callable_()
        elapsed = perf_counter() - start
        if elapsed >= min_batch_seconds or batch >= 1_000_000:
            break
        if elapsed <= 0.0:
            batch *= 8
        else:
            batch = max(batch * 2, int(batch * min_batch_seconds / elapsed) + 1)

    per_call_ms = [elapsed / batch * 1000.0]
    for _ in range(repeats - 1):
        start = perf_counter()
        for _ in range(batch):
            result = callable_()
        per_call_ms.append((perf_counter() - start) / batch * 1000.0)
    return float(np.median(per_call_ms)), result, tuple(per_call_ms)


def _time_to_ms(value) -> float:
    if value is None:
        return 0.0
    raw_value = value.value if hasattr(value, "value") else value
    return float(raw_value) * 1000.0


def _validate_counts(repeats: int, warmup: int) -> None:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")


def _shuffled_backend_cases(
    seed: int, key: str, *, include_aot: bool
) -> list[tuple[str, bool]]:
    cases = list(backend_cases(include_aot))
    random.Random(f"{seed}:{key}").shuffle(cases)
    return cases
