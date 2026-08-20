"""Terminal tables and JSON export for the observability suite's results.

Rendering is kept out of the runners so a saved JSON run can be re-rendered, and
so table width can adapt to the terminal without touching measurement code.
"""

from __future__ import annotations

import math
import shutil
import textwrap
from dataclasses import asdict

# Cap the free-text Reason column so a verbose backend error (e.g. the rust
# output-sensitivity SolverError) can't blow the table line width out.
_MAX_REASON_WIDTH = 60
_DEFAULT_TABLE_WIDTH = 120
# Each compact validation row's fixed width: its own columns and the separating
# spaces, with the identity, status and reason columns excluded.
_SOLVER_VALIDATION_COLUMNS = 80
_SENSITIVITY_VALIDATION_COLUMNS = 74
_INFERENCE_VALIDATION_COLUMNS = 63
_BACKEND_COMPARISON_ORDER = {
    "casadi_idaklu": 0,
    "casadi_idaklu_aot": 1,
    "rust_idaklu": 2,
    "rust_diffsol": 3,
}


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _fits(table: str, width: int) -> bool:
    """Whether every line of a rendered table is inside ``width``.

    Measured rather than predicted from the column list, so adding a column
    narrows the layout instead of overflowing it.
    """
    return max(map(len, table.splitlines()), default=0) <= width


def _delta_text(comparison) -> str:
    """A comparison's worst absolute difference, or ``-`` when it was not run."""
    return "-" if comparison is None else f"{comparison.max_abs_diff:.2e}"


def _points_text(requested_output_points: int) -> str:
    """Requested output points, or ``-`` when the protocol owns its own grid."""
    return str(requested_output_points) if requested_output_points else "-"


def _reference_caption(results) -> str:
    """One line naming what this lane's Δ columns were measured against."""
    tolerances = sorted(
        {result.reference_tolerance for result in results if result.reference_tolerance}
    )
    if not tolerances:
        return (
            "Δ is the raw difference from casadi_idaklu at the scenario tolerance; "
            "no converged reference was run."
        )
    listed = ", ".join(f"{tolerance:g}" for tolerance in tolerances)
    caption = (
        f"Δ is the error against a converged casadi_idaklu reference at "
        f"atol=rtol={listed}. Base Δ is the raw difference from casadi_idaklu at "
        "the scenario tolerance, gated on nothing."
    )
    if any(
        result.supported and result.reference_tolerance is None for result in results
    ):
        caption += (
            " Some rows had no reference available; for those, Δ already holds "
            "that baseline difference and Base Δ is blank."
        )
    return caption


def _wrap_caption(caption: str, width: int) -> str:
    return "\n".join(textwrap.wrap(caption, width=max(width, 40)))


def _backend_comparison_key(backend: str) -> tuple[bool, int, str]:
    output_only = backend.endswith("_out")
    base_backend = backend.removesuffix("_out")
    rank = _BACKEND_COMPARISON_ORDER.get(base_backend, len(_BACKEND_COMPARISON_ORDER))
    return output_only, rank, backend


def _ordered_backend_results(results) -> list:
    results = list(results)
    scenario_order: dict[str, int] = {}
    protocol_order: dict[str, int] = {}
    for result in results:
        scenario_order.setdefault(result.scenario, len(scenario_order))
        protocol_order.setdefault(result.protocol, len(protocol_order))
    return sorted(
        results,
        key=lambda result: (
            scenario_order[result.scenario],
            protocol_order[result.protocol],
            *_backend_comparison_key(result.backend),
        ),
    )


def render_artifact_table(results) -> str:
    """Render the artifact lane as a fixed-width table."""
    # Run times are per-call kernel cost in microseconds; prep is one-time in ms.
    headers = (
        f"{'Scenario':<16} {'Operation':<16} {'Rust Prep':>10} {'Rust Run µs':>11} "
        f"{'CasADi Prep':>12} {'CasADi Run µs':>13} {'AOT Prep':>10} {'AOT Run µs':>11} "
        f"{'Rust Spd':>9} {'AOT Spd':>8} {'Rust Abs':>10} {'AOT Abs':>10} "
        f"{'Status':>8}"
    )
    lines = [headers, "-" * len(headers)]
    for result in results:
        rust_run = result.candidate_timings.run_ms
        casadi_run = result.baseline_timings.run_ms
        rust_spd = casadi_run / rust_run if rust_run > 0 else float("inf")
        aot_timings = result.aot_timings
        aot_comparison = result.aot_comparison
        aot_prep = aot_timings.prepare_ms if aot_timings else float("nan")
        aot_run = aot_timings.run_ms if aot_timings else float("nan")
        aot_spd = casadi_run / aot_run if aot_timings and aot_run > 0 else float("nan")
        aot_abs = aot_comparison.max_abs_diff if aot_comparison else float("nan")
        lines.append(
            f"{result.scenario:<16} {result.operation:<16} "
            f"{result.candidate_timings.prepare_ms:>10.3f} "
            f"{result.candidate_timings.run_ms * 1000.0:>11.3f} "
            f"{result.baseline_timings.prepare_ms:>12.3f} "
            f"{result.baseline_timings.run_ms * 1000.0:>13.3f} "
            f"{aot_prep:>10.3f} "
            f"{aot_run * 1000.0:>11.3f} "
            f"{rust_spd:>8.2f}x "
            f"{aot_spd:>7.2f}x "
            f"{result.comparison.max_abs_diff:>10.3e} "
            f"{aot_abs:>10.3e} "
            f"{result.status:>8}"
        )
    return "\n".join(lines)


def _solver_cells(result) -> dict[str, str]:
    state_abs = (
        result.state_comparison.max_abs_diff
        if result.state_comparison
        else float("nan")
    )
    output_abs = (
        result.output_comparison.max_abs_diff
        if result.output_comparison
        else float("nan")
    )
    trajectory = result.trajectory_comparison
    telemetry = result.jacobian_telemetry
    return {
        "scenario": result.scenario,
        "protocol": result.protocol,
        "backend": result.backend,
        "points": _points_text(result.requested_output_points),
        "build": f"{result.timings.build_ms:.2f}",
        "prepare": f"{result.timings.prepare_ms:.2f}",
        "cold_startup": f"{result.timings.cold_startup_ms:.2f}",
        "warm_set_up": f"{result.timings.warm_set_up_ms:.2f}",
        "solve": f"{result.timings.solve_ms:.2f}",
        "wall": f"{result.timings.wall_solve_ms:.2f}",
        "integration": f"{result.timings.integration_ms:.2f}",
        "observe": f"{result.timings.observe_ms:.2f}",
        "e2e": f"{result.timings.e2e_ms:.2f}",
        "coverage": f"{trajectory.coverage:.3f}" if trajectory else "-",
        "final_time_diff": (f"{trajectory.final_time_diff:.2e}" if trajectory else "-"),
        "colors": str(telemetry.n_colors) if telemetry else "-",
        "dense_rows": str(telemetry.n_dense_rows) if telemetry else "-",
        "dense_entries": str(telemetry.dense_row_entries) if telemetry else "-",
        "dense_tape": str(telemetry.dense_row_tape_instructions) if telemetry else "-",
        "state_abs": "-" if math.isnan(state_abs) else f"{state_abs:.2e}",
        "output_abs": "-" if math.isnan(output_abs) else f"{output_abs:.2e}",
        "base_delta": _delta_text(result.baseline_delta),
        "status": result.status,
        "reason": result.reason or "",
    }


def _identity_widths(
    rows: list[dict[str, str]], *, scenario_max: int, backend_max: int
) -> tuple[int, int, int]:
    scenario_width = min(
        scenario_max, max(len("Scenario"), *(len(row["scenario"]) for row in rows))
    )
    backend_width = min(
        backend_max, max(len("Backend"), *(len(row["backend"]) for row in rows))
    )
    status_width = min(12, max(len("Status"), *(len(row["status"]) for row in rows)))
    return scenario_width, backend_width, status_width


def _render_solver_wide(rows: list[dict[str, str]]) -> str:
    scenario_width, backend_width, status_width = _identity_widths(
        rows, scenario_max=18, backend_max=26
    )
    headers = (
        f"{'Scenario':<{scenario_width}} {'Backend':<{backend_width}} "
        f"{'Protocol':<13} "
        f"{'Pts':>5} {'Build':>8} {'Prep':>8} {'Cold':>8} "
        f"{'WarmSet':>8} {'Solve':>8} {'Wall':>8} {'Integr.':>8} "
        f"{'Obs':>4} {'E2E':>8} {'Cover':>7} {'T End Δ':>9} {'Clr':>4} "
        f"{'Dense':>5} {'DEntry':>6} {'Tape':>7} {'State Abs':>8} {'Output Abs':>8} "
        f"{'Base Δ':>9} {'Status':>{status_width}} Reason"
    )
    lines = [headers, "-" * len(headers)]
    for row in rows:
        reason = _truncate(row["reason"], _MAX_REASON_WIDTH)
        lines.append(
            (
                f"{_truncate(row['scenario'], scenario_width):<{scenario_width}} "
                f"{_truncate(row['backend'], backend_width):<{backend_width}} "
                f"{_truncate(row['protocol'], 13):<13} "
                f"{row['points']:>5} {row['build']:>8} {row['prepare']:>8} "
                f"{row['cold_startup']:>8} "
                f"{row['warm_set_up']:>8} {row['solve']:>8} {row['wall']:>8} "
                f"{row['integration']:>10} {row['observe']:>4} {row['e2e']:>8} "
                f"{row['coverage']:>7} {row['final_time_diff']:>9} "
                f"{row['colors']:>4} {row['dense_rows']:>5} "
                f"{row['dense_entries']:>6} {row['dense_tape']:>7} "
                f"{row['state_abs']:>8} {row['output_abs']:>8} "
                f"{row['base_delta']:>9} "
                f"{row['status']:>{status_width}} "
                f"{reason}"
            ).rstrip()
        )
    return "\n".join(lines)


def _compact_identity_widths(
    rows: list[dict[str, str]], width: int, validation_columns: int
) -> tuple[int, int, int]:
    """Identity widths that leave room for the timing *and* validation rows.

    ``validation_columns`` is the validation row's fixed width with the identity
    columns, the status and the reason excluded. Budgeting on the timing row
    alone let the validation row overflow whenever it was the wider of the two.
    """
    scenario_width, backend_width, status_width = _identity_widths(
        rows, scenario_max=18, backend_max=23
    )
    timing_fixed_width = max(77, 63 + status_width) + 14
    validation_fixed_width = validation_columns + status_width + len(" Reason")
    fixed_width = max(timing_fixed_width, validation_fixed_width)
    excess = scenario_width + backend_width + fixed_width - width
    backend_reduction = min(max(excess, 0), backend_width - 16)
    backend_width -= backend_reduction
    excess -= backend_reduction
    scenario_width -= min(max(excess, 0), scenario_width - 8)
    return scenario_width, backend_width, status_width


def _render_solver_compact(rows: list[dict[str, str]], width: int) -> str:
    scenario_width, backend_width, status_width = _compact_identity_widths(
        rows, width, _SOLVER_VALIDATION_COLUMNS
    )
    timing_header = (
        f"{'Scenario':<{scenario_width}} {'Backend':<{backend_width}} "
        f"{'Protocol':<13} {'Pts':>4} "
        f"{'Build':>7} {'Prep':>7} {'Cold':>7} {'Warm':>7} {'Solve':>7} "
        f"{'Wall':>7} {'Int':>7} {'Obs':>6} {'E2E':>7}"
    )
    lines = ["Timings (ms)", timing_header, "-" * len(timing_header)]
    for row in rows:
        lines.append(
            f"{_truncate(row['scenario'], scenario_width):<{scenario_width}} "
            f"{_truncate(row['backend'], backend_width):<{backend_width}} "
            f"{_truncate(row['protocol'], 13):<13} "
            f"{row['points']:>4} {row['build']:>7} {row['prepare']:>7} "
            f"{row['cold_startup']:>7} {row['warm_set_up']:>7} "
            f"{row['solve']:>7} {row['wall']:>7} "
            f"{row['integration']:>7} {row['observe']:>6} {row['e2e']:>7}"
        )

    validation_prefix = (
        f"{'Scenario':<{scenario_width}} {'Backend':<{backend_width}} "
        f"{'Protocol':<13} "
        f"{'Cover':>5} {'Δt':>8} {'Clr':>3} {'Rows':>4} {'Entry':>5} "
        f"{'Tape':>7} {'State Δ':>8} {'Output Δ':>8} {'Base Δ':>9} "
        f"{'Status':>{status_width}}"
    )
    reason_width = max(6, min(_MAX_REASON_WIDTH, width - len(validation_prefix) - 1))
    validation_header = f"{validation_prefix} Reason"
    lines.extend(["", "Validation", validation_header, "-" * len(validation_header)])
    for row in rows:
        reason = _truncate(row["reason"], reason_width)
        lines.append(
            (
                f"{_truncate(row['scenario'], scenario_width):<{scenario_width}} "
                f"{_truncate(row['backend'], backend_width):<{backend_width}} "
                f"{_truncate(row['protocol'], 13):<13} "
                f"{row['coverage']:>5} {row['final_time_diff']:>8} "
                f"{row['colors']:>3} {row['dense_rows']:>4} "
                f"{row['dense_entries']:>5} {row['dense_tape']:>7} "
                f"{row['state_abs']:>8} {row['output_abs']:>8} "
                f"{row['base_delta']:>9} "
                f"{row['status']:>{status_width}} {reason:<{reason_width}}"
            ).rstrip()
        )
    return "\n".join(lines)


def _render_solver_stacked(rows: list[dict[str, str]], width: int) -> str:
    lines = ["One block per backend; timings are milliseconds."]
    for row in rows:
        if len(lines) > 1:
            lines.append("")
        heading = (
            f"{row['scenario']} | {row['protocol']} | {row['backend']} | "
            f"{row['points']} pts | {row['status']}"
        )
        lines.append(_truncate(heading, width))
        details = (
            "Timing: "
            f"build {row['build']}, prepare {row['prepare']}, "
            f"cold {row['cold_startup']}, "
            f"warm {row['warm_set_up']}, solve {row['solve']}, wall {row['wall']}, "
            f"integration {row['integration']}, observe {row['observe']}, "
            f"e2e {row['e2e']}"
        )
        validation = (
            "Validation: "
            f"cover {row['coverage']}, Δt {row['final_time_diff']}, "
            f"colors {row['colors']}, dense rows/entries/sweeps "
            f"{row['dense_rows']}/{row['dense_entries']}/{row['dense_tape']}, "
            f"state Δ {row['state_abs']}, output Δ {row['output_abs']}, "
            f"base Δ {row['base_delta']}"
        )
        lines.extend(textwrap.wrap(details, width=width, subsequent_indent="  "))
        lines.extend(textwrap.wrap(validation, width=width, subsequent_indent="  "))
        if row["reason"]:
            lines.extend(
                textwrap.wrap(
                    f"Reason: {row['reason']}", width=width, subsequent_indent="  "
                )
            )
    return "\n".join(lines)


def _sensitivity_cells(result) -> dict[str, str]:
    state_abs = (
        result.state_sens_comparison.max_abs_diff
        if result.state_sens_comparison
        else float("nan")
    )
    output_abs = (
        result.output_sens_comparison.max_abs_diff
        if result.output_sens_comparison
        else float("nan")
    )
    trajectory = result.trajectory_comparison
    return {
        "scenario": result.scenario,
        "protocol": result.protocol,
        "backend": result.backend,
        "points": _points_text(result.requested_output_points),
        "build": f"{result.timings.build_ms:.2f}",
        "prepare": f"{result.timings.prepare_ms:.2f}",
        "cold_startup": f"{result.timings.cold_startup_ms:.2f}",
        "warm_set_up": f"{result.timings.warm_set_up_ms:.2f}",
        "solve": f"{result.timings.solve_ms:.2f}",
        "wall": f"{result.timings.wall_solve_ms:.2f}",
        "integration": f"{result.timings.integration_ms:.2f}",
        "observe": f"{result.timings.observe_ms:.2f}",
        "e2e": f"{result.timings.e2e_ms:.2f}",
        "coverage": f"{trajectory.coverage:.3f}" if trajectory else "-",
        "final_time_diff": (f"{trajectory.final_time_diff:.2e}" if trajectory else "-"),
        "sensitivity_parameters": ",".join(result.sensitivity_parameters) or "-",
        "state_abs": "-" if math.isnan(state_abs) else f"{state_abs:.2e}",
        "output_abs": "-" if math.isnan(output_abs) else f"{output_abs:.2e}",
        "base_delta": _delta_text(result.baseline_delta),
        "status": result.status,
        "reason": result.reason or "",
    }


def _render_sensitivity_wide(rows: list[dict[str, str]]) -> str:
    scenario_width, backend_width, status_width = _identity_widths(
        rows, scenario_max=18, backend_max=26
    )
    headers = (
        f"{'Scenario':<{scenario_width}} {'Backend':<{backend_width}} "
        f"{'Protocol':<13} "
        f"{'Pts':>5} {'Build':>8} {'Prep':>8} {'Cold':>8} "
        f"{'WarmSet':>8} {'Solve':>8} {'Wall':>8} {'Integr.':>10} "
        f"{'Obs':>8} {'E2E':>8} {'Cover':>7} {'T End Δ':>9} {'Params':<12} "
        f"{'State Sens':>11} {'Output Sens':>12} {'Base Δ':>9} "
        f"{'Status':>{status_width}} Reason"
    )
    lines = [headers, "-" * len(headers)]
    for row in rows:
        reason = _truncate(row["reason"], _MAX_REASON_WIDTH)
        lines.append(
            (
                f"{_truncate(row['scenario'], scenario_width):<{scenario_width}} "
                f"{_truncate(row['backend'], backend_width):<{backend_width}} "
                f"{_truncate(row['protocol'], 13):<13} "
                f"{row['points']:>5} {row['build']:>8} {row['prepare']:>8} "
                f"{row['cold_startup']:>8} {row['warm_set_up']:>8} "
                f"{row['solve']:>8} {row['wall']:>8} "
                f"{row['integration']:>10} {row['observe']:>8} {row['e2e']:>8} "
                f"{row['coverage']:>7} {row['final_time_diff']:>9} "
                f"{_truncate(row['sensitivity_parameters'], 12):<12} "
                f"{row['state_abs']:>11} {row['output_abs']:>12} "
                f"{row['base_delta']:>9} "
                f"{row['status']:>{status_width}} {reason}"
            ).rstrip()
        )
    return "\n".join(lines)


def _render_sensitivity_compact(rows: list[dict[str, str]], width: int) -> str:
    scenario_width, backend_width, status_width = _compact_identity_widths(
        rows, width, _SENSITIVITY_VALIDATION_COLUMNS
    )
    timing_header = (
        f"{'Scenario':<{scenario_width}} {'Backend':<{backend_width}} "
        f"{'Protocol':<13} {'Pts':>4} "
        f"{'Build':>7} {'Prep':>7} {'Cold':>7} {'Warm':>7} {'Solve':>7} "
        f"{'Wall':>7} {'Int':>7} {'Obs':>6} {'E2E':>7}"
    )
    lines = ["Timings (ms)", timing_header, "-" * len(timing_header)]
    for row in rows:
        lines.append(
            f"{_truncate(row['scenario'], scenario_width):<{scenario_width}} "
            f"{_truncate(row['backend'], backend_width):<{backend_width}} "
            f"{_truncate(row['protocol'], 13):<13} "
            f"{row['points']:>4} {row['build']:>7} {row['prepare']:>7} "
            f"{row['cold_startup']:>7} {row['warm_set_up']:>7} "
            f"{row['solve']:>7} {row['wall']:>7} {row['integration']:>7} "
            f"{row['observe']:>6} {row['e2e']:>7}"
        )

    validation_prefix = (
        f"{'Scenario':<{scenario_width}} {'Backend':<{backend_width}} "
        f"{'Protocol':<13} {'Params':<12} "
        f"{'Cover':>5} {'Δt':>8} {'State Δ':>9} {'Output Δ':>9} {'Base Δ':>9} "
        f"{'Status':>{status_width}}"
    )
    reason_width = max(6, min(_MAX_REASON_WIDTH, width - len(validation_prefix) - 1))
    validation_header = f"{validation_prefix} Reason"
    lines.extend(["", "Validation", validation_header, "-" * len(validation_header)])
    for row in rows:
        reason = _truncate(row["reason"], reason_width)
        lines.append(
            (
                f"{_truncate(row['scenario'], scenario_width):<{scenario_width}} "
                f"{_truncate(row['backend'], backend_width):<{backend_width}} "
                f"{_truncate(row['protocol'], 13):<13} "
                f"{_truncate(row['sensitivity_parameters'], 12):<12} "
                f"{row['coverage']:>5} {row['final_time_diff']:>8} "
                f"{row['state_abs']:>9} {row['output_abs']:>9} "
                f"{row['base_delta']:>9} "
                f"{row['status']:>{status_width}} {reason:<{reason_width}}"
            ).rstrip()
        )
    return "\n".join(lines)


def _render_sensitivity_stacked(rows: list[dict[str, str]], width: int) -> str:
    lines = ["One block per backend; timings are milliseconds."]
    for row in rows:
        if len(lines) > 1:
            lines.append("")
        heading = (
            f"{row['scenario']} | {row['protocol']} | {row['backend']} | "
            f"{row['points']} pts | {row['status']}"
        )
        lines.append(_truncate(heading, width))
        details = (
            "Timing: "
            f"build {row['build']}, prepare {row['prepare']}, "
            f"cold {row['cold_startup']}, warm {row['warm_set_up']}, "
            f"solve {row['solve']}, wall {row['wall']}, "
            f"integration {row['integration']}, observe {row['observe']}, "
            f"e2e {row['e2e']}"
        )
        validation = (
            "Validation: "
            f"cover {row['coverage']}, Δt {row['final_time_diff']}, "
            f"parameters {row['sensitivity_parameters']}, "
            f"state sensitivity Δ {row['state_abs']}, "
            f"output sensitivity Δ {row['output_abs']}, "
            f"base Δ {row['base_delta']}"
        )
        lines.extend(textwrap.wrap(details, width=width, subsequent_indent="  "))
        lines.extend(textwrap.wrap(validation, width=width, subsequent_indent="  "))
        if row["reason"]:
            lines.extend(
                textwrap.wrap(
                    f"Reason: {row['reason']}", width=width, subsequent_indent="  "
                )
            )
    return "\n".join(lines)


def _cache_status_text(statuses: tuple[str, ...]) -> str:
    if not statuses:
        return "-"
    if len(set(statuses)) == 1:
        suffix = f"x{len(statuses)}" if len(statuses) > 1 else ""
        return f"{statuses[0]}{suffix}"
    return "+".join(statuses)


def _render_aot_profile(results, width: int) -> str:
    profiled = [result for result in results if result.aot_profile is not None]
    if not profiled:
        return ""
    scenario_width = min(
        16, max(len("Scenario"), *(len(result.scenario) for result in profiled))
    )
    backend_width = min(
        23, max(len("Backend"), *(len(result.backend) for result in profiled))
    )
    fixed_width = 98
    excess = scenario_width + backend_width + fixed_width - width
    backend_reduction = min(max(excess, 0), backend_width - 16)
    backend_width -= backend_reduction
    excess -= backend_reduction
    scenario_width -= min(max(excess, 0), scenario_width - 8)

    title = "AOT profile (isolated cache; phase timings in ms)"
    if scenario_width + backend_width + fixed_width > width:
        lines = [title]
        for result in profiled:
            profile = result.aot_profile
            heading = f"{result.scenario} | {result.protocol} | {result.backend}"
            details = (
                f"fresh {_cache_status_text(profile.fresh_cache_statuses)}, "
                f"disk {_cache_status_text(profile.disk_cache_statuses)}, "
                f"codegen {profile.codegen_ms:.2f}, compiler {profile.compiler_ms:.2f}, "
                f"fresh load {profile.fresh_load_ms:.2f}, "
                f"disk load {profile.disk_load_ms:.2f}, "
                f"disk prep {profile.disk_prepare_ms:.2f}, "
                f"disk cold {profile.disk_cold_startup_ms:.2f}, "
                f"library {profile.library_size_bytes / 1024**2:.2f} MiB, "
                f"verified {'yes' if profile.verified else 'no'}"
            )
            lines.extend(["", _truncate(heading, width)])
            lines.extend(textwrap.wrap(details, width=width, subsequent_indent="  "))
        return "\n".join(lines)

    header = (
        f"{'Scenario':<{scenario_width}} {'Backend':<{backend_width}} "
        f"{'Protocol':<13} "
        f"{'Fresh':>7} {'Disk':>7} {'Gen':>8} {'Compiler':>9} "
        f"{'FLoad':>7} {'DLoad':>7} {'DPrep':>8} {'DCold':>8} "
        f"{'MiB':>6} {'Status':>6}"
    )
    lines = [title, header, "-" * len(header)]
    for result in profiled:
        profile = result.aot_profile
        lines.append(
            f"{_truncate(result.scenario, scenario_width):<{scenario_width}} "
            f"{_truncate(result.backend, backend_width):<{backend_width}} "
            f"{_truncate(result.protocol, 13):<13} "
            f"{_cache_status_text(profile.fresh_cache_statuses):>7} "
            f"{_cache_status_text(profile.disk_cache_statuses):>7} "
            f"{profile.codegen_ms:>8.2f} {profile.compiler_ms:>9.2f} "
            f"{profile.fresh_load_ms:>7.2f} {profile.disk_load_ms:>7.2f} "
            f"{profile.disk_prepare_ms:>8.2f} "
            f"{profile.disk_cold_startup_ms:>8.2f} "
            f"{profile.library_size_bytes / 1024**2:>6.2f} "
            f"{'pass' if profile.verified else 'fail':>6}"
        )
    return "\n".join(lines)


def _render_lane(
    results,
    width: int | None,
    *,
    cells,
    wide,
    compact,
    stacked,
    empty_message: str,
    title: str = "",
    with_aot: bool = True,
) -> str:
    """Render one lane, narrowing to a compact then a stacked layout as needed.

    ``width`` defaults to the detected terminal width. ``title`` labels the wide
    layout only; the narrower layouts carry their own headings.
    """
    ordered = _ordered_backend_results(results)
    rows = [cells(result) for result in ordered]
    if not rows:
        return empty_message
    table_width = (
        width or shutil.get_terminal_size(fallback=(_DEFAULT_TABLE_WIDTH, 24)).columns
    )
    wide_table = wide(rows)
    compact_table = compact(rows, table_width)
    if _fits(wide_table, table_width):
        main_table = f"{title}\n{wide_table}" if title else wide_table
    elif _fits(compact_table, table_width):
        main_table = compact_table
    else:
        main_table = stacked(rows, max(table_width, 40))
    tables = [_wrap_caption(_reference_caption(ordered), table_width), main_table]
    if with_aot and (aot_table := _render_aot_profile(ordered, max(table_width, 40))):
        tables.append(aot_table)
    return "\n\n".join(tables)


def render_solver_table(results, *, width: int | None = None) -> str:
    """Render the solver lane, narrowing to a compact layout if ``width`` is tight.

    ``width`` defaults to the detected terminal width.
    """
    return _render_lane(
        results,
        width,
        cells=_solver_cells,
        wide=_render_solver_wide,
        compact=_render_solver_compact,
        stacked=_render_solver_stacked,
        empty_message="No solver results.",
    )


def render_sensitivity_table(results, *, width: int | None = None) -> str:
    """Render the sensitivity lane, with the same width handling as the solver table.

    Solves differentiate the parameters in ``runners.SENSITIVITY_INPUTS``.
    """
    # State/Output Sens are max abs diffs of the stacked "all" sensitivity block
    # against the CasADi-IDAKLU baseline.
    return _render_lane(
        results,
        width,
        cells=_sensitivity_cells,
        wide=_render_sensitivity_wide,
        compact=_render_sensitivity_compact,
        stacked=_render_sensitivity_stacked,
        empty_message="No sensitivity results.",
    )


def _inference_cells(result) -> dict[str, str]:
    if result.eval_samples_ms:
        spread = f"{result.eval_p10_ms:.2f}-{result.eval_p90_ms:.2f}"
        eval_p50 = f"{result.eval_median_ms:.2f}"
        solve = f"{result.solve_median_ms:.2f}"
        observe = f"{result.observe_median_ms:.2f}"
    else:
        spread = eval_p50 = solve = observe = "-"
    trajectory = result.trajectory_comparison
    return {
        "scenario": result.scenario,
        "protocol": result.protocol,
        "backend": result.backend,
        "points": _points_text(result.requested_output_points),
        "build": f"{result.build_ms:.2f}",
        "setup": f"{result.setup_ms:.2f}",
        "cold_observe": f"{result.cold_observe_ms:.2f}",
        "aot": result.aot_cache_status,
        "eval_p50": eval_p50,
        "spread": spread,
        "solve": solve,
        "observe": observe,
        "coverage": f"{trajectory.coverage:.3f}" if trajectory else "-",
        "final_time_diff": (f"{trajectory.final_time_diff:.2e}" if trajectory else "-"),
        "output_abs": _delta_text(result.output_comparison),
        "sensitivity_abs": _delta_text(result.sensitivity_comparison),
        "base_delta": _delta_text(result.baseline_delta),
        "status": result.status,
        "reason": result.reason or "",
    }


def _render_inference_wide(rows: list[dict[str, str]]) -> str:
    scenario_width, backend_width, status_width = _identity_widths(
        rows, scenario_max=18, backend_max=20
    )
    header = (
        f"{'Scenario':<{scenario_width}} {'Backend':<{backend_width}} "
        f"{'Protocol':<13} {'Pts':>5} {'Build':>8} {'Setup':>8} {'ColdObs':>8} "
        f"{'AOT':>7} {'Eval p50':>9} {'p10-p90':>17} {'Solve':>8} {'Obs':>8} "
        f"{'Cover':>6} {'T End Δ':>9} {'Output Δ':>10} {'Sens Δ':>10} "
        f"{'Base Δ':>9} {'Status':>{status_width}} Reason"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            (
                f"{_truncate(row['scenario'], scenario_width):<{scenario_width}} "
                f"{_truncate(row['backend'], backend_width):<{backend_width}} "
                f"{_truncate(row['protocol'], 13):<13} "
                f"{row['points']:>5} {row['build']:>8} {row['setup']:>8} "
                f"{row['cold_observe']:>8} {row['aot']:>7} {row['eval_p50']:>9} "
                f"{row['spread']:>17} {row['solve']:>8} {row['observe']:>8} "
                f"{row['coverage']:>6} {row['final_time_diff']:>9} "
                f"{row['output_abs']:>10} {row['sensitivity_abs']:>10} "
                f"{row['base_delta']:>9} "
                f"{row['status']:>{status_width}} "
                f"{_truncate(row['reason'], _MAX_REASON_WIDTH)}"
            ).rstrip()
        )
    return "\n".join(lines)


def _render_inference_compact(rows: list[dict[str, str]], width: int) -> str:
    scenario_width, backend_width, status_width = _compact_identity_widths(
        rows, width, _INFERENCE_VALIDATION_COLUMNS
    )
    timing_header = (
        f"{'Scenario':<{scenario_width}} {'Backend':<{backend_width}} "
        f"{'Protocol':<13} {'Pts':>4} {'Build':>7} {'Setup':>7} {'ColdObs':>7} "
        f"{'AOT':>6} {'Eval p50':>8} {'p10-p90':>15} {'Solve':>7} {'Obs':>6}"
    )
    lines = ["Timings (ms)", timing_header, "-" * len(timing_header)]
    for row in rows:
        lines.append(
            f"{_truncate(row['scenario'], scenario_width):<{scenario_width}} "
            f"{_truncate(row['backend'], backend_width):<{backend_width}} "
            f"{_truncate(row['protocol'], 13):<13} "
            f"{row['points']:>4} {row['build']:>7} {row['setup']:>7} "
            f"{row['cold_observe']:>7} {row['aot']:>6} {row['eval_p50']:>8} "
            f"{row['spread']:>15} {row['solve']:>7} {row['observe']:>6}"
        )

    validation_prefix = (
        f"{'Scenario':<{scenario_width}} {'Backend':<{backend_width}} "
        f"{'Protocol':<13} {'Cover':>5} {'Δt':>8} {'Output Δ':>10} "
        f"{'Sens Δ':>10} {'Base Δ':>9} {'Status':>{status_width}}"
    )
    reason_width = max(6, min(_MAX_REASON_WIDTH, width - len(validation_prefix) - 1))
    validation_header = f"{validation_prefix} Reason"
    lines.extend(["", "Validation", validation_header, "-" * len(validation_header)])
    for row in rows:
        lines.append(
            (
                f"{_truncate(row['scenario'], scenario_width):<{scenario_width}} "
                f"{_truncate(row['backend'], backend_width):<{backend_width}} "
                f"{_truncate(row['protocol'], 13):<13} "
                f"{row['coverage']:>5} {row['final_time_diff']:>8} "
                f"{row['output_abs']:>10} {row['sensitivity_abs']:>10} "
                f"{row['base_delta']:>9} "
                f"{row['status']:>{status_width}} "
                f"{_truncate(row['reason'], reason_width):<{reason_width}}"
            ).rstrip()
        )
    return "\n".join(lines)


def _render_inference_stacked(rows: list[dict[str, str]], width: int) -> str:
    lines = ["One block per backend; timings are milliseconds."]
    for row in rows:
        if len(lines) > 1:
            lines.append("")
        heading = (
            f"{row['scenario']} | {row['protocol']} | {row['backend']} | "
            f"{row['points']} pts | {row['status']}"
        )
        lines.append(_truncate(heading, width))
        details = (
            "Timing: "
            f"build {row['build']}, setup {row['setup']}, "
            f"cold observe {row['cold_observe']}, aot {row['aot']}, "
            f"eval p50 {row['eval_p50']}, p10-p90 {row['spread']}, "
            f"solve {row['solve']}, observe {row['observe']}"
        )
        validation = (
            "Validation: "
            f"cover {row['coverage']}, Δt {row['final_time_diff']}, "
            f"output Δ {row['output_abs']}, sensitivity Δ {row['sensitivity_abs']}, "
            f"base Δ {row['base_delta']}"
        )
        lines.extend(textwrap.wrap(details, width=width, subsequent_indent="  "))
        lines.extend(textwrap.wrap(validation, width=width, subsequent_indent="  "))
        if row["reason"]:
            lines.extend(
                textwrap.wrap(
                    f"Reason: {row['reason']}", width=width, subsequent_indent="  "
                )
            )
    return "\n".join(lines)


def render_inference_table(results, *, width: int | None = None) -> str:
    """Render the inference lane: one-time costs beside per-evaluation costs.

    Same width handling as the solver and sensitivity lanes; ``width`` defaults
    to the detected terminal width.
    """
    return _render_lane(
        results,
        width,
        cells=_inference_cells,
        wide=_render_inference_wide,
        compact=_render_inference_compact,
        stacked=_render_inference_stacked,
        empty_message="No inference results.",
        title="Per-evaluation costs under changing inputs (ms)",
        with_aot=False,
    )


def suite_to_jsonable(lane: str, results, *, metadata: dict | None = None) -> dict:
    """Convert one lane's results to JSON-serialisable form for ``--json``.

    Solver and sensitivity results are emitted in backend-comparison order so a
    diff between two saved runs lines up row for row.
    """
    ordered_results = (
        _ordered_backend_results(results)
        if lane in {"solver", "sensitivity", "inference"}
        else list(results)
    )
    return {
        "lane": lane,
        "metadata": metadata or {},
        "results": [asdict(result) for result in ordered_results],
    }
