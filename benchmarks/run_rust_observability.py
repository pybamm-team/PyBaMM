"""Entry point for the Rust-vs-CasADi observability benchmark suite.

Runs one or more measurement lanes, prints a table per lane, and optionally writes
the same results as JSON. Runnable either as a module or as a script — the
``sys.path`` insert below covers the script case.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.rust_observability.registry import (
    DEFAULT_OUTPUT_POINTS,
    DEFAULT_PROTOCOLS,
    INFERENCE_INPUTS,
    get_artifact_scenarios,
    get_inference_scenarios,
    get_solver_scenarios,
)
from benchmarks.rust_observability.report import (
    render_artifact_table,
    render_inference_table,
    render_sensitivity_table,
    render_solver_table,
    suite_to_jsonable,
)
from benchmarks.rust_observability.runners import (
    DEFAULT_REFERENCE_TOLERANCE,
    SENSITIVITY_INPUTS,
    run_artifact_lane,
    run_inference_lane,
    run_sensitivity_lane,
    run_solver_lane,
)

_DEFAULT_REPEATS = 10
_DEFAULT_INFERENCE_REPEATS = 50


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the suite."""
    parser = argparse.ArgumentParser(
        description="Run the lightweight Rust-vs-CasADi observability suite."
    )
    parser.add_argument(
        "--lane",
        choices=("artifact", "solver", "sensitivity", "inference", "all"),
        default="all",
    )
    parser.add_argument("--artifact-scenarios", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument(
        "--protocols",
        nargs="*",
        default=list(DEFAULT_PROTOCOLS),
        help="Operating protocols to run (default: %(default)s).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help=(
            f"Timed repeats per case (default: {_DEFAULT_REPEATS}, "
            f"{_DEFAULT_INFERENCE_REPEATS} on the inference lane)."
        ),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--output-points",
        type=int,
        default=DEFAULT_OUTPUT_POINTS,
        help="Requested solver output points (default: %(default)s).",
    )
    parser.add_argument(
        "--backend-order-seed",
        type=int,
        default=0,
        help="Seed used to randomize candidate backend execution order.",
    )
    parser.add_argument(
        "--aot",
        choices=("solver", "all", "none"),
        default="solver",
        help="Which lanes run the CasADi AOT rows (default: %(default)s).",
    )
    parser.add_argument(
        "--reference-tolerance",
        type=float,
        default=DEFAULT_REFERENCE_TOLERANCE,
        help=(
            "Tolerance of the converged CasADi reference every row is judged "
            "against (default: %(default)s). 0 drops the reference and compares "
            "backends against each other at the scenario tolerance instead."
        ),
    )
    parser.add_argument(
        "--inference-sensitivities",
        action="store_true",
        help="Request forward sensitivities in the inference lane.",
    )
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=0,
        help="Seed for the inference lane's input vectors (default: %(default)s).",
    )
    parser.add_argument("--json", type=Path)
    return parser


def include_aot_for(lane: str, aot: str) -> bool:
    """Whether ``lane`` runs the AOT backend rows under the ``--aot`` setting."""
    if aot == "none":
        return False
    if aot == "all":
        return True
    return lane == "solver"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout


def _git_metadata() -> dict[str, str | bool | None]:
    """Identify the source that produced a run.

    A dirty tree makes the revision alone ambiguous, so the combined staged and
    unstaged diff is digested too and two different local implementations of the
    same idea no longer share their metadata. Untracked files are outside the
    digest, as they are outside the diff.
    """
    try:
        revision = _git("rev-parse", "HEAD").strip()
        status = _git("status", "--porcelain")
        diff = _git("diff", "HEAD")
        digest = hashlib.sha256(diff.encode()).hexdigest()[:16] if diff else None
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": None, "git_dirty": None, "git_diff_digest": None}
    return {
        "git_revision": revision,
        "git_dirty": bool(status),
        "git_diff_digest": digest,
    }


def _runtime_metadata() -> dict:
    return {
        **_git_metadata(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {
            name: _package_version(name)
            for name in ("pybamm", "pybammsolvers", "casadi")
        },
    }


def repeats_for(lane: str, requested: int | None) -> int:
    """Timed repeats for one lane; ``None`` takes that lane's own default.

    The inference lane reports a p10-p90 spread, which needs more samples than
    the paired medians the other lanes report.
    """
    if requested is not None:
        return requested
    return _DEFAULT_INFERENCE_REPEATS if lane == "inference" else _DEFAULT_REPEATS


def _comparison_metadata(args, runtime_metadata: dict, repeats: int) -> dict:
    """The metadata every comparison lane records, before its own additions."""
    return {
        "repeats": repeats,
        "warmup": args.warmup,
        "requested_output_points": args.output_points,
        "reference_tolerance": args.reference_tolerance or None,
        "protocols": args.protocols,
        "aot": args.aot,
        "backend_order_seed": args.backend_order_seed,
        **runtime_metadata,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the selected lanes and print their tables.

    Returns
    -------
    int
        Process exit status; 0 once every requested lane has run.
    """
    args = build_parser().parse_args(argv)
    if args.repeats is not None and args.repeats < 1:
        raise ValueError("repeats must be at least 1")
    repeats = repeats_for("solver", args.repeats)
    inference_repeats = repeats_for("inference", args.repeats)
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    payloads = {}
    runtime_metadata = _runtime_metadata()

    if args.lane in {"artifact", "all"}:
        artifact_results = run_artifact_lane(
            get_artifact_scenarios(args.artifact_scenarios),
            repeats=repeats,
            warmup=args.warmup,
        )
        print("Artifact lane")
        print(render_artifact_table(artifact_results))
        payloads["artifact"] = suite_to_jsonable(
            "artifact",
            artifact_results,
            metadata={
                **runtime_metadata,
                "repeats": repeats,
                "warmup": args.warmup,
            },
        )
        print()

    if args.lane in {"solver", "all"}:
        solver_results = run_solver_lane(
            get_solver_scenarios(
                args.models, args.protocols, output_points=args.output_points
            ),
            repeats=repeats,
            warmup=args.warmup,
            include_aot=include_aot_for("solver", args.aot),
            backend_order_seed=args.backend_order_seed,
            reference_tolerance=args.reference_tolerance,
        )
        print("Solver lane")
        print(render_solver_table(solver_results))
        payloads["solver"] = suite_to_jsonable(
            "solver",
            solver_results,
            metadata={
                **_comparison_metadata(args, runtime_metadata, repeats),
                "diffsol_always_attempted": True,
                "aot_profile": "isolated miss plus fresh-process disk reload",
            },
        )
        print()

    if args.lane in {"sensitivity", "all"}:
        sensitivity_results = run_sensitivity_lane(
            get_solver_scenarios(
                args.models, args.protocols, output_points=args.output_points
            ),
            repeats=repeats,
            warmup=args.warmup,
            include_aot=include_aot_for("sensitivity", args.aot),
            backend_order_seed=args.backend_order_seed,
            reference_tolerance=args.reference_tolerance,
        )
        print("Sensitivity lane")
        print(render_sensitivity_table(sensitivity_results))
        payloads["sensitivity"] = suite_to_jsonable(
            "sensitivity",
            sensitivity_results,
            metadata={
                **_comparison_metadata(args, runtime_metadata, repeats),
                # The superset asked for; a protocol can narrow it, so the
                # parameters actually differentiated are recorded per row.
                "sensitivity_parameters_requested": sorted(SENSITIVITY_INPUTS.values()),
                "aot_profile": "isolated miss plus fresh-process disk reload",
            },
        )

    if args.lane in {"inference", "all"}:
        inference_results = run_inference_lane(
            get_inference_scenarios(
                args.models, args.protocols, output_points=args.output_points
            ),
            repeats=inference_repeats,
            warmup=args.warmup,
            seed=args.inference_seed,
            sensitivities=args.inference_sensitivities,
            include_aot=include_aot_for("inference", args.aot),
            backend_order_seed=args.backend_order_seed,
            reference_tolerance=args.reference_tolerance,
        )
        print()
        print("Inference lane")
        print(render_inference_table(inference_results))
        payloads["inference"] = suite_to_jsonable(
            "inference",
            inference_results,
            metadata={
                **_comparison_metadata(args, runtime_metadata, inference_repeats),
                "inference_parameters": sorted(INFERENCE_INPUTS.values()),
                "inference_seed": args.inference_seed,
                "inference_sensitivities": args.inference_sensitivities,
            },
        )

    if args.json:
        json_payload = payloads
        if args.lane in {"artifact", "solver", "sensitivity", "inference"}:
            json_payload = payloads[args.lane]
        args.json.write_text(json.dumps(json_payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
