#!/usr/bin/env python3

import argparse
import gc
import inspect
import json
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

import pybamm


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    experiment_factory: callable
    solve_kwargs: dict
    solve_inputs_list: tuple[dict, ...] = ({}, {}, {})


def make_event_driven_cccv_experiment():
    return pybamm.Experiment(
        [("Charge at C/3 until 4.1 V", "Hold at 4.1 V until C/20")]
    )


def make_mixed_control_experiment():
    s = pybamm.step.string
    return pybamm.Experiment(
        [
            (
                s("Discharge at C/20 for 20 minutes"),
                s("Charge at 1 A for 10 minutes"),
                s("Hold at 4.1 V for 10 minutes"),
                "Discharge at 2 W for 10 minutes",
                "Discharge at 4 Ohm for 10 minutes",
            )
        ]
    )


def make_start_time_padding_experiment():
    return pybamm.Experiment(
        [
            pybamm.step.Current(
                1,
                duration=20 * 60,
                start_time=datetime(2023, 1, 1, 8, 0, 0),
            ),
            pybamm.step.rest(
                duration=20 * 60,
                start_time=datetime(2023, 1, 1, 9, 0, 0),
            ),
        ]
    )


def make_input_parameter_repeat_experiment():
    s = pybamm.step.string
    return pybamm.Experiment(
        [
            (
                pybamm.step.current(
                    pybamm.InputParameter("I_app"),
                    duration=20 * 60,
                ),
                s("Charge at 1 A for 10 minutes"),
                s("Hold at 4.1 V for 10 minutes"),
                "Discharge at 2 W for 10 minutes",
                "Discharge at 4 Ohm for 10 minutes",
            )
        ]
    )


SCENARIOS = [
    Scenario(
        name="event_driven_cccv",
        description="Charge to voltage cutoff, then hold until current cutoff",
        experiment_factory=make_event_driven_cccv_experiment,
        solve_kwargs={"initial_soc": 0.2, "calc_esoh": False},
    ),
    Scenario(
        name="mixed_control",
        description="Current, voltage, power, and resistance control in one cycle",
        experiment_factory=make_mixed_control_experiment,
        solve_kwargs={"calc_esoh": False},
    ),
    Scenario(
        name="start_time_padding",
        description="Timestamped steps with a gap that activates padding rest",
        experiment_factory=make_start_time_padding_experiment,
        solve_kwargs={"calc_esoh": False},
    ),
    Scenario(
        name="input_parameter_repeated_solves",
        description=(
            "Mixed-control cycle repeated on one built simulation with varying "
            "input current in the first step"
        ),
        experiment_factory=make_input_parameter_repeat_experiment,
        solve_kwargs={"calc_esoh": False},
        solve_inputs_list=(
            {"I_app": 0.5},
            {"I_app": 1.0},
            {"I_app": 1.5},
        ),
    ),
]

MODEL_FACTORIES = {
    "SPM": pybamm.lithium_ion.SPM,
    "SPMe": pybamm.lithium_ion.SPMe,
    "DFN": pybamm.lithium_ion.DFN,
}

MODES = ("legacy", "unified")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark legacy vs unified experiment model paths for representative "
            "experiment setups."
        )
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repetitions for each model/scenario/mode pair.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of untimed warmup repetitions for each pair.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_FACTORIES),
        default=list(MODEL_FACTORIES),
        help="Base models to benchmark.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=[scenario.name for scenario in SCENARIOS],
        default=[scenario.name for scenario in SCENARIOS],
        help="Benchmark scenarios to run.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Optional path to write the raw benchmark results as JSON.",
    )
    return parser.parse_args()


def simulation_supports_experiment_model_mode():
    return (
        "experiment_model_mode"
        in inspect.signature(pybamm.Simulation.__init__).parameters
    )


def mode_is_supported(mode):
    if mode == "legacy":
        return True
    return simulation_supports_experiment_model_mode()


def run_case(model_name, scenario, mode):
    model = MODEL_FACTORIES[model_name]()
    solver = pybamm.IDAKLUSolver()
    experiment = scenario.experiment_factory()
    simulation_kwargs = {
        "experiment": experiment,
        "solver": solver,
    }
    if simulation_supports_experiment_model_mode():
        simulation_kwargs["experiment_model_mode"] = mode
    elif mode != "legacy":
        raise NotImplementedError(
            "This PyBaMM version does not support experiment_model_mode="
            f"{mode!r}. Run this script on the unified-experiment branch to "
            "benchmark that mode."
        )

    sim = pybamm.Simulation(model, **simulation_kwargs)

    t0 = time.perf_counter()
    sim.build_for_experiment()
    t1 = time.perf_counter()
    solutions = []
    solve_step_s = []
    for solve_inputs in scenario.solve_inputs_list:
        solve_kwargs = dict(scenario.solve_kwargs)
        if solve_inputs:
            solve_kwargs["inputs"] = solve_inputs
        solve_t0 = time.perf_counter()
        solutions.append(sim.solve(**solve_kwargs))
        solve_step_s.append(time.perf_counter() - solve_t0)
    t2 = time.perf_counter()
    solution = solutions[-1]

    return {
        "build_s": t1 - t0,
        "solve_s": sum(solve_step_s),
        "solve_step_s": solve_step_s,
        "total_s": t2 - t0,
        "solve_count": len(solutions),
        "termination": solution.termination,
        "uses_unified_model": getattr(sim, "_experiment_uses_unified_model", False),
        "solutions": solutions,
        "status": "ok",
    }


def median(values):
    return statistics.median(values)


def format_seconds(value):
    return f"{value:.3f}"


def format_ratio(unified_value, legacy_value):
    return f"{unified_value / legacy_value:.2f}x"


def _raise_mismatch(message):
    raise RuntimeError(f"Legacy/unified benchmark mismatch: {message}")


def assert_solutions_match(legacy_solution, unified_solution):
    if legacy_solution.termination != unified_solution.termination:
        _raise_mismatch(
            "top-level termination differs: "
            f"{legacy_solution.termination!r} != {unified_solution.termination!r}"
        )
    np.testing.assert_allclose(
        legacy_solution.t[-1], unified_solution.t[-1], rtol=5e-5, atol=5e-4
    )
    np.testing.assert_allclose(
        legacy_solution["Discharge capacity [A.h]"].data[-1],
        unified_solution["Discharge capacity [A.h]"].data[-1],
        rtol=5e-5,
        atol=5e-5,
    )

    if len(legacy_solution.cycles) != len(unified_solution.cycles):
        _raise_mismatch(
            "number of cycles differs: "
            f"{len(legacy_solution.cycles)} != {len(unified_solution.cycles)}"
        )
    for legacy_cycle, unified_cycle in zip(
        legacy_solution.cycles, unified_solution.cycles, strict=True
    ):
        if len(legacy_cycle.steps) != len(unified_cycle.steps):
            _raise_mismatch(
                "number of steps in a cycle differs: "
                f"{len(legacy_cycle.steps)} != {len(unified_cycle.steps)}"
            )
        for legacy_step, unified_step in zip(
            legacy_cycle.steps, unified_cycle.steps, strict=True
        ):
            if legacy_step.termination != unified_step.termination:
                _raise_mismatch(
                    "step termination differs: "
                    f"{legacy_step.termination!r} != {unified_step.termination!r}"
                )
            np.testing.assert_allclose(
                legacy_step.t[-1], unified_step.t[-1], rtol=5e-5, atol=5e-4
            )
            np.testing.assert_allclose(
                legacy_step["Voltage [V]"].data[-1],
                unified_step["Voltage [V]"].data[-1],
                rtol=5e-5,
                atol=5e-5,
            )
            np.testing.assert_allclose(
                legacy_step["Current [A]"].data[-1],
                unified_step["Current [A]"].data[-1],
                rtol=5e-5,
                atol=5e-5,
            )


def verify_scenario_pair(model_name, scenario):
    legacy_result = run_case(model_name, scenario, "legacy")
    unified_result = run_case(model_name, scenario, "unified")
    if len(legacy_result["solutions"]) != len(unified_result["solutions"]):
        _raise_mismatch(
            "number of repeated solves differs: "
            f"{len(legacy_result['solutions'])} != {len(unified_result['solutions'])}"
        )
    for legacy_solution, unified_solution in zip(
        legacy_result["solutions"], unified_result["solutions"], strict=True
    ):
        assert_solutions_match(legacy_solution, unified_solution)
    gc.collect()


def unsupported_result(model_name, scenario, mode, reason):
    return {
        "model": model_name,
        "scenario": scenario.name,
        "description": scenario.description,
        "mode": mode,
        "build_s": None,
        "solve_s": None,
        "solve_step_s": [None] * len(scenario.solve_inputs_list),
        "total_s": None,
        "solve_count": len(scenario.solve_inputs_list),
        "termination": "n/a",
        "uses_unified_model": False,
        "status": reason,
    }


def main():
    args = parse_args()
    scenarios = [scenario for scenario in SCENARIOS if scenario.name in args.scenarios]
    results = []
    supported_modes = [mode for mode in MODES if mode_is_supported(mode)]
    max_solve_count = max(len(scenario.solve_inputs_list) for scenario in scenarios)

    print(
        f"Running {len(scenarios)} scenario(s) across {len(args.models)} model(s), "
        f"{len(supported_modes)} supported mode(s), warmup={args.warmup}, "
        f"repeats={args.repeats}"
    )

    for model_name in args.models:
        for scenario in scenarios:
            if mode_is_supported("unified"):
                print(f"* verifying {model_name} / {scenario.name} legacy vs unified")
                verify_scenario_pair(model_name, scenario)
            for mode in MODES:
                if not mode_is_supported(mode):
                    print(f"- {model_name} / {scenario.name} / {mode} (unsupported)")
                    results.append(
                        unsupported_result(
                            model_name,
                            scenario,
                            mode,
                            "unsupported on this PyBaMM version",
                        )
                    )
                    continue
                print(f"- {model_name} / {scenario.name} / {mode}")
                for _ in range(args.warmup):
                    run_case(model_name, scenario, mode)
                    gc.collect()

                samples = [
                    run_case(model_name, scenario, mode) for _ in range(args.repeats)
                ]
                gc.collect()

                results.append(
                    {
                        "model": model_name,
                        "scenario": scenario.name,
                        "description": scenario.description,
                        "mode": mode,
                        "build_s": median([sample["build_s"] for sample in samples]),
                        "solve_s": median([sample["solve_s"] for sample in samples]),
                        "solve_step_s": [
                            median(
                                [
                                    sample["solve_step_s"][solve_index]
                                    for sample in samples
                                ]
                            )
                            for solve_index in range(samples[-1]["solve_count"])
                        ],
                        "total_s": median([sample["total_s"] for sample in samples]),
                        "solve_count": samples[-1]["solve_count"],
                        "termination": samples[-1]["termination"],
                        "uses_unified_model": samples[-1]["uses_unified_model"],
                        "status": "ok",
                    }
                )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    print("\n## Raw Results")
    solve_headers = " | ".join(
        f"Solve {solve_index} (s)" for solve_index in range(1, max_solve_count + 1)
    )
    print(
        "| Model | Scenario | Mode | Solves | Build (s) | "
        f"{solve_headers} | Solve Total (s) | Total (s) | Termination |"
    )
    print(
        "| --- | --- | --- | ---: | ---: | "
        + " | ".join(["---:"] * max_solve_count)
        + " | ---: | ---: | --- |"
    )
    for row in results:
        solve_step_displays = [
            (
                format_seconds(row["solve_step_s"][solve_index])
                if solve_index < len(row["solve_step_s"])
                and row["solve_step_s"][solve_index] is not None
                else "n/a"
            )
            for solve_index in range(max_solve_count)
        ]
        print(
            "| "
            f"{row['model']} | {row['scenario']} | {row['mode']} | "
            f"{row['solve_count']} | "
            f"{format_seconds(row['build_s']) if row['build_s'] is not None else 'n/a'} | "
            + " | ".join(solve_step_displays)
            + " | "
            f"{format_seconds(row['solve_s']) if row['solve_s'] is not None else 'n/a'} | "
            f"{format_seconds(row['total_s']) if row['total_s'] is not None else 'n/a'} | "
            f"{row['termination'] if row['status'] == 'ok' else row['status']} |"
        )

    print("\n## Unified vs Legacy Comparison")
    print(
        "| Model | Scenario | Legacy Total (s) | Unified Total (s) | Unified / Legacy |"
    )
    print("| --- | --- | ---: | ---: | ---: |")
    for model_name in args.models:
        for scenario in scenarios:
            legacy_row = next(
                row
                for row in results
                if row["model"] == model_name
                and row["scenario"] == scenario.name
                and row["mode"] == "legacy"
            )
            unified_row = next(
                row
                for row in results
                if row["model"] == model_name
                and row["scenario"] == scenario.name
                and row["mode"] == "unified"
            )
            if legacy_row["status"] != "ok" or unified_row["status"] != "ok":
                unified_display = (
                    format_seconds(unified_row["total_s"])
                    if unified_row["total_s"] is not None
                    else "n/a"
                )
                ratio_display = (
                    format_ratio(unified_row["total_s"], legacy_row["total_s"])
                    if unified_row["total_s"] is not None
                    and legacy_row["total_s"] is not None
                    else "n/a"
                )
            else:
                unified_display = format_seconds(unified_row["total_s"])
                ratio_display = format_ratio(
                    unified_row["total_s"], legacy_row["total_s"]
                )
            print(
                "| "
                f"{model_name} | {scenario.name} | "
                f"{format_seconds(legacy_row['total_s'])} | "
                f"{unified_display} | "
                f"{ratio_display} |"
            )


if __name__ == "__main__":
    main()
