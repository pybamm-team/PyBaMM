from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import benchmarks.run_rust_observability as run_module
import pybamm
from benchmarks.run_rust_observability import build_parser, include_aot_for
from benchmarks.rust_observability import runners
from benchmarks.rust_observability.registry import (
    BASE_PARAMETER_SET,
    CHARGE_C_RATE,
    DEFAULT_OUTPUT_POINTS,
    INFERENCE_COMPLEMENTS,
    INFERENCE_INPUTS,
    INFERENCE_SPREADS,
    TRIANGLE_AMPLITUDE_A,
    TRIANGLE_PERIOD_S,
    get_inference_scenarios,
    get_protocol_names,
    get_solver_scenarios,
    inference_nominal_values,
)
from benchmarks.rust_observability.report import (
    _fits,
    render_inference_table,
    render_sensitivity_table,
    render_solver_table,
    suite_to_jsonable,
)
from benchmarks.rust_observability.runners import (
    _BASELINE_CASE,
    DEFAULT_REFERENCE_TOLERANCE,
    AotProfile,
    ComparisonSummary,
    InferenceResult,
    JacobianTelemetry,
    PhaseTiming,
    RepeatObservation,
    SensitivityResult,
    SolverResult,
    TimingSamples,
    TrajectorySummary,
    _aot_worker_payload,
    _build_and_time,
    _build_sensitivity_parameters,
    _comparable_length,
    _get_jacobian_telemetry,
    _observation_grid,
    _observed_sensitivities,
    _observed_values,
    _reference_ladder,
    _reference_tolerances,
    _resolve_reference,
    _sensitivity_tolerances,
    _shuffled_backend_cases,
    _solve_kwargs,
    _summarize_cache_statuses,
    _summarize_timing_samples,
    _worst_repeat_comparison,
    _worst_repeat_trajectory,
    backend_cases,
    resolve_reference_tolerance,
    sample_input_vectors,
    summarize_diff,
    summarize_trajectory,
)


def make_repeat(
    values, *, times=None, sensitivities=None, final_time=None, termination="final time"
):
    """A ``RepeatObservation`` from just the values, for comparison tests."""
    values = np.asarray(values, dtype=np.float64)
    times = np.arange(values.shape[0], dtype=np.float64) if times is None else times
    return RepeatObservation(
        values=values,
        times=np.asarray(times, dtype=np.float64),
        sensitivities=sensitivities,
        sensitivity_times=None if sensitivities is None else times,
        final_time=float(times[-1]) if final_time is None else final_time,
        termination=termination,
    )


class TestProtocolRegistry:
    def test_default_protocol_preserves_scenario_identity(self):
        scenarios = get_solver_scenarios(["SPM", "SPMe", "DFN"])

        assert [s.name for s in scenarios] == ["SPM", "SPMe", "DFN"]
        assert {s.protocol for s in scenarios} == {"cc_discharge"}
        assert all(s.initial_soc is None for s in scenarios)
        assert all(s.plan.experiment is None for s in scenarios)

    def test_cross_product_is_model_major(self):
        scenarios = get_solver_scenarios(
            ["SPM", "DFN"], ["cc_discharge", "drive_cycle"]
        )

        assert [(s.name, s.protocol) for s in scenarios] == [
            ("SPM", "cc_discharge"),
            ("SPM", "drive_cycle"),
            ("DFN", "cc_discharge"),
            ("DFN", "drive_cycle"),
        ]

    def test_unknown_protocol_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown names requested: bogus"):
            get_solver_scenarios(["SPM"], ["bogus"])

    def test_interpolant_protocols_pass_breakpoints_as_t_eval(self):
        for protocol in ("drive_cycle", "pulse_train"):
            scenario = get_solver_scenarios(["SPM"], [protocol])[0]
            parameter_values = scenario.parameter_values_builder()
            current = parameter_values["Current function [A]"]

            assert isinstance(current, pybamm.Interpolant)
            assert scenario.initial_soc == 0.5
            # Every breakpoint must be in t_eval or PyBaMM warns about resolution.
            breakpoints = np.asarray(current.x[0], dtype=np.float64)
            assert np.isin(
                np.round(breakpoints, 12), np.round(scenario.plan.t_eval, 12)
            ).all()

    def test_triangle_wave_is_exact_through_its_vertices(self):
        scenario = get_solver_scenarios(["SPM"], ["drive_cycle"])[0]
        current = scenario.parameter_values_builder()["Current function [A]"]
        # Interpolant.x is a list of arrays; Interpolant.y is a plain ndarray.
        vertices = np.asarray(current.x[0], dtype=np.float64)
        values = np.asarray(current.y, dtype=np.float64).reshape(-1)

        dense_t = np.linspace(vertices[0], vertices[-1], 2001)
        expected = (
            TRIANGLE_AMPLITUDE_A
            * (2.0 / np.pi)
            * np.arcsin(np.sin(2.0 * np.pi * dense_t / TRIANGLE_PERIOD_S))
        )
        np.testing.assert_allclose(
            np.interp(dense_t, vertices, values), expected, atol=1e-12
        )

    def test_charge_and_experiment_plans(self):
        charge = get_solver_scenarios(["SPM"], ["cc_charge"])[0]
        assert charge.initial_soc == 0.0
        parameter_values = charge.parameter_values_builder()
        current = parameter_values["Current function [A]"]
        nominal_capacity = float(parameter_values["Nominal cell capacity [A.h]"])
        assert float(current) == pytest.approx(-CHARGE_C_RATE * nominal_capacity)

        experiment = get_solver_scenarios(["SPM"], ["experiment"])[0]
        assert experiment.plan.experiment is not None
        assert experiment.plan.t_interp is None
        # Period-driven grid, so the output-points knob does not apply.
        assert experiment.plan.requested_points == 0

    def test_output_points_only_resizes_grid_protocols(self):
        grid = get_solver_scenarios(["SPM"], ["cc_discharge"], output_points=250)[0]
        assert grid.plan.t_interp.size == 250
        assert grid.plan.requested_points == 250

        experiment = get_solver_scenarios(["SPM"], ["experiment"], output_points=250)[0]
        assert experiment.plan.t_interp is None


class TestRustObservabilityBenchmark:
    def test_output_points_are_explicit(self):
        default = get_solver_scenarios(["SPM"])[0]
        dense = get_solver_scenarios(["SPM"], output_points=1000)[0]

        assert default.plan.t_interp.size == DEFAULT_OUTPUT_POINTS
        assert dense.plan.t_interp.size == 1000
        np.testing.assert_array_equal(
            default.plan.t_interp[[0, -1]], dense.plan.t_interp[[0, -1]]
        )

        with pytest.raises(ValueError, match="at least 2"):
            get_solver_scenarios(["SPM"], output_points=1)

    def test_e2e_uses_paired_wall_samples(self):
        samples = TimingSamples(
            warm_set_up_ms=(0.1, 0.1, 0.1),
            solve_ms=(1.0, 2.0, 100.0),
            wall_solve_ms=(1.1, 2.1, 100.1),
            integration_ms=(0.9, 1.9, 99.9),
            observe_ms=(100.0, 2.0, 1.0),
            e2e_ms=(101.1, 4.1, 101.1),
        )

        timing = _summarize_timing_samples(
            samples,
            build_ms=3.0,
            cold_set_up_ms=4.0,
            cold_observe_ms=5.0,
            cold_startup_ms=12.0,
        )

        assert timing.prepare_ms == pytest.approx(9.0)
        assert timing.cold_startup_ms == pytest.approx(12.0)
        assert timing.cold_startup_ms >= timing.build_ms + timing.prepare_ms
        assert timing.solve_ms == pytest.approx(2.0)
        assert timing.observe_ms == pytest.approx(2.0)
        assert timing.e2e_ms == pytest.approx(101.1)
        assert timing.e2e_ms != pytest.approx(timing.solve_ms + timing.observe_ms)

    def test_trajectory_rejects_early_termination(self):
        baseline = SimpleNamespace(
            t=np.array([0.0, 1.0, 2.0]), termination="final time"
        )
        candidate = SimpleNamespace(t=np.array([0.0, 1.0]), termination="event")

        summary = summarize_trajectory(baseline, candidate, atol=1e-9, rtol=1e-9)

        assert summary.status == "warn"
        assert summary.common_points == 2
        assert summary.coverage == pytest.approx(2 / 3)

    def test_trajectory_allows_terminal_roundoff(self):
        baseline = SimpleNamespace(
            t=np.array([0.0, 1.0, 2.0]), termination="final time"
        )
        candidate = SimpleNamespace(
            t=np.array([0.0, 1.0, 2.0 + 1e-6]), termination="final time"
        )

        summary = summarize_trajectory(baseline, candidate, atol=1e-9, rtol=1e-6)

        assert summary.status == "pass"
        assert summary.coverage == 1.0

    def test_shape_mismatch_warns(self):
        summary = summarize_diff(
            np.zeros((2, 3)),
            np.zeros((1, 3)),
            atol=1e-9,
            rtol=1e-9,
        )

        assert summary.status == "warn"
        assert np.isinf(summary.max_abs_diff)

    def test_report_order_compares_backends_within_output_mode(self):
        shuffled_backends = [
            "rust_diffsol_out",
            "rust_idaklu",
            "casadi_idaklu_aot_out",
            "casadi_idaklu",
            "rust_diffsol",
            "casadi_idaklu_out",
            "rust_idaklu_out",
            "casadi_idaklu_aot",
        ]
        results = [
            SolverResult(
                scenario="SPM",
                backend=backend,
                timings=PhaseTiming(),
                requested_output_points=100,
            )
            for backend in shuffled_backends
        ]
        expected = [
            "casadi_idaklu",
            "casadi_idaklu_aot",
            "rust_idaklu",
            "rust_diffsol",
            "casadi_idaklu_out",
            "casadi_idaklu_aot_out",
            "rust_idaklu_out",
            "rust_diffsol_out",
        ]

        # Wide enough that the layout is the one-table one, whatever columns it
        # carries, so this asserts on ordering rather than on the layout choice.
        report = render_solver_table(results, width=250)
        rendered = [
            line.split()[1] for line in report.splitlines() if line.startswith("SPM ")
        ]
        assert rendered == expected

        payload = suite_to_jsonable("solver", results)
        assert [result["backend"] for result in payload["results"]] == expected

    def test_aot_profile_is_rendered_and_serialized(self):
        profile = AotProfile(
            fresh_cache_statuses=("miss", "miss"),
            disk_cache_statuses=("disk", "disk"),
            codegen_ms=100.0,
            compiler_ms=17000.0,
            fresh_load_ms=1.0,
            disk_load_ms=2.0,
            fresh_total_ms=17101.0,
            disk_total_ms=3.0,
            disk_prepare_ms=45.0,
            disk_cold_startup_ms=110.0,
            library_size_bytes=2 * 1024**2,
            verified=True,
        )
        result = SolverResult(
            scenario="DFN",
            backend="casadi_idaklu_aot",
            timings=PhaseTiming(prepare_ms=17150.0, cold_startup_ms=17220.0),
            requested_output_points=100,
            aot_profile=profile,
        )

        report = render_solver_table([result], width=120)
        assert "AOT profile (isolated cache" in report
        assert "missx2" in report
        assert "diskx2" in report
        assert "17000.00" in report

        payload = suite_to_jsonable("solver", [result])
        encoded = json.loads(json.dumps(payload))
        assert encoded["results"][0]["aot_profile"]["verified"] is True
        assert encoded["results"][0]["aot_profile"]["compiler_ms"] == 17000.0

    def test_sensitivity_table_matches_solver_width_layout(self):
        trajectory = TrajectorySummary(
            baseline_points=100,
            candidate_points=100,
            common_points=100,
            coverage=1.0,
            max_time_diff=0.0,
            final_time_diff=0.0,
            baseline_termination="final time",
            candidate_termination="final time",
            status="pass",
        )
        comparison = ComparisonSummary(0.0, 0.0, 0.0)
        timing = PhaseTiming(
            build_ms=22.81,
            prepare_ms=29.49,
            cold_startup_ms=57.86,
            warm_set_up_ms=0.08,
            solve_ms=4.43,
            wall_solve_ms=4.55,
            integration_ms=4.32,
            observe_ms=16.39,
            e2e_ms=20.97,
        )
        results = [
            SensitivityResult(
                scenario="SPM",
                backend="casadi_idaklu",
                timings=timing,
                requested_output_points=1000,
            ),
            SensitivityResult(
                scenario="SPM",
                backend="casadi_idaklu_aot",
                timings=timing,
                requested_output_points=1000,
                state_sens_comparison=comparison,
                output_sens_comparison=comparison,
                trajectory_comparison=trajectory,
            ),
        ]

        compact = render_sensitivity_table(results, width=120)
        assert "Timings (ms)" in compact
        assert "Validation" in compact
        assert _fits(compact, 120)

        stacked = render_sensitivity_table(results, width=80)
        assert "One block per backend" in stacked
        assert _fits(stacked, 80)

    def test_solver_table_snapshot_and_json_samples(self, monkeypatch, snapshot):
        trajectory = TrajectorySummary(
            baseline_points=100,
            candidate_points=100,
            common_points=100,
            coverage=1.0,
            max_time_diff=0.0,
            final_time_diff=0.0,
            baseline_termination="event",
            candidate_termination="event",
            status="pass",
        )
        comparison = ComparisonSummary(1e-8, 1e-7, 0.01)
        samples = TimingSamples(
            warm_set_up_ms=(0.1,),
            solve_ms=(1.2,),
            wall_solve_ms=(1.3,),
            integration_ms=(1.1,),
            observe_ms=(0.2,),
            e2e_ms=(1.5,),
        )
        timing = PhaseTiming(
            build_ms=10.0,
            prepare_ms=5.2,
            cold_startup_ms=16.5,
            set_up_ms=5.0,
            warm_set_up_ms=0.1,
            solve_ms=1.2,
            wall_solve_ms=1.3,
            integration_ms=1.1,
            observe_ms=0.2,
            e2e_ms=1.5,
        )
        telemetry = {
            "SPM": JacobianTelemetry(
                strategy="coloring",
                n_colors=5,
                nnz=123,
                n_dense_rows=0,
                dense_row_entries=0,
                dense_row_tape_instructions=0,
                split_eval_primal_instructions=10,
                split_eval_total_instructions=20,
                split_eval_raw_instructions=20,
                branch_block_lens=(),
            ),
            "SPMe": JacobianTelemetry(
                strategy="coloring",
                n_colors=3,
                nnz=361,
                n_dense_rows=1,
                dense_row_entries=65,
                dense_row_tape_instructions=4096,
                split_eval_primal_instructions=10,
                split_eval_total_instructions=20,
                split_eval_raw_instructions=20,
                branch_block_lens=(),
            ),
            "DFN": JacobianTelemetry(
                strategy="coloring",
                n_colors=9,
                nnz=3673,
                n_dense_rows=0,
                dense_row_entries=0,
                dense_row_tape_instructions=0,
                split_eval_primal_instructions=10,
                split_eval_total_instructions=20,
                split_eval_raw_instructions=20,
                branch_block_lens=(),
            ),
        }
        results = [
            SolverResult(
                scenario=name,
                protocol="cc_discharge",
                backend="rust_idaklu",
                timings=timing,
                requested_output_points=100,
                timing_samples=samples,
                state_comparison=comparison,
                output_comparison=comparison,
                trajectory_comparison=trajectory,
                jacobian_telemetry=stats,
            )
            for name, stats in telemetry.items()
        ]

        compact = render_solver_table(results, width=120)
        snapshot.assert_match(
            compact + "\n",
            "rust_observability_solver_table.snapshot",
        )
        assert _fits(compact, 120)
        monkeypatch.setenv("COLUMNS", "120")
        assert render_solver_table(results) == compact

        wide = render_solver_table(results, width=240)
        assert "Prep" in wide
        assert "Cold" in wide
        assert "Validation\n" not in wide

        narrow = render_solver_table(results, width=80)
        assert _fits(narrow, 80)
        assert "One block per backend" in narrow

        payload = suite_to_jsonable("solver", results)
        encoded = json.loads(json.dumps(payload))
        assert encoded["results"][0]["timings"]["prepare_ms"] == 5.2
        assert encoded["results"][0]["timings"]["cold_startup_ms"] == 16.5
        assert encoded["results"][0]["timing_samples"]["e2e_ms"] == [1.5]


class TestProtocolSolveWiring:
    def test_grid_protocol_solve_kwargs(self):
        scenario = get_solver_scenarios(["SPM"], ["cc_discharge"])[0]

        kwargs = _solve_kwargs(scenario)

        assert kwargs["t_eval"] == [0.0, 3600.0]
        np.testing.assert_array_equal(kwargs["t_interp"], scenario.plan.t_interp)
        assert "experiment" not in kwargs

    def test_interpolant_protocol_passes_breakpoints(self):
        scenario = get_solver_scenarios(["SPM"], ["pulse_train"])[0]

        kwargs = _solve_kwargs(scenario)

        np.testing.assert_array_equal(kwargs["t_eval"], scenario.plan.t_eval)

    def test_experiment_protocol_omits_time_arguments(self):
        scenario = get_solver_scenarios(["SPM"], ["experiment"])[0]

        kwargs = _solve_kwargs(scenario)

        assert kwargs == {}

    def test_extra_kwargs_are_merged(self):
        scenario = get_solver_scenarios(["SPM"], ["cc_discharge"])[0]

        kwargs = _solve_kwargs(scenario, {"inputs": {"D_n": 1.0}})

        assert kwargs["inputs"] == {"D_n": 1.0}
        assert kwargs["t_eval"] == [0.0, 3600.0]


class _BuildSpy:
    """Stands in for a ``pybamm.Simulation``, recording ``build`` calls."""

    def __init__(self):
        self.build_calls: list[tuple[float | None, dict | None]] = []

    def build(self, initial_soc=None, inputs=None):
        self.build_calls.append((initial_soc, inputs))


class TestExperimentBuildInteraction:
    """Regression tests for two bugs surfaced by wiring the experiment protocol.

    Pre-building an experiment-attached ``Simulation`` reparameterises the same
    model that ``Simulation.solve`` then parameterises again per step, tripping
    PyBaMM's reparameterised-model guard and making every variable
    unprocessable. Separately, "Current function [A]" cannot be a sensitivity
    input once an ``Experiment`` supplies its own control law.
    """

    def test_build_and_time_skips_build_for_experiment_plan(self):
        scenario = get_solver_scenarios(["SPM"], ["experiment"])[0]
        simulation = _BuildSpy()

        build_ms = _build_and_time(simulation, scenario)

        # No `.build()` call means this is pure guard-check overhead, not a real build.
        assert build_ms < 1.0
        assert simulation.build_calls == []

    def test_build_and_time_builds_grid_plan_with_initial_soc(self):
        scenario = get_solver_scenarios(["SPM"], ["cc_charge"])[0]
        simulation = _BuildSpy()

        _build_and_time(simulation, scenario)

        assert simulation.build_calls == [(scenario.initial_soc, None)]

    def test_build_and_time_forwards_inputs_for_the_initial_state(self):
        scenario = get_solver_scenarios(["SPM"], ["cc_charge"])[0]
        simulation = _BuildSpy()
        inputs = {"eps_p": 0.66}

        _build_and_time(simulation, scenario, inputs=inputs)

        # Mapping initial_soc to concentrations runs an ElectrodeSOH solve, which
        # cannot evaluate a symbolic parameter without its value.
        assert simulation.build_calls == [(0.0, inputs)]

    def test_sensitivity_parameters_drop_dead_current_input_under_experiment(self):
        scenario = get_solver_scenarios(["SPM"], ["experiment"])[0]
        parameter_values, inputs = _build_sensitivity_parameters(scenario)

        assert "I" not in inputs
        assert "eps_p" in inputs
        assert not isinstance(
            parameter_values["Current function [A]"], pybamm.InputParameter
        )

    def test_sensitivity_parameters_include_current_input_for_grid_protocol(self):
        scenario = get_solver_scenarios(["SPM"], ["cc_discharge"])[0]
        parameter_values, inputs = _build_sensitivity_parameters(scenario)

        assert "I" in inputs
        assert isinstance(
            parameter_values["Current function [A]"], pybamm.InputParameter
        )

    def test_jacobian_telemetry_handles_never_set_up_solver(self):
        # `_setup` is only assigned inside `set_up()`, never in `__init__`; the
        # experiment path solves via a per-step copy, leaving this one bare.
        solver = pybamm.IDAKLUSolver()

        assert _get_jacobian_telemetry(solver, "rust_idaklu") is None


class TestAotGating:
    def test_backend_cases_drop_aot_when_disabled(self):
        with_aot = backend_cases(include_aot=True)
        without_aot = backend_cases(include_aot=False)

        assert any(backend == "casadi_idaklu_aot" for backend, _ in with_aot)
        assert not any(backend == "casadi_idaklu_aot" for backend, _ in without_aot)
        assert len(with_aot) - len(without_aot) == 2  # full-state and output-only rows

    def test_disk_worker_payload_is_names_not_objects(self):
        scenario = get_solver_scenarios(["SPM"], ["drive_cycle"])[0]

        payload = _aot_worker_payload(
            "solver", scenario, output_only=False, cache_dir="/tmp/x", output_points=100
        )

        # Rebuilt by name in the worker, so no PyBaMM object crosses the boundary.
        assert payload == ("solver", "SPM", "drive_cycle", 100, False, "/tmp/x")
        pickle.loads(pickle.dumps(payload))


class TestInferenceLane:
    def test_input_vectors_are_seed_reproducible_and_varying(self):
        nominal = {"D_n": 3.3e-14, "eps_p": 0.665}

        first = sample_input_vectors(nominal, 5, seed=0)
        again = sample_input_vectors(nominal, 5, seed=0)
        other = sample_input_vectors(nominal, 5, seed=1)

        assert first == again  # identical across backends for a given seed
        assert first != other
        assert len(first) == 5
        assert all(set(vector) == set(nominal) for vector in first)
        # Every repeat differs, or the lane is not measuring changing inputs.
        assert len({vector["D_n"] for vector in first}) == 5

    def test_input_vectors_stay_within_spread(self):
        nominal = {"D_n": 3.3e-14}

        vectors = sample_input_vectors(nominal, 200, seed=3, spread=0.2)

        values = np.array([vector["D_n"] for vector in vectors])
        assert values.min() >= 3.3e-14 * 0.8
        assert values.max() <= 3.3e-14 * 1.2

    def test_spread_can_differ_per_parameter(self):
        """One width cannot suit parameters of different natures.

        20% is routine for a diffusivity but takes a volume fraction to a porosity
        DFN cannot converge at and SPMe saturates on charge.
        """
        nominal = {"D_n": 3.3e-14, "eps_n": 0.75}

        vectors = sample_input_vectors(
            nominal, 200, seed=3, spread={"D_n": 0.2, "eps_n": 0.05}
        )

        diffusivity = np.array([vector["D_n"] for vector in vectors])
        fraction = np.array([vector["eps_n"] for vector in vectors])
        assert diffusivity.min() < 3.3e-14 * 0.85
        assert fraction.min() >= 0.75 * 0.95
        assert fraction.max() <= 0.75 * 1.05

    def test_a_missing_per_parameter_spread_is_rejected(self):
        # A new fitted parameter must not silently inherit another's width.
        with pytest.raises(KeyError):
            sample_input_vectors(
                {"D_n": 3.3e-14, "eps_n": 0.75}, 2, seed=0, spread={"D_n": 0.2}
            )

    def test_every_fitted_parameter_has_a_spread(self):
        assert set(INFERENCE_SPREADS) == set(INFERENCE_INPUTS.values())

    def test_sampled_porosity_stays_physical(self):
        """Porosity is the complement of a fitted fraction, so its width is derived.

        At 20% it ranged 0.165-0.39 about a nominal 0.25, well outside anything the
        base set describes.
        """
        vectors = sample_input_vectors(
            inference_nominal_values(), 200, seed=0, spread=INFERENCE_SPREADS
        )

        for input_name in INFERENCE_COMPLEMENTS.values():
            porosity = np.array([1.0 - vector[input_name] for vector in vectors])
            assert porosity.min() > 0.2
            assert porosity.max() < 0.4

    def test_inference_scenario_makes_every_parameter_an_input(self):
        scenario = get_inference_scenarios(["SPM"])[0]
        parameter_values = scenario.parameter_values_builder()

        for pybamm_name, input_name in INFERENCE_INPUTS.items():
            assert isinstance(parameter_values[pybamm_name], pybamm.InputParameter)
            assert parameter_values[pybamm_name].name == input_name

    def test_inference_scenarios_keep_protocol_identity(self):
        scenarios = get_inference_scenarios(["SPM", "DFN"], ["cc_discharge"])

        assert [(s.name, s.protocol) for s in scenarios] == [
            ("SPM", "cc_discharge"),
            ("DFN", "cc_discharge"),
        ]

    @pytest.mark.parametrize(
        "protocol", ["cc_discharge", "cc_charge", "drive_cycle", "pulse_train"]
    )
    def test_inference_preserves_the_protocol_control_law(self, protocol):
        """The inference lane layers inputs on the protocol, it does not replace it.

        Replacing the builder silently reverted every protocol to a plain
        constant-current discharge while the table still named the protocol.
        """
        solver = get_solver_scenarios(["SPM"], [protocol])[0]
        inference = get_inference_scenarios(["SPM"], [protocol])[0]
        key = "Current function [A]"

        expected = solver.parameter_values_builder()[key]
        actual = inference.parameter_values_builder()[key]

        assert type(actual) is type(expected)
        if isinstance(expected, pybamm.Interpolant):
            np.testing.assert_array_equal(actual.x[0], expected.x[0])
            np.testing.assert_array_equal(actual.y, expected.y)
        else:
            assert float(actual) == pytest.approx(float(expected))

    def test_inference_keeps_electrode_volume_fractions_feasible(self):
        scenario = get_inference_scenarios(["SPM"], ["cc_discharge"])[0]
        parameter_values = scenario.parameter_values_builder()
        nominal = inference_nominal_values()

        for porosity_name, input_name in INFERENCE_COMPLEMENTS.items():
            porosity = parameter_values[porosity_name]
            # Solid and pore volume must still sum to 1 at any sampled value.
            assert porosity.evaluate(inputs={input_name: 0.9}) == pytest.approx(0.1)
            assert porosity.evaluate(
                inputs={input_name: nominal[input_name]}
            ) == pytest.approx(1.0 - nominal[input_name])

    def test_every_protocol_shares_one_base_parameter_set(self):
        base = pybamm.ParameterValues(BASE_PARAMETER_SET)

        for protocol in get_protocol_names():
            scenario = get_solver_scenarios(["SPM"], [protocol])[0]
            parameter_values = scenario.parameter_values_builder()
            # Only the current law may differ, or rows are not comparable.
            assert float(parameter_values["Nominal cell capacity [A.h]"]) == float(
                base["Nominal cell capacity [A.h]"]
            )

    def test_nominal_values_cover_every_fitted_parameter(self):
        nominal = inference_nominal_values()

        assert set(nominal) == set(INFERENCE_INPUTS.values())
        assert all(np.isfinite(value) and value > 0.0 for value in nominal.values())

    def test_observation_grid_interpolates_between_solver_nodes(self):
        scenario = get_inference_scenarios(["SPM"], ["cc_discharge"], output_points=5)[
            0
        ]

        nodes = scenario.plan.t_interp
        grid = _observation_grid(scenario)

        # Midpoints, so no observation time coincides with a solver output node.
        assert not np.isin(grid, nodes).any()
        # One per interval, less the first: it straddles the initial transient,
        # which no two-point interpolant can reach from its endpoints.
        assert grid.size == nodes.size - 2
        assert grid[0] > nodes[1]
        np.testing.assert_allclose(grid, 0.5 * (nodes[1:-1] + nodes[2:]))

    def test_observation_grid_is_empty_without_a_declared_grid(self):
        scenario = get_inference_scenarios(["SPM"], ["experiment"])[0]

        assert _observation_grid(scenario).size == 0

    def test_inference_result_reports_spread_and_one_time_costs(self):
        result = InferenceResult(
            scenario="SPM",
            protocol="cc_discharge",
            backend="rust_idaklu",
            build_ms=10.0,
            setup_ms=5.0,
            aot_cache_status="disk",
            eval_samples_ms=(1.0, 2.0, 3.0, 4.0, 5.0),
            solve_samples_ms=(0.8, 1.8, 2.8, 3.8, 4.8),
            observe_samples_ms=(0.2, 0.2, 0.2, 0.2, 0.2),
            requested_output_points=100,
        )

        assert result.eval_median_ms == pytest.approx(3.0)
        assert result.eval_p10_ms == pytest.approx(1.4)
        assert result.eval_p90_ms == pytest.approx(4.6)
        assert result.solve_median_ms == pytest.approx(2.8)
        assert result.observe_median_ms == pytest.approx(0.2)
        assert result.status == "baseline"

    def test_unsupported_inference_result_reports_reason(self):
        result = InferenceResult(
            scenario="SPM",
            protocol="experiment",
            backend="rust_diffsol",
            build_ms=0.0,
            setup_ms=0.0,
            aot_cache_status="-",
            eval_samples_ms=(),
            solve_samples_ms=(),
            observe_samples_ms=(),
            requested_output_points=0,
            supported=False,
            reason="SolverError: nope",
        )

        assert result.status == "unsupported"

    def test_cache_status_reports_what_the_compiler_did(self):
        assert _summarize_cache_statuses(None) == "-"
        assert _summarize_cache_statuses([]) == "-"
        assert (
            _summarize_cache_statuses([SimpleNamespace(cache_status="miss")] * 3)
            == "miss"
        )
        # A warm in-process reuse must not be reported as a fresh compile.
        assert (
            _summarize_cache_statuses(
                [
                    SimpleNamespace(cache_status="memory"),
                    SimpleNamespace(cache_status="disk"),
                ]
            )
            == "disk+memory"
        )

    def _compare(self, baseline, candidate, scenario):
        return _worst_repeat_comparison(
            baseline,
            candidate,
            select=_observed_values,
            atol=scenario.atol,
            rtol=scenario.rtol,
        )

    def test_ragged_repeats_compare_on_the_aligned_prefix(self):
        scenario = get_solver_scenarios(["SPM"], ["cc_discharge"])[0]
        # Both grids are prefixes of the same observation times, so trimming to
        # the shorter one compares like for like rather than shifting by one.
        trace = np.linspace(4.0, 3.0, 12)
        times = np.arange(12.0)
        summary = self._compare(
            [make_repeat(trace, times=times)],
            [make_repeat(trace[:9], times=times[:9], final_time=11.0)],
            scenario,
        )

        assert summary.max_abs_diff == pytest.approx(0.0)
        assert summary.status == "pass"

    def test_worst_repeat_drives_the_comparison(self):
        scenario = get_solver_scenarios(["SPM"], ["cc_discharge"])[0]
        off = np.full(10, 4.0)
        off[0] = 3.6
        # Second repeat is far off; the worst case must win, not the last.
        summary = self._compare(
            [make_repeat(np.full(10, 4.0)), make_repeat(np.full(10, 4.0))],
            [make_repeat(np.full(10, 4.0)), make_repeat(off)],
            scenario,
        )

        assert summary.max_abs_diff == pytest.approx(0.4)
        assert summary.status == "warn"

    def test_a_failing_repeat_is_never_masked_by_a_larger_permitted_one(self):
        """Tolerance scales with the baseline magnitude, so a repeat sitting at a
        lower voltage can breach it on a *smaller* absolute difference than one
        sitting higher up is allowed.
        """
        scenario = get_solver_scenarios(["SPM"], ["cc_discharge"])[0]
        tight = np.full(20, 2.5)
        tight_off = tight.copy()
        tight_off[5] += 4.0e-6  # breaches atol + rtol * 2.5
        loose = np.full(20, 4.2)
        loose_off = loose.copy()
        loose_off[5] += 4.5e-6  # larger, but inside atol + rtol * 4.2

        assert self._compare([make_repeat(loose)], [make_repeat(loose_off)], scenario)
        assert (
            self._compare([make_repeat(tight)], [make_repeat(tight_off)], scenario)
        ).status == "warn"
        summary = self._compare(
            [make_repeat(tight), make_repeat(loose)],
            [make_repeat(tight_off), make_repeat(loose_off)],
            scenario,
        )

        assert summary.status == "warn"
        assert summary.max_abs_diff == pytest.approx(4.0e-6)

    def test_endpoint_window_is_measured_not_a_fixed_point_count(self):
        scenario = get_solver_scenarios(["SPM"], ["cc_discharge"])[0]
        times = np.arange(10.0)
        # Terminations 2 s apart, so the last two points measure the endpoint gap.
        baseline = make_repeat(np.full(10, 4.0), times=times, final_time=9.0)
        candidate = make_repeat(np.full(10, 4.0), times=times, final_time=7.0)
        candidate.values[-2:] = 3.0

        assert _comparable_length(times, times, endpoint_gap=0.0) == 10
        assert _comparable_length(times, times, endpoint_gap=2.0) == 8
        assert self._compare([baseline], [candidate], scenario).status == "pass"

    def test_matching_terminations_drop_nothing(self):
        scenario = get_solver_scenarios(["SPM"], ["cc_discharge"])[0]
        times = np.arange(10.0)
        candidate = make_repeat(np.full(10, 4.0), times=times)
        candidate.values[-1] = 3.0
        baseline = make_repeat(np.full(10, 4.0), times=times)

        # Terminations coincide, so a final-point disagreement is real, not tail
        # noise, and must not be trimmed away.
        assert self._compare([baseline], [candidate], scenario).status == "warn"

    def test_early_termination_cannot_pass_on_a_matching_prefix(self):
        """A candidate stopping a tenth of the way in agrees perfectly wherever
        both were observed, so coverage and termination have to carry the verdict.
        """
        scenario = get_solver_scenarios(["SPM"], ["cc_discharge"])[0]
        times = np.linspace(0.0, 3600.0, 1000)
        trace = np.linspace(4.2, 2.5, 1000)
        baseline = make_repeat(trace, times=times, termination="final time")
        early = make_repeat(
            trace[:100], times=times[:100], termination="event: Minimum voltage [V]"
        )

        # Nothing outlives an endpoint gap this wide, so the values are not
        # comparable at all rather than agreeing on a truncated prefix.
        values = self._compare([baseline], [early], scenario)
        assert values.status == "warn"
        assert values.max_abs_diff == float("inf")

        trajectory = _worst_repeat_trajectory([baseline], [early], scenario)
        assert trajectory.status == "warn"
        assert trajectory.coverage == pytest.approx(0.1)
        assert trajectory.candidate_termination.startswith("event")
        assert trajectory.final_time_diff == pytest.approx(times[-1] - times[99])

    def test_gradient_comparison_is_skipped_when_not_requested(self):
        scenario = get_solver_scenarios(["SPM"], ["cc_discharge"])[0]
        atol, rtol = _sensitivity_tolerances(scenario)
        repeats = [make_repeat(np.full(4, 4.0))]

        assert (
            _worst_repeat_comparison(
                repeats, repeats, select=_observed_sensitivities, atol=atol, rtol=rtol
            )
            is None
        )

    def test_gradient_tolerance_is_looser_than_the_state_tolerance(self):
        scenario = get_solver_scenarios(["SPM"], ["cc_discharge"])[0]
        atol, rtol = _sensitivity_tolerances(scenario)

        # Forward sensitivities are not error-controlled to the state tolerance,
        # and the gate sits a decade above that noise floor rather than on it.
        assert atol == pytest.approx(10.0 * scenario.atol**0.5)
        assert rtol == pytest.approx(10.0 * scenario.rtol**0.5)
        assert atol > scenario.atol and rtol > scenario.rtol

    def test_gradient_comparison_catches_a_broken_chain_rule(self):
        scenario = get_solver_scenarios(["SPM"], ["cc_discharge"])[0]
        atol, rtol = _sensitivity_tolerances(scenario)
        gradient = np.tile(np.array([[0.34, 18.0]]), (6, 1))
        baseline = [make_repeat(np.full(6, 4.0), sensitivities=gradient)]
        noisy = [
            make_repeat(np.full(6, 4.0), sensitivities=gradient + 2.0e-6)
        ]  # integrator noise
        broken = [make_repeat(np.full(6, 4.0), sensitivities=gradient * 0.2)]

        def compare(repeats):
            return _worst_repeat_comparison(
                baseline, repeats, select=_observed_sensitivities, atol=atol, rtol=rtol
            )

        assert compare(noisy).status == "pass"
        assert compare(broken).status == "warn"


class TestProtocolReporting:
    def test_protocol_column_is_rendered_and_serialized(self):
        results = [
            SolverResult(
                scenario="SPM",
                protocol=protocol,
                backend="rust_idaklu",
                timings=PhaseTiming(build_ms=1.0),
                requested_output_points=100,
            )
            for protocol in ("cc_discharge", "drive_cycle")
        ]

        table = render_solver_table(results, width=240)

        assert "Protocol" in table
        assert "drive_cycle" in table
        payload = suite_to_jsonable("solver", results)
        assert [r["protocol"] for r in payload["results"]] == [
            "cc_discharge",
            "drive_cycle",
        ]

    def test_period_driven_grid_reports_no_point_count(self):
        results = [
            SolverResult(
                scenario="SPM",
                protocol="experiment",
                backend="rust_idaklu",
                timings=PhaseTiming(build_ms=1.0),
                requested_output_points=0,
            )
        ]

        # 0 means "the protocol owns its grid", which must not print as "0".
        assert " 0 " not in render_solver_table(results, width=240)
        assert "-" in render_solver_table(results, width=240)

    def test_inference_table_shows_one_time_and_per_eval_costs(self):
        results = [
            InferenceResult(
                scenario="SPM",
                protocol="cc_discharge",
                backend="casadi_idaklu",
                build_ms=10.0,
                setup_ms=5.0,
                aot_cache_status="-",
                eval_samples_ms=(1.0, 2.0, 3.0),
                solve_samples_ms=(0.8, 1.8, 2.8),
                observe_samples_ms=(0.2, 0.2, 0.2),
                requested_output_points=99,
            )
        ]

        table = render_inference_table(results, width=200)

        assert "Build" in table and "Setup" in table
        assert "Eval p50" in table
        assert "p10-p90" in table
        assert "baseline" in table
        payload = suite_to_jsonable("inference", results)
        encoded = json.loads(json.dumps(payload))
        assert encoded["results"][0]["eval_samples_ms"] == [1.0, 2.0, 3.0]
        # Medians are properties, so JSON carries the raw samples by design.
        assert "eval_median_ms" not in encoded["results"][0]

    def test_inference_table_handles_an_unsupported_row(self):
        results = [
            InferenceResult(
                scenario="SPM",
                protocol="experiment",
                backend="rust_diffsol",
                build_ms=0.0,
                setup_ms=0.0,
                aot_cache_status="-",
                eval_samples_ms=(),
                solve_samples_ms=(),
                observe_samples_ms=(),
                requested_output_points=0,
                supported=False,
                reason="SolverError: unsupported",
            )
        ]

        table = render_inference_table(results, width=200)

        assert "unsupported" in table
        assert "SolverError" in table

    def test_empty_inference_results_render_a_message(self):
        assert render_inference_table([]) == "No inference results."

    def test_inference_table_reports_coverage_and_gradient_parity(self):
        trajectory = TrajectorySummary(
            baseline_points=99,
            candidate_points=40,
            common_points=40,
            coverage=0.404,
            max_time_diff=0.0,
            final_time_diff=2100.0,
            baseline_termination="final time",
            candidate_termination="event",
            status="warn",
        )
        result = InferenceResult(
            scenario="SPM",
            protocol="cc_discharge",
            backend="rust_diffsol",
            build_ms=10.0,
            setup_ms=5.0,
            cold_observe_ms=7.5,
            aot_cache_status="-",
            eval_samples_ms=(1.0, 2.0, 3.0),
            solve_samples_ms=(0.8, 1.8, 2.8),
            observe_samples_ms=(0.2, 0.2, 0.2),
            requested_output_points=99,
            output_comparison=ComparisonSummary(1e-8, 1e-8, 0.01),
            sensitivity_comparison=ComparisonSummary(15.6, 28.0, 28.0),
            trajectory_comparison=trajectory,
        )

        table = render_inference_table([result], width=250)

        assert "ColdObs" in table and "7.50" in table
        assert "Cover" in table and "0.404" in table
        assert "Sens Δ" in table and "1.56e+01" in table
        # A clean value comparison must not read as an overall pass.
        assert result.status == "warn"

    def test_cold_observation_is_reported_separately_from_setup(self):
        result = InferenceResult(
            scenario="SPM",
            protocol="cc_discharge",
            backend="casadi_idaklu",
            build_ms=10.0,
            setup_ms=5.0,
            cold_observe_ms=9.15,
            aot_cache_status="-",
            eval_samples_ms=(1.0,),
            solve_samples_ms=(0.8,),
            observe_samples_ms=(0.2,),
            requested_output_points=99,
        )

        encoded = json.loads(json.dumps(suite_to_jsonable("inference", [result])))
        assert encoded["results"][0]["cold_observe_ms"] == 9.15
        assert encoded["results"][0]["setup_ms"] == 5.0


class TestReportAttribution:
    def _solver_rows(self, protocols):
        comparison = ComparisonSummary(1e-8, 1e-7, 0.01)
        trajectory = TrajectorySummary(
            baseline_points=100,
            candidate_points=100,
            common_points=100,
            coverage=1.0,
            max_time_diff=0.0,
            final_time_diff=0.0,
            baseline_termination="final time",
            candidate_termination="final time",
            status="pass",
        )
        return [
            SolverResult(
                scenario="SPM",
                protocol=protocol,
                backend="rust_idaklu",
                timings=PhaseTiming(),
                requested_output_points=100,
                state_comparison=comparison,
                output_comparison=comparison,
                trajectory_comparison=trajectory,
            )
            for protocol in protocols
        ]

    def test_compact_validation_table_carries_the_protocol(self):
        rows = self._solver_rows(("cc_discharge", "drive_cycle"))

        compact = render_solver_table(rows, width=120)
        validation = compact.split("\n\nValidation\n", maxsplit=1)[1]

        # Same scenario and backend on both rows, so without the protocol the
        # two validation lines are indistinguishable.
        assert "Protocol" in validation.splitlines()[0]
        assert "cc_discharge" in validation and "drive_cycle" in validation
        assert _fits(compact, 120)

    def test_aot_profile_carries_the_protocol(self):
        profile = AotProfile(
            fresh_cache_statuses=("miss",),
            disk_cache_statuses=("disk",),
            codegen_ms=1.0,
            compiler_ms=2.0,
            fresh_load_ms=1.0,
            disk_load_ms=1.0,
            fresh_total_ms=4.0,
            disk_total_ms=2.0,
            disk_prepare_ms=1.0,
            disk_cold_startup_ms=1.0,
            library_size_bytes=1024,
            verified=True,
        )
        rows = [
            SolverResult(
                scenario="SPM",
                protocol=protocol,
                backend="casadi_idaklu_aot",
                timings=PhaseTiming(),
                requested_output_points=100,
                aot_profile=profile,
            )
            for protocol in ("cc_discharge", "drive_cycle")
        ]

        for width in (200, 120, 80):
            report = render_solver_table(rows, width=width)
            aot = report.split("AOT profile", maxsplit=1)[1]
            assert "cc_discharge" in aot and "drive_cycle" in aot
            assert _fits(report, width)

    def test_sensitivity_rows_record_the_parameters_actually_differentiated(self):
        rows = [
            SensitivityResult(
                scenario="SPM",
                protocol="cc_discharge",
                backend="rust_idaklu",
                timings=PhaseTiming(),
                requested_output_points=100,
                sensitivity_parameters=("I", "eps_p"),
            ),
            SensitivityResult(
                scenario="SPM",
                protocol="drive_cycle",
                backend="rust_idaklu",
                timings=PhaseTiming(),
                requested_output_points=100,
                # An Interpolant current cannot be an input parameter.
                sensitivity_parameters=("eps_p",),
            ),
        ]

        for width in (250, 120, 80):
            table = render_sensitivity_table(rows, width=width)
            assert "I,eps_p" in table
            assert _fits(table, width)

        encoded = json.loads(json.dumps(suite_to_jsonable("sensitivity", rows)))
        assert encoded["results"][1]["sensitivity_parameters"] == ["eps_p"]

    def test_run_metadata_distinguishes_two_dirty_trees(self, monkeypatch):
        outputs = {"diff": "--- a\n+++ b\n+one implementation\n"}

        def fake_git(*arguments):
            if arguments[0] == "rev-parse":
                return "abc123\n"
            if arguments[0] == "status":
                return " M benchmarks/x.py\n"
            return outputs["diff"]

        monkeypatch.setattr(run_module, "_git", fake_git)
        first = run_module._git_metadata()
        outputs["diff"] = "--- a\n+++ b\n+a different implementation\n"
        second = run_module._git_metadata()

        assert first["git_revision"] == second["git_revision"]
        assert first["git_dirty"] is True
        # The revision alone cannot tell two local implementations apart.
        assert first["git_diff_digest"] != second["git_diff_digest"]

    def test_clean_tree_has_no_diff_digest(self, monkeypatch):
        def fake_git(*arguments):
            return "abc123\n" if arguments[0] == "rev-parse" else ""

        monkeypatch.setattr(run_module, "_git", fake_git)

        metadata = run_module._git_metadata()

        assert metadata["git_dirty"] is False
        assert metadata["git_diff_digest"] is None


class TestBackendOrdering:
    def test_baseline_is_shuffled_in_with_the_candidates(self):
        cases = backend_cases(include_aot=True)

        assert _BASELINE_CASE in cases
        # Pinning the baseline first measures it on a systematically colder
        # machine than everything it is compared against.
        orders = {
            _shuffled_backend_cases(
                0, f"solver:SPM:{protocol}", include_aot=True
            ).index(_BASELINE_CASE)
            for protocol in get_protocol_names()
        }
        assert orders != {0}

    def test_shuffle_key_separates_protocols_of_one_model(self):
        orders = {
            tuple(_shuffled_backend_cases(0, f"solver:SPM:{p}", include_aot=True))
            for p in get_protocol_names()
        }

        # One key per model gave every protocol the same order, so a slow first
        # slot always landed on the same backend.
        assert len(orders) > 1

    def test_shuffle_is_reproducible_for_a_fixed_seed(self):
        first = _shuffled_backend_cases(3, "solver:DFN:cc_charge", include_aot=False)
        again = _shuffled_backend_cases(3, "solver:DFN:cc_charge", include_aot=False)
        other = _shuffled_backend_cases(4, "solver:DFN:cc_charge", include_aot=False)

        assert first == again
        assert first != other
        assert sorted(first) == sorted(backend_cases(include_aot=False))


class TestConvergedReference:
    """The oracle every row is judged against, and its fallbacks."""

    def test_a_reference_no_tighter_than_the_scenario_is_rejected(self):
        scenarios = get_solver_scenarios(["SPM"])

        # 1e-6 scenario, so anything looser than 1e-8 measures two approximations
        # against each other -- the artifact the reference exists to remove.
        for tolerance in (1e-6, 1e-7):
            with pytest.raises(ValueError, match="decades tighter"):
                resolve_reference_tolerance(scenarios, tolerance)
        assert resolve_reference_tolerance(scenarios, 1e-8) == 1e-8
        assert resolve_reference_tolerance(scenarios, 1e-10) == 1e-10

    def test_zero_disables_the_reference(self):
        scenarios = get_solver_scenarios(["SPM"])

        assert resolve_reference_tolerance(scenarios, 0.0) is None
        assert resolve_reference_tolerance(scenarios, None) is None

    def test_ladder_loosens_a_decade_at_a_time_up_to_the_ceiling(self):
        scenario = get_solver_scenarios(["SPM"])[0]

        ladder = _reference_ladder(scenario, 1e-11)

        np.testing.assert_allclose(ladder, [1e-11, 1e-10, 1e-9, 1e-8], rtol=1e-12)

    def test_the_tightest_converging_tolerance_wins(self):
        scenario = get_solver_scenarios(["SPM"])[0]
        attempted = []

        def solve(tolerance):
            attempted.append(tolerance)
            # DFN under a ramping current behaves like this: converged only after
            # loosening, so a usable reference exists but not at the tightest rung.
            if tolerance < 1e-9:
                raise pybamm.SolverError("IDA_BAD_K")
            return "converged"

        used, value = _resolve_reference(scenario, 1e-11, solve, what="reference")

        assert (used, value) == (1e-9, "converged")
        np.testing.assert_allclose(attempted, [1e-11, 1e-10, 1e-9], rtol=1e-12)

    def test_an_unreachable_reference_degrades_rather_than_raising(self):
        scenario = get_solver_scenarios(["SPM"])[0]

        def solve(tolerance):
            raise pybamm.SolverError("IDA_BAD_K")

        assert _resolve_reference(scenario, 1e-10, solve, what="reference") == (
            None,
            None,
        )

    def test_a_failed_reference_keeps_the_gradient_gate(self, monkeypatch):
        # Status has to flip on backend correctness, not on whether the
        # reference happened to converge; the state gate is decades tighter.
        scenario = get_solver_scenarios(["SPM"], ["cc_discharge"])[0]
        _, gradient_rtol = _sensitivity_tolerances(scenario)
        captured: list[tuple[float | None, float]] = []

        def result(backend):
            return SensitivityResult(
                scenario=scenario.name,
                backend=backend,
                timings=PhaseTiming(),
                requested_output_points=scenario.plan.requested_points,
            )

        monkeypatch.setattr(
            runners,
            "_execute_backend_cases",
            lambda *a, **k: {
                runners._BASELINE_CASE: (result("casadi_idaklu"), None, None, None),
                ("rust_diffsol", False): (result("rust_diffsol"), None, None, None),
            },
        )
        monkeypatch.setattr(
            runners, "_sensitivity_reference", lambda *a, **k: (None, None)
        )

        def spy(*args, atol, rtol, **kwargs):
            captured.append((atol, rtol))
            return (None, None, None)

        monkeypatch.setattr(runners, "_compare_sensitivity_pair", spy)

        runners.run_sensitivity_lane(
            [scenario], repeats=1, warmup=0, reference_tolerance=None
        )

        assert captured == [(None, pytest.approx(gradient_rtol))]

    def test_reference_gate_is_looser_than_one_tolerance_unit(self):
        scenario = get_solver_scenarios(["SPM"])[0]

        atol, rtol = _reference_tolerances(scenario)

        # A tolerance bounds one step's local error; the global error accumulates
        # it, so a correct solve lands several tolerance units from the answer.
        assert atol > scenario.atol
        assert rtol > scenario.rtol

    def test_a_doomed_rung_retries_the_failing_draw_before_the_rest(self, monkeypatch):
        # Re-solving converged draws for a rung that cannot converge costs
        # rungs x repeats converged DFN solves instead of rungs + repeats.
        scenario = get_inference_scenarios(["SPM"], ["cc_charge"])[0]
        vectors = [{"eps_p": 0.60}, {"eps_p": 0.62}, {"eps_p": 0.64}, {"eps_p": 0.66}]
        attempted: list[dict] = []

        def solve(**kwargs):
            inputs = kwargs["inputs"]
            attempted.append(inputs)
            if inputs == vectors[3]:
                raise pybamm.SolverError("IDA_BAD_K")

        monkeypatch.setattr(runners, "_make_solver", lambda *a, **k: None)
        monkeypatch.setattr(
            runners, "_build_simulation", lambda *a, **k: SimpleNamespace(solve=solve)
        )
        monkeypatch.setattr(runners, "_build_and_time", lambda *a, **k: 0.0)
        monkeypatch.setattr(
            runners, "_observe_inference", lambda solution, inputs, **kwargs: inputs
        )

        used, _ = runners._inference_reference(
            scenario,
            vectors,
            warmup=1,
            grid=np.zeros(1),
            sensitivities=False,
            tolerance=1e-11,
        )

        assert used is None
        assert attempted == [vectors[1], vectors[2], vectors[3], *[vectors[3]] * 3]

    def test_inference_reference_fixes_y0_at_the_first_draw(self, monkeypatch):
        # The lane holds y0 at vectors[0]; a reference resolved from a later draw
        # starts the cell at a different SOC and moves the event by ~80 s.
        scenario = get_inference_scenarios(["SPM"], ["cc_charge"])[0]
        vectors = [{"eps_p": 0.60}, {"eps_p": 0.62}, {"eps_p": 0.64}, {"eps_p": 0.66}]
        built_with = []
        observed = []

        monkeypatch.setattr(runners, "_make_solver", lambda *a, **k: None)
        monkeypatch.setattr(
            runners, "_build_simulation", lambda *a, **k: SimpleNamespace(solve=dict)
        )
        monkeypatch.setattr(
            runners,
            "_build_and_time",
            lambda simulation, scenario, inputs=None: built_with.append(inputs) or 0.0,
        )
        monkeypatch.setattr(
            runners,
            "_observe_inference",
            lambda solution, inputs, **kwargs: observed.append(inputs),
        )

        used, _ = runners._inference_reference(
            scenario,
            vectors,
            warmup=2,
            grid=np.zeros(1),
            sensitivities=False,
            tolerance=1e-10,
        )

        assert used == 1e-10
        assert built_with == [vectors[0]]
        assert observed == vectors[2:]


class TestReferenceReporting:
    def _rows(self, reference_tolerance):
        comparison = ComparisonSummary(
            max_abs_diff=1e-6,
            max_rel_diff=1e-6,
            max_normalized_error=0.1,
        )
        return [
            SolverResult(
                scenario="SPM",
                backend=backend,
                timings=PhaseTiming(),
                requested_output_points=100,
                state_comparison=comparison,
                output_comparison=comparison,
                reference_tolerance=reference_tolerance,
                baseline_delta=None if backend == "casadi_idaklu" else comparison,
            )
            for backend in ("casadi_idaklu", "rust_idaklu")
        ]

    def test_the_baseline_row_is_gated_once_a_reference_exists(self):
        gated, ungated = (
            self._rows(1e-10)[0],
            SolverResult(
                scenario="SPM",
                backend="casadi_idaklu",
                timings=PhaseTiming(),
                requested_output_points=100,
            ),
        )

        assert gated.status == "pass"
        assert ungated.status == "baseline"

    def test_the_table_names_what_the_deltas_are_measured_against(self):
        with_reference = render_solver_table(self._rows(1e-10), width=200)
        without = render_solver_table(self._rows(None), width=200)

        assert "converged casadi_idaklu reference at atol=rtol=1e-10" in with_reference
        assert "no converged reference was run" in without

    def test_every_comparison_lane_carries_the_reference_columns(self):
        comparison = ComparisonSummary(
            max_abs_diff=1e-6,
            max_rel_diff=1e-6,
            max_normalized_error=0.1,
        )
        sensitivity = render_sensitivity_table(
            [
                SensitivityResult(
                    scenario="SPM",
                    backend="rust_diffsol",
                    timings=PhaseTiming(),
                    requested_output_points=100,
                    output_sens_comparison=comparison,
                    reference_tolerance=1e-10,
                    baseline_delta=comparison,
                )
            ],
            width=200,
        )
        inference = render_inference_table(
            [
                InferenceResult(
                    scenario="SPM",
                    protocol="cc_discharge",
                    backend="rust_diffsol",
                    build_ms=1.0,
                    setup_ms=1.0,
                    aot_cache_status="-",
                    eval_samples_ms=(1.0,),
                    solve_samples_ms=(1.0,),
                    observe_samples_ms=(1.0,),
                    requested_output_points=100,
                    output_comparison=comparison,
                    reference_tolerance=1e-10,
                    baseline_delta=comparison,
                )
            ],
            width=200,
        )

        for table in (sensitivity, inference):
            assert "converged casadi_idaklu reference at atol=rtol=1e-10" in table
            assert "Base Δ" in table

    def test_the_cross_backend_delta_is_reported_beside_the_reference_error(self):
        table = render_solver_table(self._rows(1e-10), width=200)
        payload = suite_to_jsonable("solver", self._rows(1e-10))

        assert "Base Δ" in table
        assert payload["results"][0]["baseline_delta"] is None
        assert payload["results"][1]["baseline_delta"]["max_abs_diff"] == 1e-6
        assert payload["results"][1]["reference_tolerance"] == 1e-10


class TestCli:
    def test_reference_tolerance_flag(self):
        parser = build_parser()

        assert parser.parse_args([]).reference_tolerance == DEFAULT_REFERENCE_TOLERANCE
        assert (
            parser.parse_args(["--reference-tolerance", "0"]).reference_tolerance == 0
        )

    def test_lane_and_protocol_flags(self):
        parser = build_parser()

        args = parser.parse_args(
            ["--lane", "inference", "--protocols", "drive_cycle", "pulse_train"]
        )

        assert args.lane == "inference"
        assert args.protocols == ["drive_cycle", "pulse_train"]
        assert args.aot == "solver"
        assert args.inference_sensitivities is False
        assert args.inference_seed == 0

    def test_default_protocols_preserve_the_baseline_run(self):
        args = build_parser().parse_args([])

        assert args.protocols == ["cc_discharge"]
        assert args.lane == "all"

    def test_an_empty_protocol_list_means_all_of_them_as_it_does_for_models(self):
        # `--protocols` and `--models` are parallel flags, so a bare one has to
        # mean the same thing on both.
        every_protocol = get_protocol_names()
        selected = {scenario.protocol for scenario in get_solver_scenarios(["SPM"], [])}
        assert selected == set(every_protocol)
        assert [
            scenario.protocol for scenario in get_solver_scenarios(["SPM"], None)
        ] == ["cc_discharge"]

    def test_aot_choices(self):
        parser = build_parser()

        assert parser.parse_args(["--aot", "none"]).aot == "none"
        assert parser.parse_args(["--aot", "all"]).aot == "all"
        with pytest.raises(SystemExit):
            parser.parse_args(["--aot", "sometimes"])

    def test_include_aot_resolution(self):
        assert include_aot_for("solver", "solver") is True
        assert include_aot_for("sensitivity", "solver") is False
        assert include_aot_for("sensitivity", "all") is True
        assert include_aot_for("solver", "all") is True
        assert include_aot_for("solver", "none") is False
        assert include_aot_for("sensitivity", "none") is False

    def test_removed_diffsol_flag_is_gone(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--include-diffsol"])
