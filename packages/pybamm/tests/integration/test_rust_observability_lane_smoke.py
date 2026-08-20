from __future__ import annotations

import pytest

import pybamm
from benchmarks.run_rust_observability import build_parser, repeats_for
from benchmarks.rust_observability.registry import (
    INFERENCE_INPUTS,
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
    run_inference_lane,
    run_sensitivity_lane,
    run_solver_lane,
    sample_input_vectors,
)

# Enough points for the observation grid to interpolate, small enough to run
# every protocol on every lane.
SMOKE_POINTS = 20
PROTOCOLS = get_protocol_names()


def assert_renders(lane: str, results, renderer):
    """Every row must render and serialise, whatever its status."""
    assert results
    for width in (250, 120, 80):
        table = renderer(results, width=width)
        assert _fits(table, width)
    payload = suite_to_jsonable(lane, results)
    assert len(payload["results"]) == len(results)
    return {result.backend: result.status for result in results}


@pytest.mark.parametrize("protocol", PROTOCOLS)
class TestLanesRunForEveryProtocol:
    """Wiring a protocol into the registry does not prove a lane can run it.

    Each lane is exercised end to end so a protocol that builds but cannot be
    solved, observed, differentiated or rendered fails here rather than in a
    benchmarking session.
    """

    def test_solver_lane(self, protocol):
        results = run_solver_lane(
            get_solver_scenarios(["SPM"], [protocol], output_points=SMOKE_POINTS),
            repeats=1,
            warmup=0,
            include_aot=False,
        )
        statuses = assert_renders("solver", results, render_solver_table)

        # Gated like any other row now that it is scored against the reference.
        assert statuses["casadi_idaklu"] in {"pass", "warn"}
        assert all(result.reference_tolerance for result in results if result.supported)
        assert any(backend.startswith("rust_") for backend in statuses)
        for result in results:
            if result.supported and result.trajectory_comparison is not None:
                assert result.trajectory_comparison.coverage > 0.0

    def test_sensitivity_lane(self, protocol):
        results = run_sensitivity_lane(
            get_solver_scenarios(["SPM"], [protocol], output_points=SMOKE_POINTS),
            repeats=1,
            warmup=0,
            include_aot=False,
        )
        statuses = assert_renders("sensitivity", results, render_sensitivity_table)

        assert statuses["casadi_idaklu"] in {"pass", "warn"}
        # An Interpolant or Experiment current cannot be a sensitivity input.
        expected = (
            {"eps_p"}
            if protocol in {"drive_cycle", "pulse_train", "experiment"}
            else {"I", "eps_p"}
        )
        for result in results:
            if result.supported:
                assert set(result.sensitivity_parameters) == expected

    def test_inference_lane(self, protocol):
        results = run_inference_lane(
            get_inference_scenarios(["SPM"], [protocol], output_points=SMOKE_POINTS),
            repeats=2,
            warmup=0,
            include_aot=False,
        )
        statuses = assert_renders("inference", results, render_inference_table)

        assert statuses["casadi_idaklu"] in {"pass", "warn"}
        for result in results:
            if result.supported:
                assert result.output_comparison is not None
                assert result.trajectory_comparison is not None
                assert result.sensitivity_comparison is None
                # The baseline has nothing to differ from; every other row does.
                is_baseline = result.backend == "casadi_idaklu"
                assert (result.baseline_delta is None) is is_baseline
            if result.supported:
                assert result.cold_observe_ms > 0.0
                assert len(result.eval_samples_ms) == 2


class TestDefaultSamplingDepthIsFeasible:
    """The lane draws `warmup + repeats` vectors, so a hazard can hide past a short draw.

    A fitted maximum concentration made the eleventh draw start the cell above its
    voltage cutoff, which every smoke test at two or four vectors missed. The depth
    is read from the lane's own default rather than pinned here.
    """

    def test_every_default_depth_draw_solves(self):
        defaults = build_parser().parse_args([])
        results = run_inference_lane(
            get_inference_scenarios(
                ["SPM"], ["cc_discharge"], output_points=SMOKE_POINTS
            ),
            repeats=repeats_for("inference", defaults.repeats),
            warmup=defaults.warmup,
            seed=defaults.inference_seed,
            include_aot=False,
        )

        unsupported = {r.backend: r.reason for r in results if not r.supported}
        assert not unsupported, f"default-depth draws are infeasible: {unsupported}"

    def test_no_fitted_parameter_defines_the_initial_state(self):
        # The base set fixes the initial concentration absolutely, so fitting a
        # maximum concentration moves the stoichiometry rather than the capacity.
        assert not [
            name
            for name in INFERENCE_INPUTS
            if name.startswith("Maximum concentration")
        ]

    def test_sampled_draws_keep_the_initial_stoichiometry_physical(self):
        defaults = build_parser().parse_args([])
        scenario = get_inference_scenarios(["SPM"], ["cc_discharge"])[0]
        vectors = sample_input_vectors(
            inference_nominal_values(),
            defaults.warmup + repeats_for("inference", defaults.repeats),
            seed=defaults.inference_seed,
        )
        built = scenario.parameter_values_builder()

        for electrode in ("negative", "positive"):
            initial = built[f"Initial concentration in {electrode} electrode [mol.m-3]"]
            maximum = built[f"Maximum concentration in {electrode} electrode [mol.m-3]"]
            for vector in vectors:
                stoichiometry = pybamm.Scalar(1) * initial / maximum
                value = float(stoichiometry.evaluate(inputs=vector))
                assert 0.0 < value < 1.0, (
                    f"{electrode} stoichiometry {value:.4f} is outside [0, 1] "
                    f"for inputs {vector}"
                )


class TestInferenceGradientsAreValidated:
    def test_requesting_sensitivities_compares_them(self):
        results = run_inference_lane(
            get_inference_scenarios(
                ["SPM"], ["cc_discharge"], output_points=SMOKE_POINTS
            ),
            repeats=2,
            warmup=0,
            sensitivities=True,
            include_aot=False,
        )

        candidates = [
            r for r in results if r.backend != "casadi_idaklu" and r.supported
        ]
        assert candidates
        for result in candidates:
            assert result.sensitivity_comparison is not None

    def test_gradients_cost_more_than_values_alone(self):
        common = {
            "scenarios": get_inference_scenarios(
                ["SPM"], ["cc_discharge"], output_points=SMOKE_POINTS
            ),
            "repeats": 2,
            "warmup": 1,
            "include_aot": False,
        }
        values_only = run_inference_lane(**common, sensitivities=False)
        with_gradients = run_inference_lane(**common, sensitivities=True)

        def baseline(results):
            return next(r for r in results if r.backend == "casadi_idaklu")

        # Materialising the chain rule is real work; if it were free the lane
        # would not be observing the gradient at all.
        assert (
            baseline(with_gradients).observe_median_ms
            > baseline(values_only).observe_median_ms
        )
