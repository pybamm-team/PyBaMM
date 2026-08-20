from __future__ import annotations

import numpy as np
import pytest

from benchmarks.rust_observability.registry import (
    INFERENCE_INPUTS,
    get_inference_scenarios,
    get_protocol_names,
    inference_nominal_values,
)
from benchmarks.rust_observability.runners import (
    _build_and_time,
    _build_simulation,
    _make_solver,
    _solve_kwargs,
)

_BACKEND = "casadi_idaklu"


def _scenarios():
    """Every model x protocol the inference lane actually runs."""
    return get_inference_scenarios(
        ["SPM", "SPMe", "DFN"], get_protocol_names(), output_points=50
    )


class TestInferenceParametersAreLive:
    @pytest.mark.parametrize(
        "scenario", _scenarios(), ids=lambda s: f"{s.name}-{s.protocol}"
    )
    def test_every_fitted_parameter_moves_the_voltage(self, scenario):
        """A dead parameter would inflate the count without perturbing the graph."""
        # INFERENCE_COMPLEMENTS makes dV/d(eps) a different quantity, so the
        # lane's own builders have to be what is measured.
        nominal = inference_nominal_values()
        solver = _make_solver(_BACKEND, atol=scenario.atol, rtol=scenario.rtol)
        simulation = _build_simulation(scenario, solver, _BACKEND)
        _build_and_time(simulation, scenario, inputs=nominal)
        names = sorted(INFERENCE_INPUTS.values())
        solution = simulation.solve(
            **_solve_kwargs(
                scenario,
                {"inputs": nominal, "calculate_sensitivities": names},
            )
        )

        sensitivities = solution["Voltage [V]"].sensitivities
        for input_name in names:
            magnitude = float(np.max(np.abs(sensitivities[input_name])))
            assert magnitude > 0.0, (
                f"{input_name} has zero sensitivity on {scenario.name} under "
                f"{scenario.protocol}; it is not used by this model and would "
                "be a dead fitted parameter"
            )
