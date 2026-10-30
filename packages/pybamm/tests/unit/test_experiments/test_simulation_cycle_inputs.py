from datetime import datetime

import numpy as np
import pytest

import pybamm

DIFFUSIVITY = "Negative particle diffusivity [m2.s-1]"


def _parameter_values():
    parameter_values = pybamm.ParameterValues("Chen2020")
    initial = parameter_values[DIFFUSIVITY]
    parameter_values[DIFFUSIVITY] = "[input]"
    return parameter_values, initial


def _simulation(cycle, n_cycles=3, **kwargs):
    parameter_values, initial = _parameter_values()
    sim = pybamm.Simulation(
        pybamm.lithium_ion.SPM(),
        parameter_values=parameter_values,
        experiment=pybamm.Experiment([cycle] * n_cycles),
        **kwargs,
    )
    return sim, initial


CYCLE = ("Discharge at 1C for 10 minutes", "Rest for 5 minutes")


def _double(cycle_number, cycle_solution, inputs):
    return {DIFFUSIVITY: inputs[DIFFUSIVITY] * 2}


def _cycle_diffusivity(cycle):
    return float(np.asarray(cycle.steps[0].all_inputs[0][DIFFUSIVITY]).item())


class TestSimulationCycleInputs:
    def test_updates_apply_from_next_cycle(self):
        sim, initial = _simulation(CYCLE)
        calls = []

        def cycle_inputs(cycle_number, cycle_solution, inputs):
            calls.append((cycle_number, inputs[DIFFUSIVITY]))
            return _double(cycle_number, cycle_solution, inputs)

        sol = sim.solve(inputs={DIFFUSIVITY: initial}, cycle_inputs=cycle_inputs)

        # Not called after the last cycle
        assert [n for n, _ in calls] == [1, 2]
        np.testing.assert_allclose([d for _, d in calls], [initial, 2 * initial])
        np.testing.assert_allclose(
            [_cycle_diffusivity(cycle) for cycle in sol.cycles],
            [initial, 2 * initial, 4 * initial],
        )

    def test_none_keeps_inputs(self):
        sim, initial = _simulation(CYCLE)
        sol = sim.solve(inputs={DIFFUSIVITY: initial}, cycle_inputs=lambda *args: None)
        np.testing.assert_allclose(
            [_cycle_diffusivity(cycle) for cycle in sol.cycles], [initial] * 3
        )

    def test_matches_starting_solution_continuation(self):
        sim, initial = _simulation(CYCLE, n_cycles=2)
        sol = sim.solve(inputs={DIFFUSIVITY: initial}, cycle_inputs=_double)

        single, _ = _simulation(CYCLE, n_cycles=1)
        first = single.solve(inputs={DIFFUSIVITY: initial})
        continued = single.solve(
            starting_solution=first, inputs={DIFFUSIVITY: 2 * initial}
        )
        np.testing.assert_allclose(
            sol.cycles[1]["Voltage [V]"].entries,
            continued.cycles[1]["Voltage [V]"].entries,
            rtol=1e-6,
        )

    def test_not_called_for_starting_solution_cycles(self):
        sim, initial = _simulation(CYCLE, n_cycles=2)
        first = sim.solve(inputs={DIFFUSIVITY: initial})
        calls = []

        def cycle_inputs(cycle_number, cycle_solution, inputs):
            calls.append(cycle_number)

        sim.solve(
            starting_solution=first,
            inputs={DIFFUSIVITY: initial},
            cycle_inputs=cycle_inputs,
        )
        assert calls == [3]

    def test_sees_cycles_not_saved(self):
        sim, initial = _simulation(CYCLE, n_cycles=4)
        seen = {}

        def cycle_inputs(cycle_number, cycle_solution, inputs):
            seen[cycle_number] = cycle_solution

        sol = sim.solve(
            inputs={DIFFUSIVITY: initial}, cycle_inputs=cycle_inputs, save_at_cycles=[3]
        )
        assert [cycle is None for cycle in sol.cycles] == [False, True, False, False]
        cycle_2 = seen[2]
        assert len(cycle_2.steps) == 2
        assert len(cycle_2.t) > 2
        assert cycle_2.cycle_summary_variables["Capacity [A.h]"] > 0

    def test_skipped_step_keeps_its_index(self):
        cycle = (
            "Discharge at 1C for 10 minutes",
            "Discharge at 1C until 4.5 V",
            "Rest for 5 minutes",
        )
        sim, initial = _simulation(cycle, n_cycles=2)
        seen = []

        def cycle_inputs(cycle_number, cycle_solution, inputs):
            seen.append(cycle_solution.steps)

        sim.solve(inputs={DIFFUSIVITY: initial}, cycle_inputs=cycle_inputs)
        steps = seen[0]
        assert len(steps) == 3
        assert isinstance(steps[1], pybamm.EmptySolution)
        assert not isinstance(steps[2], pybamm.EmptySolution)

    def test_padding_rest_joins_its_step(self):
        parameter_values, initial = _parameter_values()
        experiment = pybamm.Experiment(
            [
                (
                    pybamm.step.string(
                        "Discharge at 0.5C for 10 minutes",
                        start_time=datetime(2023, 1, 1, 8, 0, 0),
                    ),
                    pybamm.step.string(
                        "Rest for 10 minutes", start_time=datetime(2023, 1, 1, 9, 0, 0)
                    ),
                ),
                ("Rest for 10 minutes",),
            ]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            parameter_values=parameter_values,
            experiment=experiment,
        )
        seen = []

        def cycle_inputs(cycle_number, cycle_solution, inputs):
            seen.append(cycle_solution.steps)

        sim.solve(inputs={DIFFUSIVITY: initial}, cycle_inputs=cycle_inputs)
        steps = seen[0]
        assert len(steps) == 2
        assert steps[0].t[-1] == pytest.approx(3600)

    @pytest.mark.parametrize("compile_model", [False, True])
    def test_unified_matches_legacy(self, compile_model):
        solutions = []
        for mode in ["legacy", "unified"]:
            sim, initial = _simulation(
                CYCLE,
                experiment_model_mode=mode,
                solver=pybamm.IDAKLUSolver(options={"compile": compile_model}),
            )
            solutions.append(
                sim.solve(inputs={DIFFUSIVITY: initial}, cycle_inputs=_double)
            )
        legacy, unified = solutions
        np.testing.assert_allclose(
            [_cycle_diffusivity(cycle) for cycle in unified.cycles],
            [_cycle_diffusivity(cycle) for cycle in legacy.cycles],
        )
        np.testing.assert_allclose(
            unified["Voltage [V]"](legacy.t), legacy["Voltage [V]"].entries, rtol=1e-5
        )

    def test_errors(self):
        sim, initial = _simulation(CYCLE, n_cycles=2)
        inputs = {DIFFUSIVITY: initial}

        with pytest.raises(TypeError, match="must be callable"):
            sim.solve(inputs=inputs, cycle_inputs={DIFFUSIVITY: initial})
        with pytest.raises(TypeError, match="must return a dict"):
            sim.solve(inputs=inputs, cycle_inputs=lambda *args: [initial])
        with pytest.raises(ValueError, match="not one of the inputs"):
            sim.solve(inputs=inputs, cycle_inputs=lambda *args: {"Other": 1.0})
        with pytest.raises(ValueError, match="not one of the inputs"):
            sim.solve(inputs=inputs, cycle_inputs=lambda *args: {"start time": 0.0})
        with pytest.raises(ValueError, match="shape"):
            sim.solve(
                inputs=inputs,
                cycle_inputs=lambda *args: {DIFFUSIVITY: np.array([initial, initial])},
            )
        sim, _ = _simulation(CYCLE, n_cycles=2)
        with pytest.raises(ValueError, match="sensitivities"):
            sim.solve(
                inputs=inputs,
                cycle_inputs=_double,
                calculate_sensitivities=[DIFFUSIVITY],
            )

        no_experiment = pybamm.Simulation(pybamm.lithium_ion.SPM())
        with pytest.raises(ValueError, match="only be provided if simulating"):
            no_experiment.solve([0, 600], cycle_inputs=_double)
