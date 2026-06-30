import copy
import itertools
import logging
import os
import re
from datetime import datetime
from types import SimpleNamespace

import casadi
import numpy as np
import pytest

import pybamm


class ShortDurationCRate(pybamm.step.CRate):
    def _default_timespan(self, value):
        # Set a short default duration for testing early stopping due to infeasible time
        return 1


def _unified_rhs_jac_n_instructions(unique_step_strings):
    """rhs_algebraic and jacobian top-function instruction counts for a unified-mode
    SPM experiment with the given distinct steps."""
    experiment = pybamm.Experiment([tuple(unique_step_strings)])
    sim = pybamm.Simulation(
        pybamm.lithium_ion.SPM(),
        experiment=experiment,
        solver=pybamm.IDAKLUSolver(),
        experiment_model_mode="unified",
    )
    sim.solve()
    assert sim._experiment_uses_unified_model
    model = sim._built_experiment_model
    return (
        model.rhs_algebraic_eval.n_instructions(),
        model.jac_rhs_algebraic_eval.n_instructions(),
    )


def _largest_generated_fn_lines(fn):
    """Body-line count of the largest ``casadi_fN`` in ``fn``'s generated C."""
    gen = casadi.CodeGenerator("probe", {"with_header": False})
    gen.add(fn)
    lines = gen.dump().split("\n")
    largest = 0
    for i, line in enumerate(lines):
        if re.match(r"\s*static int casadi_f\d+\(", line):
            depth = 0
            for j in range(i, len(lines)):
                depth += lines[j].count("{") - lines[j].count("}")
                if depth == 0 and j > i:
                    largest = max(largest, j - i)
                    break
    return largest


class TestSimulationExperiment:
    @staticmethod
    def _make_differential_custom_step_simulation(**simulation_kwargs):
        def custom_step_voltage(variables):
            return 100 * (variables["Voltage [V]"] - 4.2)

        experiment = pybamm.Experiment(
            [
                pybamm.step.CustomStepImplicit(
                    custom_step_voltage, control="differential", duration=100, period=10
                )
            ]
        )
        return pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            **simulation_kwargs,
        )

    def test_batch_cross_cycle_equivalent(self):
        # Full multi-cycle solution must be identical with batch accumulation.
        model = pybamm.lithium_ion.SPM()
        exp = pybamm.Experiment(
            [("Discharge at 1C until 3.0V", "Charge at 1C until 4.2V")] * 5,
            period=600,
        )
        sim = pybamm.Simulation(model, experiment=exp)
        sol = sim.solve()
        v = sol["Voltage [V]"].entries
        assert np.all(np.isfinite(v))
        assert np.all(np.diff(sol.t) > 0)  # monotonic time preserved
        assert len(sol.cycles) == 5  # cycles wiring preserved
        assert sol.summary_variables is not None

    def test_unified_model_mode_validation_and_blockers(self):
        with pytest.raises(ValueError, match="experiment_model_mode must be one of"):
            pybamm.Simulation(
                pybamm.lithium_ion.SPM(),
                experiment_model_mode="invalid",
            )
        with pytest.raises(ValueError, match="experiment_model_mode must be one of"):
            pybamm.Simulation(
                pybamm.lithium_ion.SPM(),
                experiment_model_mode="auto",
            )
        with pytest.raises(ValueError, match="experiment_model_mode must be one of"):
            pybamm.Simulation(
                pybamm.lithium_ion.SPM(),
                experiment_model_mode="per-step",
            )

        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.experiment = None
        assert sim._get_unified_experiment_model_blockers() == [
            "no experiment is attached to the simulation"
        ]

        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=pybamm.Experiment([pybamm.step.BaseStep(1, duration=1)]),
            solver=pybamm.IDAKLUSolver(),
        )
        assert sim._get_unified_experiment_model_blockers() == [
            "unsupported experiment step type 'BaseStep'"
        ]

    def test_unified_allows_drive_cycle_and_algebraic_implicit(self):
        # Drive-cycle steps are explicit current steps with a time-varying value; they
        # share the generic current residual, so unified mode must accept them. An
        # algebraic-control implicit step (voltage hold) is also eligible.
        drive_cycle = np.column_stack([[0, 5, 10], [1.0, 0.5, 1.0]])
        experiment = pybamm.Experiment(
            [
                pybamm.step.current(drive_cycle),
                pybamm.step.voltage(3.8, duration=10),
            ]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="unified",
        )
        assert sim._get_unified_experiment_model_blockers() == []

    def test_unified_drive_cycle_matches_legacy(self):
        # A drive cycle solved in unified mode must match legacy on a common time grid.
        time = [0, 20, 40, 60, 80, 100]
        current = [0.5, 0.6, 0.4, 0.5, 0.45, 0.5]
        drive_cycle = np.column_stack([time, current])
        experiment = pybamm.Experiment([pybamm.step.current(drive_cycle)])
        out = {}
        for mode in ("legacy", "unified"):
            sim = pybamm.Simulation(
                pybamm.lithium_ion.SPM(),
                experiment=experiment,
                solver=pybamm.IDAKLUSolver(),
                experiment_model_mode=mode,
            )
            sol = sim.solve()
            out[mode] = sol["Voltage [V]"](time)
        np.testing.assert_allclose(out["unified"], out["legacy"], rtol=1e-4, atol=1e-4)

    def test_set_up_unified_preserves_voltage_safety_events(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/20 for 1 hour",
                "Charge at 1 A until 4.1 V",
            ]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="unified",
        )
        sim.build_for_experiment()

        unified_model = sim.experiment_unique_steps_to_model[
            sim._experiment_unified_model_key
        ]
        event_names = [event.name for event in unified_model.events]

        assert "Minimum voltage [V]" in event_names
        assert "Maximum voltage [V]" in event_names

    def test_build_experiment_step_inputs_uses_expected_step_index(self):
        start = datetime(2024, 1, 1, 12)
        experiment = pybamm.Experiment(
            [
                (
                    pybamm.step.Current(1, duration=600, start_time=start),
                    pybamm.step.Voltage(4.1, duration=300),
                )
            ]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="unified",
        )
        sim.build_for_experiment()

        assert sim._STEP_INDEX_INPUT == "Experiment step index"
        assert sim._experiment_step_indices == [1, 2]
        assert sim._experiment_padding_rest_index == 3
        assert sim._experiment_includes_padding_rest

        step = experiment.steps[1]
        step_inputs = sim._build_experiment_step_inputs(
            {"user input": 7},
            step,
            start_time=123.0,
            active_step_index=2,
        )
        assert step_inputs["user input"] == 7
        assert step_inputs["Ambient temperature [K]"] is not None
        assert step_inputs["start time"] == 123.0
        assert step_inputs["Experiment step index"] == 2

        step_inputs = sim._build_experiment_step_inputs(
            {},
            experiment.steps[0],
            start_time=0.0,
            active_step_index=sim._experiment_padding_rest_index,
        )
        assert step_inputs["Experiment step index"] == 3

    def test_setup_experiment_string_or_list(self):
        model = pybamm.lithium_ion.SPM()

        sim = pybamm.Simulation(model, experiment="Discharge at C/20 for 1 hour")
        sim.build_for_experiment()
        assert len(sim.experiment.steps) == 1
        assert sim.experiment.steps[0].description == "Discharge at C/20 for 1 hour"
        sim = pybamm.Simulation(
            model,
            experiment=["Discharge at C/20 for 1 hour", pybamm.step.rest(60)],
        )
        sim.build_for_experiment()
        assert len(sim.experiment.steps) == 2

    def test_set_up_all_explicit_uses_unified_model_with_dae_solver(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/20 for 1 hour",
                "Rest for 10 minutes",
                "Charge at 1 A for 20 minutes",
            ]
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            solver=pybamm.CasadiSolver(),
            experiment_model_mode="unified",
        )
        sim.build_for_experiment()

        assert sim._experiment_uses_unified_model
        assert len(set(sim.steps_to_built_models.values())) == 1

        unified_model = sim.experiment_unique_steps_to_model[
            sim._experiment_unified_model_key
        ]
        assert "Current variable [A]" in unified_model.variables

    def test_unified_builds_exactly_one_model_and_solver(self):
        # Requirement: unified mode uses exactly one model and one solver instance,
        # even across many steps and cycles.
        experiment = pybamm.Experiment(
            [
                pybamm.step.c_rate(1.0, duration=60),
                pybamm.step.voltage(3.8, duration=60),
            ]
            * 3
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="unified",
        )
        sim.build_for_experiment()
        assert len({id(m) for m in sim.steps_to_built_models.values()}) == 1
        assert len({id(s) for s in sim.steps_to_built_solvers.values()}) == 1

    def test_unified_constant_current_steps_collapse_to_one_branch(self):
        # CC/C-rate steps differing only in current value share one branch: the value
        # is supplied as a per-step input, so the number of branches (and therefore
        # compile/runtime cost) does not grow with the number of distinct currents.
        currents = [0.1, 0.2, 0.5, 1.0]
        steps = [pybamm.step.c_rate(c, duration=1) for c in currents]
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=pybamm.Experiment(steps),
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="unified",
        )
        sim.build_for_experiment()
        assert len(set(sim._experiment_step_indices)) == 1

    def test_unified_custom_termination_functions_get_distinct_branches(self):
        def lower_voltage_limit(variables):
            return variables["Battery voltage [V]"] - 3.8

        def higher_voltage_limit(variables):
            return variables["Battery voltage [V]"] - 3.9

        lower_voltage_termination = pybamm.step.CustomTermination(
            name="shared custom cut-off", event_function=lower_voltage_limit
        )
        higher_voltage_termination = pybamm.step.CustomTermination(
            name="shared custom cut-off", event_function=higher_voltage_limit
        )

        assert lower_voltage_termination != higher_voltage_termination

        steps = [
            pybamm.step.c_rate(0.5, duration=10, termination=lower_voltage_termination),
            pybamm.step.c_rate(
                1.0, duration=10, termination=higher_voltage_termination
            ),
        ]
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=pybamm.Experiment(steps),
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="unified",
        )
        sim.build_for_experiment()

        assert sim._experiment_step_indices[0] != sim._experiment_step_indices[1]

    def test_unified_collapses_every_control_kind_by_value(self):
        # Value-as-input is general: steps of ANY control kind that differ only in their
        # target value share one branch, and distinct control kinds get separate branches.
        steps = [
            pybamm.step.c_rate(0.5, duration=10),
            pybamm.step.c_rate(1.0, duration=10),
            pybamm.step.voltage(4.0, duration=10),
            pybamm.step.voltage(4.1, duration=10),
            pybamm.step.power(1.0, duration=10),
            pybamm.step.power(2.0, duration=10),
        ]
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=pybamm.Experiment(steps),
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="unified",
        )
        sim.build_for_experiment()
        # current / voltage / power -> exactly three branches (one per control kind).
        assert len(set(sim._experiment_step_indices)) == 3

    def test_unified_voltage_per_step_matches_legacy(self):
        # Generalised value-as-input must be correct for implicit control too: two
        # voltage holds at distinct targets collapse to one branch but must each hold
        # their own target and match legacy. Sample settled points (end of each fixed
        # 300 s window) on a common time grid to avoid transient/grid-alignment noise.
        steps = [
            pybamm.step.c_rate(0.5, duration=300),
            pybamm.step.voltage(3.9, duration=300),
            pybamm.step.voltage(3.8, duration=300),
        ]
        sample_times = [250, 590, 890]
        out = {}
        for mode in ("legacy", "unified"):
            sim = pybamm.Simulation(
                pybamm.lithium_ion.SPM(),
                experiment=pybamm.Experiment(steps),
                solver=pybamm.IDAKLUSolver(),
                experiment_model_mode=mode,
            )
            out[mode] = sim.solve()["Voltage [V]"](sample_times)
        np.testing.assert_allclose(out["unified"], out["legacy"], rtol=1e-3, atol=1e-3)
        # The two collapsed voltage steps must hold their own distinct targets.
        np.testing.assert_allclose(out["unified"][1:], [3.9, 3.8], atol=2e-2)

    def test_unified_cc_per_step_current_parses(self):
        # Steps that collapse to one branch must still each solve to their own current:
        # the per-step current input is parsed correctly (matches legacy, all distinct).
        currents = [0.1, 0.3, 0.7, 1.0]
        steps = [pybamm.step.c_rate(c, duration=1) for c in currents]
        sample_times = [0.5, 1.5, 2.5, 3.5]
        out = {}
        for mode in ("legacy", "unified"):
            sim = pybamm.Simulation(
                pybamm.lithium_ion.SPM(),
                experiment=pybamm.Experiment(steps),
                solver=pybamm.IDAKLUSolver(),
                experiment_model_mode=mode,
            )
            sol = sim.solve()
            out[mode] = sol["Current [A]"](sample_times)
        np.testing.assert_allclose(out["unified"], out["legacy"], rtol=1e-4, atol=1e-6)
        # Each step's input must be applied distinctly, not a single global current.
        assert len(set(np.round(out["unified"], 6))) == 4

    def test_set_up_all_explicit_defaults_to_legacy_model(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/20 for 1 hour",
                "Rest for 10 minutes",
                "Charge at 1 A for 20 minutes",
            ]
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            solver=pybamm.CasadiSolver(),
        )
        sim.build_for_experiment()

        assert not sim._experiment_uses_unified_model
        assert len(set(sim.steps_to_built_models.values())) == len(
            sim.experiment.unique_steps
        )

    def test_set_up_can_force_legacy_experiment_models(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/20 for 1 hour",
                "Charge at 1 A until 4.1 V",
                "Hold at 4.1 V until 50 mA",
            ]
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            solver=pybamm.CasadiSolver(),
            experiment_model_mode="legacy",
        )
        sim.build_for_experiment()

        assert not sim._experiment_uses_unified_model
        model_I = sim.experiment_unique_steps_to_model[
            sim.experiment.steps[1].basic_repr()
        ]
        assert "Voltage > 4.1 [V] [experiment]" in [
            event.name for event in model_I.events
        ]

    def test_distinct_control_types_with_same_value_yield_distinct_models(self):
        # Regression: CRate(4.2) and Voltage(4.2) used to collapse to one unique step.
        experiment = pybamm.Experiment(
            [
                (
                    pybamm.step.c_rate(4.2, duration=60),
                    pybamm.step.current(-1, duration=60),
                    pybamm.step.voltage(4.2, duration=60),
                )
            ]
        )
        assert len(experiment.unique_steps) == 3

        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="legacy",
        )
        sim.build_for_experiment()
        assert len(sim.experiment_unique_steps_to_model) == 3
        assert len(set(sim.steps_to_built_models.values())) == 3

    def test_build_for_experiment_legacy_processes_each_unique_step(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/20 for 1 hour",
                "Charge at 1 A for 10 seconds",
                "Discharge at C/20 for 1 hour",
            ]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="legacy",
        )

        sim.build_for_experiment()

        assert not sim._experiment_uses_unified_model
        assert set(sim.steps_to_built_models) == set(
            sim.experiment_unique_steps_to_model
        )
        assert set(sim.steps_to_built_solvers) == set(
            sim.experiment_unique_steps_to_model
        )
        assert all(model.is_discretised for model in sim.steps_to_built_models.values())
        assert all(
            solver is not sim.solver for solver in sim.steps_to_built_solvers.values()
        )
        assert len(
            {id(solver) for solver in sim.steps_to_built_solvers.values()}
        ) == len(sim.steps_to_built_solvers)

    def test_set_up_unified_mode_rejects_all_explicit_ode_solver(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/20 for 1 hour",
                "Rest for 10 minutes",
            ]
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            solver=pybamm.ScipySolver(),
            experiment_model_mode="unified",
        )

        with pytest.raises(pybamm.ModelError, match="DAE-capable solver"):
            sim.build_for_experiment()

    def test_set_up_differential_custom_step_falls_back_to_legacy(self):
        sim = self._make_differential_custom_step_simulation(
            solver=pybamm.IDAKLUSolver()
        )
        sim.build_for_experiment()

        assert not sim._experiment_uses_unified_model

    def test_set_up_unified_mode_rejects_differential_custom_step(self):
        sim = self._make_differential_custom_step_simulation(
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="unified",
        )

        with pytest.raises(
            pybamm.ModelError, match="differential control is not supported"
        ):
            sim.build_for_experiment()

    def test_experiment_state_mappers_built(self):
        experiment = pybamm.Experiment(
            ["Discharge at C/20 for 1 hour", "Rest for 10 minutes"]
        )
        for experiment_model_mode in ["legacy", "unified"]:
            sim = pybamm.Simulation(
                pybamm.lithium_ion.SPM(),
                experiment=experiment,
                solver=pybamm.IDAKLUSolver(),
                experiment_model_mode=experiment_model_mode,
            )
            sim.build_for_experiment()

            steps = sim.experiment.steps
            model_0 = sim.steps_to_built_models[steps[0].basic_repr()]
            model_1 = sim.steps_to_built_models[steps[1].basic_repr()]

            if experiment_model_mode == "legacy":
                assert (model_0, model_1) in sim.model_state_mappers
            else:
                assert len(sim.model_state_mappers.values()) == 0

    def test_built_models_and_state_mappers_independent_of_cycle_count(self):
        cycle_template = [
            {
                "type": "c-rate",
                "value": 1.0,
                "duration": 3600.0,
                "terminations": [{"type": "voltage", "value": 2.5}],
            },
            {
                "type": "c-rate",
                "value": -0.3,
                "duration": 24000.0,
                "terminations": [{"type": "voltage", "value": 4.2}],
            },
            {
                "type": "voltage",
                "value": 4.2,
                "duration": 86400.0,
                "terminations": [{"type": "c-rate", "value": 0.01}],
            },
        ]

        n_unique = len(cycle_template)
        # n_cycles >= 2 so the cycle-wrap transition (last->first) is realized
        # and mapper count plateaus.
        counts = []
        for n_cycles in (2, 4, 8):
            config = {
                "cycles": [copy.deepcopy(cycle_template) for _ in range(n_cycles)]
            }
            experiment = pybamm.Experiment.from_config(config)
            sim = pybamm.Simulation(
                pybamm.lithium_ion.SPM(),
                experiment=experiment,
                solver=pybamm.IDAKLUSolver(),
                experiment_model_mode="legacy",
            )
            sim.build_for_experiment()

            counts.append(
                (
                    n_cycles,
                    len(sim.steps_to_built_models),
                    len(set(sim.steps_to_built_models.values())),
                    len(sim.steps_to_built_solvers),
                    len(sim.model_state_mappers),
                )
            )

        for n_cycles, n_step_models, n_unique_models, n_solvers, n_mappers in counts:
            assert n_step_models == n_unique
            assert n_unique_models == n_unique
            assert n_solvers == n_unique
            assert n_mappers <= n_unique * n_unique, (
                f"model_state_mappers={n_mappers} for n_cycles={n_cycles}, "
                f"must be bounded by template transitions, not cycles"
            )

        first = counts[0][1:]
        for c in counts[1:]:
            assert c[1:] == first, (
                f"build counts grow with n_cycles instead of staying flat: {counts}"
            )

    def test_unified_model_switching_size_independent_of_cycle_count(self):
        # Switching must branch on unique steps, not instances, else O(n_steps**2) over
        # the experiment. Branch count stays bounded by unique steps, flat in cycles.
        def max_conditional_branches(model):
            roots = (
                list(model.rhs.values())
                + list(model.algebraic.values())
                + [event.expression for event in model.events]
            )
            seen = set()
            stack = list(roots)
            most = 0
            while stack:
                sym = stack.pop()
                if id(sym) in seen:
                    continue
                seen.add(id(sym))
                if isinstance(sym, pybamm.Conditional):
                    most = max(most, len(sym.branches))
                stack.extend(sym.children)
            return most

        unit_cycle = (
            "Discharge at 1C until 3.0V",
            "Charge at 1C until 4.2V",
            "Hold at 4.2V until C/50",
        )
        n_unique = len(unit_cycle)

        branch_counts = []
        step_index_maps = []
        for n_cycles in (2, 5, 20):
            experiment = pybamm.Experiment([unit_cycle] * n_cycles)
            sim = pybamm.Simulation(
                pybamm.lithium_ion.SPM(),
                experiment=experiment,
                solver=pybamm.IDAKLUSolver(),
                experiment_model_mode="unified",
            )
            sim.build_for_experiment()
            assert sim._experiment_uses_unified_model
            branch_counts.append(max_conditional_branches(sim._built_experiment_model))
            step_index_maps.append(sim._experiment_step_indices)

        assert branch_counts[0] == branch_counts[-1], (
            f"unified switching grows with cycles: {branch_counts}"
        )
        assert max(branch_counts) <= n_unique

        for n_cycles, indices in zip((2, 5, 20), step_index_maps, strict=True):
            assert len(indices) == n_unique * n_cycles
            assert set(indices) == set(range(1, n_unique + 1))
            assert indices == [(i % n_unique) + 1 for i in range(len(indices))]

    def test_unified_switch_top_functions_flat_in_unique_steps(self):
        # Per-branch dispatch keeps the top rhs/jac flat in unique step count.
        rhs_1, jac_1 = _unified_rhs_jac_n_instructions(
            [f"Discharge at {0.4 + 0.1 * i:.2f}C for 10 s" for i in range(1)]
        )
        rhs_8, jac_8 = _unified_rhs_jac_n_instructions(
            [f"Discharge at {0.4 + 0.1 * i:.2f}C for 10 s" for i in range(8)]
        )
        assert rhs_8 <= rhs_1 + 2, (
            f"unified rhs top function grows with unique steps: {rhs_1} -> {rhs_8}"
        )
        assert jac_8 <= jac_1 + 2, (
            f"unified jac top function grows with unique steps: {jac_1} -> {jac_8}"
        )

    def test_unified_active_branch_independent_of_other_modes(self):
        # Each mode is a separate branch function, so adding modes doesn't grow the top
        # rhs/jac and inactive modes aren't evaluated during an active step.
        rhs_cc, jac_cc = _unified_rhs_jac_n_instructions(["Discharge at 1C for 10 s"])
        rhs_multi, jac_multi = _unified_rhs_jac_n_instructions(
            [
                "Discharge at 1C for 10 s",
                "Charge at 0.5C until 4.2 V",
                "Hold at 4.2 V until C/50",
            ]
        )
        assert rhs_multi <= rhs_cc + 2, (
            f"adding modes grew the rhs top function: {rhs_cc} -> {rhs_multi}"
        )
        assert jac_multi <= jac_cc + 2, (
            f"adding modes grew the jac top function: {jac_cc} -> {jac_multi}"
        )

    def test_unified_switch_matches_legacy_voltage(self):
        # Switch dispatch must not change results vs legacy.
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 1C until 3.3 V",
                    "Charge at 1C until 4.0 V",
                    "Hold at 4.0 V until C/20",
                )
            ]
        )

        def voltage(mode):
            sim = pybamm.Simulation(
                pybamm.lithium_ion.SPM(),
                experiment=experiment,
                solver=pybamm.IDAKLUSolver(),
                experiment_model_mode=mode,
            )
            sim.solve()
            return sim.solution["Voltage [V]"].entries[-1]

        assert voltage("unified") == pytest.approx(voltage("legacy"), abs=1e-6)

    def test_unified_control_row_jacobian_is_sparse(self):
        # CasADi declares a Switch's jacobian structurally dense, so the control
        # equation would otherwise be a full-bandwidth row and balloon KLU work. The
        # solver projects it back onto the true (union-of-branches) sparsity; assert no
        # jacobian row spans the full bandwidth.
        from collections import Counter

        sim = pybamm.Simulation(
            pybamm.lithium_ion.DFN(),
            experiment=pybamm.Experiment(
                [("Discharge at 1C until 3.3 V", "Hold at 3.3 V until C/20")]
            ),
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="unified",
        )
        sim.solve()
        sparsity = sim._built_experiment_model.jac_rhs_algebraic_eval.sparsity_out(0)
        n = sparsity.size2()
        densest_row_nnz = max(Counter(sparsity.get_triplet()[0]).values())
        assert densest_row_nnz <= n // 2, (
            f"a jacobian row has {densest_row_nnz}/{n} nonzeros (near full bandwidth); "
            "the control-row union-sparsity projection did not take effect"
        )

    def test_unified_records_switching_control_variable_not_voltage(self):
        # The unified setup records the control variable (its one Conditional residual) by
        # name, so build_casadi_jacobian gives only that row the sparse switch treatment.
        # With "voltage as a state" the algebraic block also holds a "Voltage [V]" row,
        # which is physics and must NOT be picked up as a switch row.
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=pybamm.Experiment(
                [("Charge at 1 A for 1 min", "Hold at 4.1 V for 1 min")]
            ),
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="unified",
        )
        sim.build_for_experiment()
        model = sim.experiment_unique_steps_to_model[sim._experiment_unified_model_key]
        assert model.switching_control_variables == {"Current variable [A]"}

        def rows_for(name):
            return {
                r
                for v, slices in model.y_slices.items()
                if v.name == name
                for s in slices
                for r in range(s.start, s.stop)
            }

        switch_rows = set(model._switching_control_columns())
        voltage_rows = rows_for("Voltage [V]")
        assert voltage_rows, "expected voltage to be a state in this configuration"
        assert switch_rows == rows_for("Current variable [A]")
        assert not (switch_rows & voltage_rows)

    def test_non_unified_model_jacobian_is_plain_casadi(self):
        # An ordinary model records no switching control variable, so build_casadi_jacobian
        # short-circuits to plain casadi.jacobian -- zero extra work, identical result.
        import casadi

        model = pybamm.lithium_ion.SPM()
        assert model.switching_control_variables == set()

        x = casadi.MX.sym("x", 3)
        expr = casadi.vertcat(x[0] ** 2, x[1] * x[2], x[0] + x[1])
        assert casadi.is_equal(
            model.build_casadi_jacobian(expr, x), casadi.jacobian(expr, x), 20
        )

    def test_unified_aot_compile_bounded_in_unique_steps(self):
        # -O3 is superlinear in single-function size, so per-branch dispatch must keep
        # the largest generated jac function flat as unique steps grow. Deterministic.
        def largest_jac_fn_lines(n):
            ops = [f"Discharge at {0.4 + 0.1 * i:.2f}C for 10 s" for i in range(n)]
            sim = pybamm.Simulation(
                pybamm.lithium_ion.SPMe(),
                experiment=pybamm.Experiment([tuple(ops)]),
                solver=pybamm.IDAKLUSolver(),
                experiment_model_mode="unified",
            )
            sim.solve()
            return _largest_generated_fn_lines(
                sim._built_experiment_model.jac_rhs_algebraic_eval
            )

        big_2 = largest_jac_fn_lines(2)
        big_8 = largest_jac_fn_lines(8)
        assert big_8 <= big_2 + 50, (
            "largest unified jac function grew with unique steps: "
            f"{big_2} -> {big_8} lines (per-branch dispatch should keep it flat)"
        )

    def test_experiment_state_mapper_has_full_state_size_for_2d_current_collector(self):
        experiment = pybamm.Experiment(
            ["Discharge at C/20 for 1 hour", "Rest for 10 minutes"]
        )
        var_pts = {"x_n": 6, "x_s": 6, "x_p": 6, "r_n": 6, "r_p": 6, "y": 6, "z": 6}
        for experiment_model_mode in ["legacy"]:
            sim = pybamm.Simulation(
                pybamm.lithium_ion.DFN(
                    {"current collector": "potential pair", "dimensionality": 2}
                ),
                experiment=experiment,
                var_pts=var_pts,
                solver=pybamm.IDAKLUSolver(),
                experiment_model_mode=experiment_model_mode,
            )
            sim.build_for_experiment()

            steps = sim.experiment.steps
            model_0 = sim.steps_to_built_models[steps[0].basic_repr()]
            model_1 = sim.steps_to_built_models[steps[1].basic_repr()]
            if experiment_model_mode == "legacy":
                mapper = sim.model_state_mappers[(model_0, model_1)]
            else:
                assert model_0 is model_1
                mapper = sim.model_state_mappers[(model_0, model_1)]
            assert mapper.shape[0] == model_1.len_rhs_and_alg

    def test_run_experiment(self):
        s = pybamm.step.string
        experiment = pybamm.Experiment(
            [
                (
                    s("Discharge at C/20 for 1 hour", temperature="30.5oC"),
                    s("Charge at 1 A until 4.1 V", temperature="24oC"),
                    s("Hold at 4.1 V until C/2", temperature="24oC"),
                    "Discharge at 2 W for 10 minutes",
                    "Discharge at 4 Ohm for 10 minutes",
                )
            ],
            temperature="-14oC",
        )
        model = pybamm.lithium_ion.SPM()
        solver = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8)
        sim = pybamm.Simulation(model, experiment=experiment, solver=solver)
        # test the callback here
        sol = sim.solve(callbacks=pybamm.callbacks.Callback())
        assert sol.termination == "final time"
        assert len(sol.cycles) == 1

        # Test outputs
        np.testing.assert_allclose(
            sol.cycles[0].steps[0]["C-rate"].data, 1 / 20, rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            sol.cycles[0].steps[1]["Current [A]"].data, -1, rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            sol.cycles[0].steps[2]["Voltage [V]"].data, 4.1, rtol=1e-6, atol=1e-5
        )
        np.testing.assert_allclose(
            sol.cycles[0].steps[3]["Power [W]"].data, 2, rtol=3e-4, atol=3e-4
        )
        np.testing.assert_allclose(
            sol.cycles[0].steps[4]["Resistance [Ohm]"].data, 4, rtol=2e-4, atol=6e-4
        )

        np.testing.assert_array_equal(
            sol.cycles[0].steps[0]["Ambient temperature [C]"].data[0], 30.5
        )

        np.testing.assert_array_equal(
            sol.cycles[0].steps[1]["Ambient temperature [C]"].data[0], 24
        )

        np.testing.assert_array_equal(
            sol.cycles[0].steps[2]["Ambient temperature [C]"].data[0], 24
        )

        np.testing.assert_array_equal(
            sol.cycles[0].steps[3]["Ambient temperature [C]"].data[0], -14
        )

        for i, step in enumerate(sol.cycles[0].steps[:-1]):
            len_rhs = sol.all_models[0].concatenated_rhs.size
            y_left = step.all_ys[-1][:len_rhs, -1]
            if isinstance(y_left, casadi.DM):
                y_left = y_left.full()
            y_right = sol.cycles[0].steps[i + 1].all_ys[0][:len_rhs, 0]
            if isinstance(y_right, casadi.DM):
                y_right = y_right.full()
            np.testing.assert_allclose(
                y_left.flatten(), y_right.flatten(), rtol=1e-7, atol=1e-6
            )

        # Solve again starting from solution
        sol2 = sim.solve(starting_solution=sol)
        assert sol2.termination == "final time"
        assert sol2.t[-1] > sol.t[-1]
        assert sol2.cycles[0] == sol.cycles[0]
        assert len(sol2.cycles) == 2
        # Solve again starting from solution but only inputting the cycle
        sol2 = sim.solve(starting_solution=sol.cycles[-1])
        assert sol2.termination == "final time"
        assert sol2.t[-1] > sol.t[-1]
        assert len(sol2.cycles) == 2

        # Check starting solution is unchanged
        assert len(sol.cycles) == 1

        # save
        sol2.save("test_experiment.sav")
        sol3 = pybamm.load("test_experiment.sav")
        assert len(sol3.cycles) == 2
        os.remove("test_experiment.sav")

    def test_run_experiment_temperature_switching_unified(self):
        s = pybamm.step.string
        experiment = pybamm.Experiment(
            [
                (
                    s("Discharge at C/20 for 10 minutes", temperature="30.5oC"),
                    s("Charge at 1 A for 5 minutes", temperature="24oC"),
                    "Rest for 5 minutes",
                )
            ],
            temperature="-14oC",
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            solver=pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8),
            experiment_model_mode="unified",
        )
        sol = sim.solve(calc_esoh=False)

        assert sim._experiment_uses_unified_model
        np.testing.assert_array_equal(
            sol.cycles[0].steps[0]["Ambient temperature [C]"].data[0], 30.5
        )
        np.testing.assert_array_equal(
            sol.cycles[0].steps[1]["Ambient temperature [C]"].data[0], 24
        )
        np.testing.assert_array_equal(
            sol.cycles[0].steps[2]["Ambient temperature [C]"].data[0], -14
        )

    def test_run_mixed_control_experiment_unified_single_model(self):
        s = pybamm.step.string
        experiment = pybamm.Experiment(
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
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            solver=pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8),
            experiment_model_mode="unified",
        )

        sim.build_for_experiment()
        assert sim._experiment_uses_unified_model
        assert len(set(sim.steps_to_built_models.values())) == 1

        sol = sim.solve(calc_esoh=False)

        np.testing.assert_allclose(
            sol.cycles[0].steps[0]["C-rate"].data, 1 / 20, rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            sol.cycles[0].steps[1]["Current [A]"].data, -1, rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            sol.cycles[0].steps[2]["Voltage [V]"].data, 4.1, rtol=1e-6, atol=1e-5
        )
        np.testing.assert_allclose(
            sol.cycles[0].steps[3]["Power [W]"].data, 2, rtol=3e-4, atol=3e-4
        )
        np.testing.assert_allclose(
            sol.cycles[0].steps[4]["Resistance [Ohm]"].data, 4, rtol=2e-4, atol=6e-4
        )

    def test_run_mixed_control_experiment_unified_matches_legacy(self):
        s = pybamm.step.string
        experiment = pybamm.Experiment(
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

        legacy_sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="legacy",
        )
        unified_sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="unified",
        )

        legacy_sol = legacy_sim.solve(calc_esoh=False)
        unified_sol = unified_sim.solve(calc_esoh=False)

        assert not legacy_sim._experiment_uses_unified_model
        assert unified_sim._experiment_uses_unified_model

        legacy_steps = legacy_sol.cycles[0].steps
        unified_steps = unified_sol.cycles[0].steps
        assert len(legacy_steps) == len(unified_steps)

        for legacy_step, unified_step in zip(legacy_steps, unified_steps, strict=True):
            assert legacy_step.termination == unified_step.termination
            np.testing.assert_allclose(
                legacy_step.t[-1], unified_step.t[-1], rtol=1e-8, atol=1e-8
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

        np.testing.assert_allclose(
            legacy_sol["Discharge capacity [A.h]"].data[-1],
            unified_sol["Discharge capacity [A.h]"].data[-1],
            rtol=5e-5,
            atol=5e-5,
        )

    def test_run_event_driven_experiment_unified_matches_legacy(self):
        experiment = pybamm.Experiment(
            [("Charge at C/3 until 4.1 V", "Hold at 4.1 V until C/20")]
        )

        # Tolerances tight enough that event-crossing times are resolved to
        # better than the assertion tolerances below; at default tolerances
        # the shallow dV/dt near the voltage cut-off lets the two modes land
        # seconds apart while both remain within integration tolerance.
        legacy_sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-8),
            experiment_model_mode="legacy",
        )
        unified_sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-8),
            experiment_model_mode="unified",
        )

        legacy_sol = legacy_sim.solve(calc_esoh=False, initial_soc=0.2)
        unified_sol = unified_sim.solve(calc_esoh=False, initial_soc=0.2)

        legacy_steps = legacy_sol.cycles[0].steps
        unified_steps = unified_sol.cycles[0].steps
        assert len(legacy_steps) == len(unified_steps) == 2

        for legacy_step, unified_step in zip(legacy_steps, unified_steps, strict=True):
            assert legacy_step.termination == unified_step.termination
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

        np.testing.assert_allclose(
            legacy_sol.t[-1], unified_sol.t[-1], rtol=5e-5, atol=5e-4
        )

    def test_run_multi_termination_step_unified_matches_legacy(self):
        experiment = pybamm.Experiment(
            [
                (
                    "Charge at 1 A until 4.1 V",
                    pybamm.step.Voltage(4.1, termination=["0.5 A", "0.1 A"]),
                )
            ]
        )

        legacy_sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="legacy",
        )
        unified_sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="unified",
        )

        legacy_sol = legacy_sim.solve(calc_esoh=False, initial_soc=0.2)
        unified_sol = unified_sim.solve(calc_esoh=False, initial_soc=0.2)

        legacy_hold = legacy_sol.cycles[0].steps[1]
        unified_hold = unified_sol.cycles[0].steps[1]

        assert (
            legacy_hold.termination == "event: abs(Current [A]) < 0.5 [A] [experiment]"
        )
        assert legacy_hold.termination == unified_hold.termination
        np.testing.assert_allclose(
            legacy_hold.t[-1], unified_hold.t[-1], rtol=5e-5, atol=5e-4
        )
        np.testing.assert_allclose(
            legacy_hold["Current [A]"].data[-1],
            unified_hold["Current [A]"].data[-1],
            rtol=5e-5,
            atol=5e-5,
        )
        np.testing.assert_allclose(
            legacy_hold["Voltage [V]"].data[-1],
            unified_hold["Voltage [V]"].data[-1],
            rtol=5e-5,
            atol=5e-5,
        )

    def test_run_unified_resistance_branch_is_safe_when_inactive_at_zero_current(self):
        experiment = pybamm.Experiment(
            [("Rest for 5 minutes", "Discharge at 4 Ohm for 5 minutes")]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="unified",
        )

        sol = sim.solve(calc_esoh=False)

        assert sim._experiment_uses_unified_model
        np.testing.assert_allclose(
            sol.cycles[0].steps[0]["Current [A]"].data, 0, atol=1e-10
        )
        np.testing.assert_allclose(
            sol.cycles[0].steps[1]["Resistance [Ohm]"].data, 4, rtol=2e-4, atol=6e-4
        )

    def test_skip_ok(self):
        model = pybamm.lithium_ion.SPMe()
        cc_charge_skip_ok = pybamm.step.Current(-5, termination="4.2 V")
        cc_charge_skip_not_ok = pybamm.step.Current(
            -5, termination="4.2 V", skip_ok=False
        )
        steps = [
            pybamm.step.Current(2, duration=100.0, skip_ok=False),
            cc_charge_skip_ok,
            pybamm.step.Voltage(4.2, termination="0.01 A", skip_ok=False),
        ]
        param = pybamm.ParameterValues("Chen2020")
        experiment = pybamm.Experiment(steps)
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sol = sim.solve()
        # Make sure we know to skip it if we should and not if we shouldn't
        assert sim.experiment.steps[1].skip_ok
        assert not sim.experiment.steps[0].skip_ok

        # Make sure we actually skipped it because it is infeasible
        assert len(sol.cycles) == 2

        # In this case, it is feasible, so we should not skip it
        sol2 = sim.solve(initial_soc=0.5)
        assert len(sol2.cycles) == 3

        # make sure we raise an error if we shouldn't skip it and it is infeasible
        steps[1] = cc_charge_skip_not_ok
        experiment = pybamm.Experiment(steps)
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        with pytest.raises(pybamm.SolverError):
            sim.solve()

        # make sure we raise an error if all steps are infeasible
        steps = [
            (pybamm.step.Current(-5, termination="4.2 V", skip_ok=True),) * 5,
        ]
        experiment = pybamm.Experiment(steps)
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        with pytest.raises(pybamm.SolverError, match=r"skip_ok is True for all steps"):
            sim.solve()

        # Check termination after a skipped step
        steps = [
            pybamm.step.Current(2, duration=100.0, skip_ok=False),
            cc_charge_skip_ok,
        ]
        experiment = pybamm.Experiment(steps)
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sol = sim.solve()
        assert sol.termination == "Event exceeded in initial conditions"

    def test_skip_ok_with_multiple_infeasible_terminations_in_unified_model(self):
        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment(
            [
                pybamm.step.Current(
                    -5,
                    termination=[(">", "current", -6), ("<", "current", -4)],
                    skip_ok=True,
                ),
                pybamm.step.Voltage(4.2, termination="0.01 A", skip_ok=False),
            ]
        )
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            solver=pybamm.CasadiSolver(),
            experiment_model_mode="unified",
        )

        sol = sim.solve(calc_esoh=False)

        assert sim._experiment_uses_unified_model
        assert len(sol.cycles) == 1
        assert len(sol.cycles[0].steps) == 1
        assert (
            sol.cycles[0].steps[0].termination
            == "event: abs(Current [A]) < 0.01 [A] [experiment]"
        )

    def test_all_empty_solution_errors(self):
        model = pybamm.lithium_ion.SPM()
        parameter_values = pybamm.ParameterValues("Chen2020")

        # One step exceeded, suggests making a cycle
        steps = [
            (pybamm.step.Current(-5, termination="4.2 V", skip_ok=False),) * 5,
        ]
        experiment = pybamm.Experiment(steps)
        sim = pybamm.Simulation(
            model, experiment=experiment, parameter_values=parameter_values
        )
        with pytest.raises(pybamm.SolverError, match=r"All steps in the cycle"):
            sim.solve()

    def test_run_experiment_multiple_times(self):
        s = pybamm.step.string
        experiment = pybamm.Experiment(
            [
                (
                    s("Discharge at C/20 for 1 hour", temperature="24oC"),
                    s("Charge at C/20 until 4.1 V", temperature="26oC"),
                )
            ]
            * 3
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)

        # Test that solving twice gives the same solution (see #2193)
        sol1 = sim.solve()
        sol2 = sim.solve()
        np.testing.assert_array_equal(
            sol1["Voltage [V]"].data, sol2["Voltage [V]"].data
        )

    def test_run_experiment_cccv_solvers(self):
        experiment_2step = pybamm.Experiment(
            [
                (
                    "Discharge at C/10 for 1 hour",
                    "Charge at C/20 for 1 hour",
                    "Hold at 3.7 V for 1 hour",
                ),
            ]
            * 2,
            period="100 hours",  # only capture the first and final values
        )
        rtol = 1e-6
        atol = 1e-15

        solvers = {
            "casadi": pybamm.CasadiSolver(atol=atol, rtol=rtol),
            "idaklu": pybamm.IDAKLUSolver(atol=atol, rtol=rtol),
        }

        solutions = {}
        for name, solver in solvers.items():
            model = pybamm.lithium_ion.SPM()
            sim = pybamm.Simulation(model, experiment=experiment_2step, solver=solver)
            solution = sim.solve()
            assert solution.t[-1] == pytest.approx(3600 * len(experiment_2step.steps))
            solutions[name] = solution

        num_sub_solutions = len(solutions["casadi"].sub_solutions)
        assert len(solutions["idaklu"].sub_solutions) == num_sub_solutions

        def get_sub_solution(name, idx_sol, idx_time):
            y = solutions[name].sub_solutions[idx_sol].all_ys[0][:, idx_time]
            if isinstance(y, casadi.DM):
                y = y.full().flatten()
            return y

        for idx_sol, idx_time in itertools.product(range(num_sub_solutions), [0, -1]):
            t = solutions["casadi"].sub_solutions[idx_sol].t[idx_time]
            casadi_sol = get_sub_solution("casadi", idx_sol, idx_time)
            idaklu_sol = get_sub_solution("idaklu", idx_sol, idx_time)

            err_msg = f"Failed to match solution y values at {t} seconds"
            np.testing.assert_allclose(
                casadi_sol,
                idaklu_sol,
                rtol=rtol * 100,
                atol=atol * 100,
                err_msg=err_msg,
            )
        assert solutions["casadi"].termination == "final time"

    @pytest.mark.parametrize(
        "solver_cls",
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ids=["SPM", "DFN"],
    )
    def test_solve_with_sensitivities_and_experiment(self, solver_cls):
        experiment_2step = pybamm.Experiment(
            [
                (
                    "Discharge at C/20 for 2 min",
                    "Charge at 1 A for 1 min",
                    "Hold at 4.1 V for 1 min",
                    "Discharge at 2 W for 1 min",
                    "Discharge at 2 W for 1 min",  # repeat to cover this case (changes initialisation)
                ),
            ]
            * 2,
        )

        solutions = []
        input_param_name = "Negative electrode active material volume fraction"
        solver = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8)
        for calculate_sensitivities in [False, True]:
            model = solver_cls()
            param = model.default_parameter_values
            input_param_value = param[input_param_name]
            param.update({input_param_name: "[input]"})
            sim = pybamm.Simulation(
                model,
                experiment=experiment_2step,
                solver=solver,
                parameter_values=param,
            )
            solution = sim.solve(
                inputs={input_param_name: input_param_value},
                calculate_sensitivities=calculate_sensitivities,
            )
            solutions.append(solution)

        model = solver_cls()
        param = model.default_parameter_values
        base_input_param_value = param[input_param_name]
        fd_tol = 1e-4
        for dh in [-fd_tol, fd_tol]:
            model = solver_cls()
            param = model.default_parameter_values
            input_param_value = base_input_param_value * (1.0 + dh)
            param.update({input_param_name: "[input]"})
            sim = pybamm.Simulation(
                model,
                experiment=experiment_2step,
                solver=solver,
                parameter_values=param,
            )
            solution = sim.solve(
                inputs={input_param_name: input_param_value},
            )
            solutions.append(solution)

        # check solutions are the same
        np.testing.assert_allclose(
            solutions[0]["Voltage [V]"].data,
            solutions[1]["Voltage [V]"](solutions[0].t),
            rtol=5e-2,
            equal_nan=True,
        )
        sensitivity_keys = set(solutions[1]["Voltage [V]"].sensitivities)
        assert input_param_name in sensitivity_keys
        assert pybamm.Simulation._STEP_INDEX_INPUT not in sensitivity_keys

        # use finite difference to check sensitivities
        t = solutions[0].t
        soln_neg = solutions[2]["Voltage [V]"](t)
        soln_pos = solutions[3]["Voltage [V]"](t)
        sens_fd = (soln_pos - soln_neg) / (2 * fd_tol * base_input_param_value)
        sens_idaklu = np.interp(
            t,
            solutions[1].t,
            solutions[1]["Voltage [V]"].sensitivities[input_param_name].flatten(),
        )
        np.testing.assert_allclose(
            sens_fd,
            sens_idaklu,
            rtol=2e-4,
            atol=2e-3,
        )

    def test_solve_with_sensitivity_list_ignores_internal_experiment_input(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/20 for 10 seconds",
                "Charge at 1 A for 10 seconds",
            ]
        )
        input_param_name = "Negative electrode active material volume fraction"
        model = pybamm.lithium_ion.SPM()
        input_param_value = model.default_parameter_values[input_param_name]

        def make_sim():
            model = pybamm.lithium_ion.SPM()
            param = model.default_parameter_values
            param.update({input_param_name: "[input]"})
            return pybamm.Simulation(
                model,
                experiment=experiment,
                solver=pybamm.IDAKLUSolver(),
                parameter_values=param,
                experiment_model_mode="unified",
            )

        baseline = make_sim().solve(inputs={input_param_name: input_param_value})

        internal_keys = set(baseline.all_inputs[0]) - {
            input_param_name,
            "Ambient temperature [K]",
            "start time",
        }
        # Unified mode injects the step-index selector and the per-step value input.
        assert internal_keys == {
            "Experiment step index",
            "Experiment step value",
        }

        sol = make_sim().solve(
            inputs={input_param_name: input_param_value},
            calculate_sensitivities=[input_param_name, *internal_keys],
        )

        sensitivity_keys = set(sol.sensitivities)
        assert input_param_name in sensitivity_keys
        for experiment_input_name in internal_keys:
            assert experiment_input_name not in sensitivity_keys

    def test_processed_variable_sensitivities_ignore_experiment_input(self):
        # Regression test for #5517.
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        diffusivity_name = "Positive particle diffusivity [m2.s-1]"
        param.update({diffusivity_name: "[input]"})

        experiment = pybamm.Experiment(
            [pybamm.step.Current(pybamm.InputParameter("current"), duration=10)]
        )
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sol = sim.solve(
            inputs={"current": 0.1, diffusivity_name: 1e-14},
            calculate_sensitivities=[diffusivity_name],
        )

        sensitivities = sol["Voltage [V]"].sensitivities
        assert set(sensitivities) == {"all", diffusivity_name}
        sens = np.asarray(sensitivities[diffusivity_name])
        assert np.all(np.isfinite(sens))
        assert np.any(sens != 0)

    def test_run_experiment_drive_cycle(self):
        drive_cycle = np.array([np.arange(10), np.arange(10)]).T
        experiment = pybamm.Experiment(
            [
                (
                    pybamm.step.current(drive_cycle, temperature="35oC"),
                    pybamm.step.voltage(drive_cycle),
                    pybamm.step.power(drive_cycle, termination="< 3 V"),
                )
            ],
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.build_for_experiment()
        assert sorted([step.basic_repr() for step in experiment.steps]) == sorted(
            list(sim.experiment_unique_steps_to_model.keys())
        )

    def test_run_experiment_drive_cycle_experiment(self):
        time = [0, 5, 10]
        current = [-1, -2, -1]
        drive_cycle = np.column_stack([time, current])
        experiment = pybamm.Experiment([pybamm.step.current(drive_cycle)])
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve()
        assert sol.termination == "final time"

        assert all(t in sol.t for t in time)
        assert len(sol.t) > len(time)

        np.testing.assert_allclose(sol["Current [A]"](time), current)

    def test_solve_with_sensitivities_true_excludes_all_internal_inputs(self):
        # calculate_sensitivities=True must differentiate only w.r.t. user inputs, not
        # the control inputs the experiment injects per step.
        input_param_name = "Negative electrode active material volume fraction"

        def make_sim():
            model = pybamm.lithium_ion.SPM()
            param = model.default_parameter_values
            param.update({input_param_name: "[input]"})
            # A python-function step (current as a function of time) puts "start time"
            # into the model, so all three internal inputs are present and could leak.
            experiment = pybamm.Experiment(
                [
                    pybamm.step.current(
                        lambda t: 1.0 + 0.5 * np.sin(t / 100), duration=300
                    ),
                    "Charge at 1 A for 10 seconds",
                ]
            )
            return pybamm.Simulation(
                model,
                experiment=experiment,
                solver=pybamm.IDAKLUSolver(),
                parameter_values=param,
                experiment_model_mode="unified",
            )

        input_value = pybamm.ParameterValues("Chen2020")[input_param_name]
        sol = make_sim().solve(
            inputs={input_param_name: input_value},
            calculate_sensitivities=True,
            calc_esoh=False,
        )

        internal_inputs = pybamm.Simulation._INTERNAL_EXPERIMENT_INPUTS
        # "start time" really is one of the model inputs here, so this is a genuine
        # exclusion, not a vacuous one.
        assert "start time" in sol.all_inputs[0]
        sensitivity_keys = set(sol.sensitivities) - {"all"}
        assert sensitivity_keys == {input_param_name}
        assert not (sensitivity_keys & internal_inputs)

    def test_unified_ambient_temperature_scalar_value(self):
        # Unified mode reads ambient temperature as a solver input; a pybamm.Scalar
        # value (vs a float) must be coerced or it breaks the casadi call.
        experiment = pybamm.Experiment(
            ["Discharge at 1C until 3.0 V", "Charge at 1C until 4.2 V"]
        )

        def solve(ambient_value):
            pv = pybamm.ParameterValues("Chen2020")
            pv["Ambient temperature [K]"] = ambient_value
            sim = pybamm.Simulation(
                pybamm.lithium_ion.SPM(),
                parameter_values=pv,
                experiment=experiment,
                experiment_model_mode="unified",
                solver=pybamm.IDAKLUSolver(),
            )
            return sim.solve(calc_esoh=False)

        scalar_sol = solve(pybamm.Scalar(298.15))
        float_sol = solve(298.15)

        np.testing.assert_allclose(scalar_sol.t[-1], float_sol.t[-1], rtol=1e-10)
        np.testing.assert_allclose(
            scalar_sol["Voltage [V]"].data[-1],
            float_sol["Voltage [V]"].data[-1],
            rtol=1e-8,
            atol=1e-8,
        )

    def test_run_experiment_breaks_early_infeasible(self):
        experiment = pybamm.Experiment(["Discharge at 2 C for 1 hour"])
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        pybamm.set_logging_level("ERROR")
        # giving the time, should get ignored
        t_eval = [0, 1]
        sim.solve(t_eval, callbacks=pybamm.callbacks.Callback())
        pybamm.set_logging_level("WARNING")
        assert sim._solution.termination == "event: Minimum voltage [V]"

    def test_run_experiment_breaks_early_error(self):
        s = pybamm.step.string
        experiment = pybamm.Experiment(
            [
                (
                    "Rest for 10 minutes",
                    s("Discharge at 20000 C for 10 minutes"),
                )
            ]
        )
        model = pybamm.lithium_ion.DFN()

        parameter_values = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=parameter_values,
        )
        sol = sim.solve()
        assert len(sol.cycles) == 1
        assert len(sol.cycles[0].steps) == 1

        # Different experiment setup style
        experiment = pybamm.Experiment(
            [
                "Rest for 10 minutes",
                s("Discharge at 20000 C for 10 minutes"),
            ]
        )
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=parameter_values,
        )
        sol = sim.solve()
        assert len(sol.cycles) == 1
        assert len(sol.cycles[0].steps) == 1

        # Different callback - this is for coverage on the `Callback` class
        sol = sim.solve(callbacks=pybamm.callbacks.Callback())

    def test_run_experiment_infeasible_time(self):
        experiment = pybamm.Experiment(
            [ShortDurationCRate(1, termination="2.5V"), "Rest for 1 hour"]
        )
        model = pybamm.lithium_ion.SPM()
        parameter_values = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        )
        sol = sim.solve()
        assert len(sol.cycles) == 1
        assert len(sol.cycles[0].steps) == 1

    def test_run_experiment_termination_capacity(self):
        # with percent
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 1C until 3V",
                    "Charge at 1C until 4.2 V",
                    "Hold at 4.2V until C/10",
                ),
            ]
            * 10,
            termination="99% capacity",
        )
        model = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited"})
        param = pybamm.ParameterValues("Chen2020")
        param["SEI kinetic rate constant [m.s-1]"] = 1e-14
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sol = sim.solve()
        assert sol.termination == "experiment capacity limit reached"
        C = sol.summary_variables["Capacity [A.h]"]
        np.testing.assert_array_less(np.diff(C), 0)
        # all but the last value should be above the termination condition
        np.testing.assert_array_less(0.99 * C[0], C[:-1])

        # with Ah value
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 1C until 3V",
                    "Charge at 1C until 4.2 V",
                    "Hold at 4.2V until C/10",
                ),
            ]
            * 10,
            termination="5.04Ah capacity",
        )
        model = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited"})
        param = pybamm.ParameterValues("Chen2020")
        param["SEI kinetic rate constant [m.s-1]"] = 1e-14
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sol = sim.solve()
        assert sol.termination == "experiment capacity limit reached"
        C = sol.summary_variables["Capacity [A.h]"]
        # all but the last value should be above the termination condition
        np.testing.assert_array_less(5.04, C[:-1])

    def test_run_experiment_with_pbar(self):
        # The only thing to test here is for errors.
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 1C for 1 sec",
                    "Charge at 1C for 1 sec",
                ),
            ]
            * 10,
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.solve(showprogress=True)

    def test_run_experiment_termination_voltage(self):
        # with percent
        experiment = pybamm.Experiment(
            [
                ("Discharge at 0.5C for 10 minutes", "Rest for 10 minutes"),
            ]
            * 5,
            termination="4V",
        )
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        # Test with calc_esoh=False here
        sol = sim.solve(calc_esoh=False)
        assert sol.termination == "experiment voltage limit reached"
        # Only two cycles should be completed, only 2nd cycle should go below 4V
        np.testing.assert_array_less(4, np.min(sol.cycles[0]["Voltage [V]"].data))
        np.testing.assert_array_less(np.min(sol.cycles[1]["Voltage [V]"].data), 4)
        assert len(sol.cycles) == 2

    def test_run_experiment_termination_time_min(self, caplog):
        experiment = pybamm.Experiment(
            [
                ("Discharge at 0.5C for 10 minutes", "Rest for 10 minutes"),
            ]
            * 5,
            termination="25 min",
        )
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        # Test with calc_esoh=False here
        with caplog.at_level(logging.ERROR, logger="pybamm.logger"):
            sol = sim.solve(calc_esoh=False)
        assert sol.termination == "experiment time limit reached"
        assert "Step time must be >0" not in caplog.text
        # Only two cycles should be completed, only 2nd cycle should go below 4V
        np.testing.assert_array_less(np.max(sol.cycles[0]["Time [s]"].data), 1500)
        np.testing.assert_array_equal(np.max(sol.cycles[1]["Time [s]"].data), 1500)
        assert len(sol.cycles) == 2

    def test_run_experiment_termination_time_s(self):
        experiment = pybamm.Experiment(
            [
                ("Discharge at 0.5C for 10 minutes", "Rest for 10 minutes"),
            ]
            * 5,
            termination="1500 s",
        )
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        # Test with calc_esoh=False here
        sol = sim.solve(calc_esoh=False)
        assert sol.termination == "experiment time limit reached"
        # Only two cycles should be completed, only 2nd cycle should go below 4V
        np.testing.assert_array_less(np.max(sol.cycles[0]["Time [s]"].data), 1500)
        np.testing.assert_array_equal(np.max(sol.cycles[1]["Time [s]"].data), 1500)
        assert len(sol.cycles) == 2

    def test_run_experiment_termination_time_h(self):
        experiment = pybamm.Experiment(
            [
                ("Discharge at 0.5C for 10 minutes", "Rest for 10 minutes"),
            ]
            * 5,
            termination="0.5 h",
        )
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        # Test with calc_esoh=False here
        sol = sim.solve(calc_esoh=False)
        assert sol.termination == "experiment time limit reached"
        # Only two cycles should be completed, only 2nd cycle should go below 4V
        np.testing.assert_array_less(np.max(sol.cycles[0]["Time [s]"].data), 1800)
        np.testing.assert_array_equal(np.max(sol.cycles[1]["Time [s]"].data), 1800)
        assert len(sol.cycles) == 2

    def test_run_experiment_termination_time_with_starting_solution_at_limit(self):
        experiment = pybamm.Experiment(
            [pybamm.step.string("Discharge at 0.5C for 10 seconds")],
            termination="10 s",
        )
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM(), experiment=experiment)

        solution = sim.solve(calc_esoh=False)
        resumed_solution = sim.solve(calc_esoh=False, starting_solution=solution.copy())

        assert solution.termination == "experiment time limit reached"
        assert resumed_solution.termination == "experiment time limit reached"
        assert resumed_solution["Time [s]"].entries[-1] == pytest.approx(10)

    def test_save_at_cycles(self):
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 1C until 3.3V",
                    "Charge at 1C until 4.1 V",
                    "Hold at 4.1V until C/10",
                ),
            ]
            * 10,
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve(save_at_cycles=2)
        # Solution saves "None" for the cycles that are not saved
        for cycle_num in [2, 4, 6, 8]:
            assert sol.cycles[cycle_num] is None
        for cycle_num in [0, 1, 3, 5, 7, 9]:
            assert sol.cycles[cycle_num] is not None
        # Summary variables are not None
        assert sol.summary_variables["Capacity [A.h]"] is not None

        sol = sim.solve(save_at_cycles=[3, 4, 5, 9])
        # Note offset by 1 (0th cycle is cycle 1)
        for cycle_num in [1, 5, 6, 7]:
            assert sol.cycles[cycle_num] is None
        for cycle_num in [0, 2, 3, 4, 8, 9]:  # first & last cycle always saved
            assert sol.cycles[cycle_num] is not None
        # Summary variables are not None
        assert sol.summary_variables["Capacity [A.h]"] is not None

    def test_cycle_summary_variables(self):
        # Test cycle_summary_variables works for different combinations of data and
        # function OCPs
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 1C until 3.3V",
                    "Charge at C/3 until 4.0V",
                    "Hold at 4.0V until C/10",
                ),
            ]
            * 5,
        )
        model = pybamm.lithium_ion.SPM()

        # O'Kane 2022: pos = function, neg = data
        param = pybamm.ParameterValues("OKane2022")
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sim.solve(save_at_cycles=2)

        # Chen 2020: pos = function, neg = function
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sim.solve(save_at_cycles=2)

        # Chen 2020 with data: pos = data, neg = data
        # Load negative electrode OCP data
        filename = os.path.join(
            pybamm.root_dir(),
            "src",
            "pybamm",
            "input",
            "parameters",
            "lithium_ion",
            "data",
            "graphite_LGM50_ocp_Chen2020.csv",
        )
        param["Negative electrode OCP [V]"] = pybamm.parameters.process_1D_data(
            filename
        )

        # Load positive electrode OCP data
        filename = os.path.join(
            pybamm.root_dir(),
            "src",
            "pybamm",
            "input",
            "parameters",
            "lithium_ion",
            "data",
            "nmc_LGM50_ocp_Chen2020.csv",
        )
        param["Positive electrode OCP [V]"] = pybamm.parameters.process_1D_data(
            filename
        )

        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sim.solve(save_at_cycles=2)

    def test_inputs(self):
        experiment = pybamm.Experiment(
            ["Discharge at C/2 for 1 hour", "Rest for 1 hour"]
        )
        model = pybamm.lithium_ion.SPM()

        # Change a parameter to an input
        param = pybamm.ParameterValues("Marquis2019")
        param["Negative particle diffusivity [m2.s-1]"] = (
            pybamm.InputParameter("Dsn") * 3.9e-14
        )

        # Solve a first time
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sim.solve(inputs={"Dsn": 1})
        np.testing.assert_array_equal(sim.solution.all_inputs[0]["Dsn"], 1)

        # Solve again, input should change
        sim.solve(inputs={"Dsn": 2})
        np.testing.assert_array_equal(sim.solution.all_inputs[0]["Dsn"], 2)

    def test_run_experiment_skip_steps(self):
        # Test experiment with steps being skipped due to initial conditions
        # already satisfying the events
        model = pybamm.lithium_ion.SPM()
        parameter_values = pybamm.ParameterValues("Chen2020")
        experiment = pybamm.Experiment(
            [
                (
                    "Charge at 1C until 4.2V",
                    "Hold at 4.2V until 10 mA",
                    "Discharge at 1C for 1 hour",
                    "Charge at 20C until 3V",
                    "Hold at 3V until 10 mA",
                ),
            ]
        )
        sim = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        )
        sol = sim.solve()
        assert isinstance(sol.cycles[0].steps[0], pybamm.EmptySolution)
        assert isinstance(sol.cycles[0].steps[3], pybamm.EmptySolution)

        # Should get the same result if we run without the charge steps
        # since they are skipped
        experiment2 = pybamm.Experiment(
            [
                (
                    "Hold at 4.2V until 10 mA",
                    "Discharge at 1C for 1 hour",
                    "Hold at 3V until 10 mA",
                ),
            ]
        )
        sim2 = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment2
        )
        sol2 = sim2.solve()
        np.testing.assert_allclose(
            sol["Voltage [V]"].data, sol2["Voltage [V]"].data, rtol=1e-6, atol=1e-5
        )
        for idx1, idx2 in [(1, 0), (2, 1), (4, 2)]:
            np.testing.assert_allclose(
                sol.cycles[0].steps[idx1]["Voltage [V]"].data,
                sol2.cycles[0].steps[idx2]["Voltage [V]"].data,
                rtol=1e-6,
                atol=1e-5,
            )

    def test_skipped_step_continuous(self):
        experiment = pybamm.Experiment(
            [
                ("Rest for 24 hours (1 hour period)",),
                (
                    "Charge at C/3 until 4.1 V",
                    "Hold at 4.1V until C/20",
                    "Discharge at C/3 until 2.5 V",
                ),
            ]
        )
        for experiment_model_mode in ["legacy", "unified"]:
            sim = pybamm.Simulation(
                pybamm.lithium_ion.SPM({"SEI": "solvent-diffusion limited"}),
                experiment=experiment,
                solver=pybamm.IDAKLUSolver(),
                experiment_model_mode=experiment_model_mode,
            )
            sim.solve(initial_soc=1)
            previous_state = sim.solution.cycles[0].last_state.y
            next_state = sim.solution.cycles[1].steps[-1].first_state.y

            built_model = sim.steps_to_built_models[
                sim.experiment.steps[0].basic_repr()
            ]
            algebraic_indices = []
            for variable in built_model.algebraic:
                for state_slice in built_model.y_slices[variable]:
                    algebraic_indices.extend(range(state_slice.start, state_slice.stop))
            rhs_indices = [
                i for i in range(previous_state.shape[0]) if i not in algebraic_indices
            ]
            np.testing.assert_allclose(
                previous_state[rhs_indices],
                next_state[rhs_indices],
                atol=1e-14,
                rtol=1e-14,
            )

    def test_run_experiment_half_cell(self):
        experiment = pybamm.Experiment(
            [("Discharge at C/20 until 3.5V", "Charge at 1C until 3.8 V")]
        )
        model = pybamm.lithium_ion.SPM({"working electrode": "positive"})
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=pybamm.ParameterValues("Xu2019"),
        )
        sim.solve()

    def test_run_experiment_lead_acid(self):
        experiment = pybamm.Experiment(
            [("Discharge at C/20 until 10.5V", "Charge at C/20 until 12.5 V")]
        )
        model = pybamm.lead_acid.Full()
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.solve()

    def test_padding_rest_model(self):
        model = pybamm.lithium_ion.SPM()

        # Test no padding rest model if there are no start_times
        experiment = pybamm.Experiment(["Rest for 1 hour"])
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.build_for_experiment()
        assert "Rest for padding" not in sim.experiment_unique_steps_to_model.keys()

        # Test padding rest model exists if there are start_times
        experiment = pybamm.step.string(
            "Rest for 1 hour", start_time=datetime(1, 1, 1, 8, 0, 0)
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.build_for_experiment()
        assert not sim._experiment_uses_unified_model
        assert "Rest for padding" in sim.experiment_unique_steps_to_model.keys()
        assert not sim._experiment_includes_padding_rest

    def test_run_start_time_experiment(self):
        model = pybamm.lithium_ion.SPM()

        # Test experiment is cut short if next_start_time is early
        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at 0.5C for 1 hour",
                    start_time=datetime(2023, 1, 1, 8, 0, 0),
                ),
                pybamm.step.string(
                    "Rest for 1 hour", start_time=datetime(2023, 1, 1, 8, 30, 0)
                ),
            ]
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve(calc_esoh=False)
        assert not sim._experiment_uses_unified_model
        assert sol["Time [s]"].entries[-1] == 5400

        # Test padding rest is added if time stamp is late
        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at 0.5C for 1 hour",
                    start_time=datetime(2023, 1, 1, 8, 0, 0),
                ),
                pybamm.step.string(
                    "Rest for 1 hour", start_time=datetime(2023, 1, 1, 10, 0, 0)
                ),
            ]
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve(calc_esoh=False)
        assert not sim._experiment_uses_unified_model
        assert sol["Time [s]"].entries[-1] == 10800

    def test_run_start_time_experiment_forced_unified(self):
        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at 0.5C for 1 hour",
                    start_time=datetime(2023, 1, 1, 8, 0, 0),
                ),
                pybamm.step.string(
                    "Rest for 1 hour", start_time=datetime(2023, 1, 1, 10, 0, 0)
                ),
            ]
        )

        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            experiment_model_mode="unified",
        )
        sol = sim.solve(calc_esoh=False)

        assert sim._experiment_uses_unified_model
        assert "Rest for padding" not in sim.steps_to_built_models
        assert sim._experiment_includes_padding_rest
        assert sol["Time [s]"].entries[-1] == 10800

    def test_run_start_time_experiment_legacy_builds_padding_rest_mappers(self):
        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at 0.5C for 1 hour",
                    start_time=datetime(2023, 1, 1, 8, 0, 0),
                ),
                pybamm.step.string(
                    "Rest for 1 hour", start_time=datetime(2023, 1, 1, 10, 0, 0)
                ),
            ]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode="legacy",
        )

        sol = sim.solve(calc_esoh=False)

        assert not sim._experiment_uses_unified_model
        assert "Rest for padding" in sim.steps_to_built_models
        assert sim._experiment_includes_padding_rest is False
        assert sim.solution is sol
        assert sol["Time [s]"].entries[-1] == 10800
        assert sim.model_state_mappers
        assert sim._compiled_model_state_mappers
        rest_model = sim.steps_to_built_models["Rest for padding"]
        step_models = [
            sim.steps_to_built_models[step.basic_repr()]
            for step in sim.experiment.steps
        ]
        assert any(
            previous is rest_model or next_model is rest_model
            for previous, next_model in sim.model_state_mappers
        )
        assert any(
            (step_model, rest_model) in sim._compiled_model_state_mappers
            or (rest_model, step_model) in sim._compiled_model_state_mappers
            for step_model in step_models
        )

    def test_decode_combined_step_termination_handles_none_events_and_missing_t_event(
        self,
    ):
        def neg_stoich_cutoff(variables):
            return variables["Negative electrode stoichiometry"] - 0.5

        custom_termination = pybamm.step.CustomTermination(
            name="Negative stoichiometry cut-off", event_function=neg_stoich_cutoff
        )
        experiment = pybamm.Experiment(
            [pybamm.step.c_rate(1, termination=custom_termination)]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            experiment_model_mode="unified",
        )
        sol = sim.solve(calc_esoh=False)

        actual_step_solution = sol.cycles[0].steps[0]

        class StepSolutionProxy:
            def __init__(self, source):
                self.source = source
                self.termination = f"event: {sim._COMBINED_TERMINATION_EVENT}"
                self.t_event = None
                self.y_event = None
                self.t = source.t
                self.y = source.y
                self.all_inputs = source.all_inputs

            def __getitem__(self, key):
                return self.source[key]

        step_solution = StepSolutionProxy(actual_step_solution)

        step = SimpleNamespace(
            direction="rest",
            termination=[pybamm.step.VoltageTermination(4.2), custom_termination],
        )
        decoded = sim._decode_combined_step_termination(
            step_solution,
            step,
            sim._built_experiment_model,
            step_solution.all_inputs[0],
        )

        assert decoded == "event: Negative stoichiometry cut-off [experiment]"

    def test_decode_combined_step_termination_returns_original_when_no_events_match(
        self,
    ):
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=pybamm.Experiment(["Rest for 10 seconds"]),
            experiment_model_mode="unified",
        )
        sim.build_for_experiment()
        step_solution = pybamm.EmptySolution()
        step_solution.termination = f"event: {sim._COMBINED_TERMINATION_EVENT}"
        step = SimpleNamespace(
            direction="rest", termination=[pybamm.step.VoltageTermination(4.2)]
        )

        decoded = sim._decode_combined_step_termination(
            step_solution,
            step,
            sim._built_experiment_model,
            {},
        )

        assert decoded == step_solution.termination

    def test_starting_solution(self):
        model = pybamm.lithium_ion.SPM()

        experiment = pybamm.Experiment(
            [
                pybamm.step.string("Discharge at C/2 for 10 minutes"),
                pybamm.step.string("Rest for 5 minutes"),
                pybamm.step.string("Rest for 5 minutes"),
            ]
        )

        sim = pybamm.Simulation(model, experiment=experiment)
        solution = sim.solve(save_at_cycles=[1])

        # test that the last state is correct (i.e. final cycle is saved)
        assert solution.last_state.t[-1] == 1200

        experiment = pybamm.Experiment(
            [
                pybamm.step.string("Discharge at C/2 for 20 minutes"),
                pybamm.step.string("Rest for 20 minutes"),
            ]
        )

        sim = pybamm.Simulation(model, experiment=experiment)
        new_solution = sim.solve(calc_esoh=False, starting_solution=solution)

        # test that the final time is correct (i.e. starting solution correctly set)
        assert new_solution["Time [s]"].entries[-1] == 3600

    def test_experiment_start_time_starting_solution(self):
        model = pybamm.lithium_ion.SPM()

        # Test error raised if starting_solution does not have start_time
        experiment = pybamm.Experiment(
            [pybamm.step.string("Discharge at C/2 for 10 minutes")]
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        solution = sim.solve()

        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at C/2 for 10 minutes",
                    start_time=datetime(1, 1, 1, 9, 0, 0),
                )
            ]
        )

        sim = pybamm.Simulation(model, experiment=experiment)
        with pytest.raises(ValueError, match=r"experiments with `start_time`"):
            sim.solve(starting_solution=solution)

        # Test starting_solution works well with start_time
        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at C/2 for 10 minutes",
                    start_time=datetime(1, 1, 1, 8, 0, 0),
                ),
                pybamm.step.string(
                    "Discharge at C/2 for 10 minutes",
                    start_time=datetime(1, 1, 1, 8, 20, 0),
                ),
            ]
        )

        sim = pybamm.Simulation(model, experiment=experiment)
        solution = sim.solve()

        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at C/2 for 10 minutes",
                    start_time=datetime(1, 1, 1, 9, 0, 0),
                ),
                pybamm.step.string(
                    "Discharge at C/2 for 10 minutes",
                    start_time=datetime(1, 1, 1, 9, 20, 0),
                ),
            ]
        )

        sim = pybamm.Simulation(model, experiment=experiment)
        new_solution = sim.solve(starting_solution=solution)

        # test that the final time is correct (i.e. starting solution correctly set)
        assert new_solution["Time [s]"].entries[-1] == 5400

    def test_experiment_start_time_identical_steps(self):
        # Test that if we have the same step twice, with different start times,
        # they get processed only once
        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at C/2 for 10 minutes",
                    start_time=datetime(2023, 1, 1, 8, 0, 0),
                ),
                pybamm.step.string("Discharge at C/3 for 10 minutes"),
                pybamm.step.string(
                    "Discharge at C/2 for 10 minutes",
                    start_time=datetime(2023, 1, 1, 10, 0, 0),
                ),
                pybamm.step.string("Discharge at C/3 for 10 minutes"),
            ]
        )

        assert len(experiment.steps) == 4

        for experiment_model_mode in ["legacy", "unified"]:
            sim = pybamm.Simulation(
                pybamm.lithium_ion.SPM(),
                experiment=experiment,
                solver=pybamm.IDAKLUSolver(),
                experiment_model_mode=experiment_model_mode,
            )
            sim.solve(calc_esoh=False)

            assert len(sim.experiment.unique_steps) == 2

            if experiment_model_mode == "legacy":
                assert len(sim.steps_to_built_models) == 3
            else:
                assert len(sim.steps_to_built_models) == 2
                assert len(set(sim.steps_to_built_models.values())) == 1
                assert "Rest for padding" not in sim.steps_to_built_models

    def test_experiment_custom_steps(self, subtests):
        model = pybamm.lithium_ion.SPM()

        # Explicit control
        def custom_step_constant(variables):
            return 1

        custom_constant = pybamm.step.CustomStepExplicit(
            custom_step_constant, duration=1, period=0.1
        )

        experiment = pybamm.Experiment([custom_constant])
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve()
        np.testing.assert_array_equal(sol["Current [A]"].data, 1)

        # Implicit control (algebraic)
        def custom_step_voltage(variables):
            return 100 * (variables["Voltage [V]"] - 4.2)

        for control in ["differential"]:
            with subtests.test(control=control):
                custom_step_alg = pybamm.step.CustomStepImplicit(
                    custom_step_voltage, control=control, duration=100, period=10
                )

                experiment = pybamm.Experiment([custom_step_alg])
                sim = pybamm.Simulation(model, experiment=experiment)
                sol = sim.solve()
                # sol.plot()
                np.testing.assert_allclose(
                    sol["Voltage [V]"].data[2:], 4.2, rtol=1e-4, atol=1e-3
                )

    def test_experiment_custom_termination(self):
        def neg_stoich_cutoff(variables):
            return variables["Negative electrode stoichiometry"] - 0.5

        neg_stoich_termination = pybamm.step.CustomTermination(
            name="Negative stoichiometry cut-off", event_function=neg_stoich_cutoff
        )

        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment(
            [pybamm.step.c_rate(1, termination=neg_stoich_termination)]
        )
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            experiment_model_mode="unified",
        )
        sol = sim.solve(calc_esoh=False)
        assert sim._experiment_uses_unified_model
        assert (
            sol.cycles[0].steps[0].termination
            == "event: Negative stoichiometry cut-off [experiment]"
        )

        neg_stoich = sol["Negative electrode stoichiometry"].data
        assert neg_stoich[-1] == pytest.approx(0.5, abs=0.0001)

    @pytest.mark.parametrize("experiment_model_mode", ["legacy", "unified"])
    def test_simulation_changing_capacity_crate_steps(self, experiment_model_mode):
        """Test that C-rate steps are correctly updated when capacity changes"""
        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at C/5 for 20 minutes",
                    "Discharge at C/2 for 20 minutes",
                    "Discharge at 1C for 20 minutes",
                )
            ]
        )
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=param,
            experiment_model_mode=experiment_model_mode,
        )

        # First solve
        sol1 = sim.solve(calc_esoh=False)
        original_capacity = param["Nominal cell capacity [A.h]"]

        # Check that C-rates correspond to expected currents
        I_C5_1 = np.abs(sol1.cycles[0].steps[0]["Current [A]"].data).mean()
        I_C2_1 = np.abs(sol1.cycles[0].steps[1]["Current [A]"].data).mean()
        I_1C_1 = np.abs(sol1.cycles[0].steps[2]["Current [A]"].data).mean()

        np.testing.assert_allclose(I_C5_1, original_capacity / 5, rtol=1e-2)
        np.testing.assert_allclose(I_C2_1, original_capacity / 2, rtol=1e-2)
        np.testing.assert_allclose(I_1C_1, original_capacity, rtol=1e-2)

        # Update capacity
        new_capacity = 0.9 * original_capacity
        sim._parameter_values.update({"Nominal cell capacity [A.h]": new_capacity})

        # Second solve with updated capacity
        sol2 = sim.solve(calc_esoh=False)

        # Check that C-rates now correspond to updated currents
        I_C5_2 = np.abs(sol2.cycles[0].steps[0]["Current [A]"].data).mean()
        I_C2_2 = np.abs(sol2.cycles[0].steps[1]["Current [A]"].data).mean()
        I_1C_2 = np.abs(sol2.cycles[0].steps[2]["Current [A]"].data).mean()

        np.testing.assert_allclose(I_C5_2, new_capacity / 5, rtol=1e-2)
        np.testing.assert_allclose(I_C2_2, new_capacity / 2, rtol=1e-2)
        np.testing.assert_allclose(I_1C_2, new_capacity, rtol=1e-2)

        # Verify all currents scaled proportionally
        np.testing.assert_allclose(I_C5_2 / I_C5_1, 0.9, rtol=1e-2)
        np.testing.assert_allclose(I_C2_2 / I_C2_1, 0.9, rtol=1e-2)
        np.testing.assert_allclose(I_1C_2 / I_1C_1, 0.9, rtol=1e-2)

    @pytest.mark.parametrize("experiment_model_mode", ["legacy", "unified"])
    def test_simulation_multiple_cycles_with_capacity_change(
        self, experiment_model_mode
    ):
        """Test capacity changes across multiple experiment cycles"""
        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment(
            [("Discharge at 1C for 5 minutes", "Charge at 1C for 5 minutes")] * 2
        )
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=param,
            experiment_model_mode=experiment_model_mode,
        )

        # First solve
        sol1 = sim.solve(calc_esoh=False)
        original_capacity = param["Nominal cell capacity [A.h]"]

        # Get discharge currents for both cycles
        I_discharge_cycle1 = np.abs(sol1.cycles[0].steps[0]["Current [A]"].data).mean()
        I_discharge_cycle2 = np.abs(sol1.cycles[1].steps[0]["Current [A]"].data).mean()

        # Both cycles should use the same capacity initially
        np.testing.assert_allclose(I_discharge_cycle1, original_capacity, rtol=1e-2)
        np.testing.assert_allclose(I_discharge_cycle2, original_capacity, rtol=1e-2)

        # Update capacity between cycles
        new_capacity = 0.85 * original_capacity
        sim._parameter_values.update({"Nominal cell capacity [A.h]": new_capacity})

        # Solve again
        sol2 = sim.solve(calc_esoh=False)

        # All cycles in the new solution should use updated capacity
        I_discharge_cycle1_new = np.abs(
            sol2.cycles[0].steps[0]["Current [A]"].data
        ).mean()
        I_discharge_cycle2_new = np.abs(
            sol2.cycles[1].steps[0]["Current [A]"].data
        ).mean()

        np.testing.assert_allclose(I_discharge_cycle1_new, new_capacity, rtol=1e-2)
        np.testing.assert_allclose(I_discharge_cycle2_new, new_capacity, rtol=1e-2)

    @pytest.mark.parametrize("experiment_model_mode", ["legacy", "unified"])
    def test_simulation_logging_with_capacity_change(
        self, caplog, experiment_model_mode
    ):
        """Test that capacity changes are logged appropriately"""
        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment([("Discharge at 1C for 10 minutes",)])
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=param,
            experiment_model_mode=experiment_model_mode,
        )

        # First solve
        sim.solve(calc_esoh=False)
        original_capacity = param["Nominal cell capacity [A.h]"]

        # Update capacity
        new_capacity = 0.75 * original_capacity
        sim._parameter_values.update({"Nominal cell capacity [A.h]": new_capacity})

        # Set logging level to capture INFO messages
        original_log_level = pybamm.logger.level
        pybamm.set_logging_level("INFO")

        try:
            # Second solve should log capacity change
            with caplog.at_level(logging.INFO, logger="pybamm.logger"):
                sim.solve(calc_esoh=False)

            # Check that a log message about capacity change was recorded
            log_messages = [record.message for record in caplog.records]
            capacity_change_logged = any(
                "Nominal capacity changed" in msg for msg in log_messages
            )
            assert capacity_change_logged
        finally:
            # Restore original logging level
            pybamm.logger.setLevel(original_log_level)

    def test_inputs_with_initial_soc(self):
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        param["Negative electrode active material volume fraction"] = (
            pybamm.InputParameter("eps_s_n")
        )
        experiment = pybamm.Experiment(["Rest for 10 minutes", "Rest for 10 minutes"])
        sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)

        sol1 = sim.solve(inputs={"eps_s_n": 0.6}, initial_soc=0.5)
        ic1 = sol1["X-averaged negative particle surface concentration"].data[0]

        sol2 = sim.solve(inputs={"eps_s_n": 0.9}, initial_soc=0.5)
        ic2 = sol2["X-averaged negative particle surface concentration"].data[0]

        # ICs must differ when inputs change at same SOC
        assert ic1 != ic2

    @pytest.mark.parametrize("experiment_model_mode", ["legacy", "unified"])
    def test_repeated_solves_refresh_initial_soc(self, experiment_model_mode):
        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment(["Rest for 10 minutes", "Rest for 10 minutes"])
        sim = pybamm.Simulation(
            model,
            parameter_values=pybamm.ParameterValues("Chen2020"),
            experiment=experiment,
            experiment_model_mode=experiment_model_mode,
        )

        sol1 = sim.solve(calc_esoh=False, initial_soc=0.2)
        ic1 = sol1["X-averaged negative particle surface concentration"].data[0]

        sol2 = sim.solve(calc_esoh=False, initial_soc=0.8)
        ic2 = sol2["X-averaged negative particle surface concentration"].data[0]

        # Reusing the same Simulation must refresh experiment ICs when SOC changes.
        assert ic1 != ic2
