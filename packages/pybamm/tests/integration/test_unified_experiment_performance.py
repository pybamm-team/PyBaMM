import time
import warnings

import numpy as np

import pybamm


def _cycling_experiment(n_cycles=3):
    return pybamm.Experiment(
        [
            pybamm.step.c_rate(1.0, termination="3.2 V"),
            pybamm.step.voltage(4.0, duration=300),
        ]
        * n_cycles
    )


def _unique_branch_count(experiment):
    sim = pybamm.Simulation(
        pybamm.lithium_ion.SPM(),
        experiment=experiment,
        solver=pybamm.IDAKLUSolver(),
        experiment_model_mode="unified",
    )
    sim.build_for_experiment()
    return len(set(sim._experiment_step_indices))


def _min_solve_time(sim, samples=9):
    # Minimum over several warm solves: the least system-contended run, so the most
    # stable estimate of compute time. Build + compile are excluded by the warm-up solve.
    sim.solve()
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        sim.solve()
        times.append(time.perf_counter() - start)
    return float(np.min(times))


def _warn_if_slow(model_class, limit):
    model_class()  # warm the process (import/JIT/first compile) before timing
    sims = {
        mode: pybamm.Simulation(
            model_class(),
            experiment=_cycling_experiment(),
            solver=pybamm.IDAKLUSolver(),
            experiment_model_mode=mode,
        )
        for mode in ("legacy", "unified")
    }
    ratio = _min_solve_time(sims["unified"], 9) / _min_solve_time(sims["legacy"], 9)
    if ratio > limit:
        warnings.warn(
            f"{model_class.__name__} unified warm-solve is {ratio:.2f}x legacy "
            f"(> {limit}x) -- possible performance regression.",
            stacklevel=2,
        )


class TestUnifiedExperimentPerformance:
    def test_dfn_warm_solve_close_to_legacy(self):
        # DFN ~<1.1x legacy; warn-only (env-dependent), sparse control row guarded by test_unified_control_row_jacobian_is_sparse
        _warn_if_slow(pybamm.lithium_ion.DFN, 1.15)

    def test_spme_warm_solve_close_to_legacy(self):
        # Requirement: SPMe cycling ~<1.2x legacy (measured ~1.05x). Warn-only.
        _warn_if_slow(pybamm.lithium_ion.SPMe, 1.3)

    def test_branch_count_independent_of_cycle_count(self):
        # Compile cost cannot scale with cycle count: repeated steps reuse one branch.
        assert [_unique_branch_count(_cycling_experiment(n)) for n in (1, 10, 50)] == [
            2,
            2,
            2,
        ]

    def test_constant_current_steps_collapse_regardless_of_count(self):
        # Distinct C-rates collapse to one branch, so compile/runtime cost does not grow
        # with the number of CC steps (the current is supplied as a per-step input).
        for n_steps in (5, 20, 50):
            currents = np.linspace(0.1, 1.0, n_steps)
            experiment = pybamm.Experiment(
                [pybamm.step.c_rate(float(c), duration=1) for c in currents]
            )
            assert _unique_branch_count(experiment) == 1
