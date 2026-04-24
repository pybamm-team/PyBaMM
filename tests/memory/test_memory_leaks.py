import gc
import tracemalloc

import pybamm


class TestMemoryAccumulation:
    """
    Tests that verify memory doesn't grow unboundedly over repeated operations.

    Uses tracemalloc to measure Python memory growth. This approach handles
    C++ extension memory correctly (idaklu solver, CasADi), unlike memray's
    limit_leaks which can show false positives from extension allocations.
    """

    def test_repeated_solve_bounded_memory(self):
        """
        Repeated solves on the same simulation should have bounded memory.

        After warmup, memory should plateau - not grow linearly with solves.
        """
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sim.solve([0, 3600])  # warmup

        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]

        # Run enough iterations to detect linear growth vs plateau
        for _ in range(100):
            sim.solve([0, 3600])

        gc.collect()
        final, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        retained_kb = (final - baseline) / 1024
        # Memory should plateau well under 200 KB for 100 solves
        # Linear leak of 2 KB/solve would show as 200 KB
        assert retained_kb < 200, (
            f"Memory grew to {retained_kb:.1f} KB after 100 solves. "
            f"Expected bounded growth (plateau), not linear accumulation."
        )

    def test_repeated_dfn_solve_bounded_memory(self):
        """Repeated DFN solves should have bounded memory."""
        model = pybamm.lithium_ion.DFN()
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.solve([0, 3600])  # warmup

        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]

        for _ in range(50):
            sim.solve([0, 3600])

        gc.collect()
        final, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        retained_kb = (final - baseline) / 1024
        assert retained_kb < 200, (
            f"Memory grew to {retained_kb:.1f} KB after 50 DFN solves."
        )

    def test_repeated_experiment_bounded_memory(self):
        """Repeated experiment solves should have bounded memory."""
        experiment = pybamm.Experiment(
            ["Discharge at C/10 for 30 minutes", "Rest for 10 minutes"] * 3
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.solve()  # warmup

        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]

        for _ in range(20):
            sim.solve()

        gc.collect()
        final, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        retained_kb = (final - baseline) / 1024
        assert retained_kb < 300, (
            f"Memory grew to {retained_kb:.1f} KB after 20 experiment solves."
        )

    def test_simulation_creation_bounded_memory(self):
        """Creating many simulations should have bounded memory."""
        # Warmup - populate shared caches
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sim.solve([0, 3600])

        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]

        # First batch - caches fill up
        for _ in range(20):
            model = pybamm.lithium_ion.SPM()
            sim = pybamm.Simulation(model)
            sim.solve([0, 3600])

        gc.collect()
        after_first_batch, _ = tracemalloc.get_traced_memory()

        # Second batch - should not grow significantly if bounded
        for _ in range(20):
            model = pybamm.lithium_ion.SPM()
            sim = pybamm.Simulation(model)
            sim.solve([0, 3600])

        gc.collect()
        after_second_batch, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        first_batch_kb = (after_first_batch - baseline) / 1024
        second_batch_growth_kb = (after_second_batch - after_first_batch) / 1024

        # First batch fills caches (can be several MB)
        # Second batch should add minimal memory if caches are bounded
        assert second_batch_growth_kb < 200, (
            f"Memory grew {second_batch_growth_kb:.1f} KB in second batch of 20 "
            f"simulations. First batch used {first_batch_kb:.1f} KB for cache "
            f"population. Second batch growth should be minimal if caches bounded."
        )


class TestSolutionMemory:
    """Tests for Solution object memory behavior."""

    def test_solution_variable_access_bounded(self):
        """
        Repeated access to solution variables should have bounded memory.

        Variables are cached on first access. Repeated access should
        hit the cache without new allocations.
        """
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 3600])

        # Warmup: first access populates variable cache
        _ = sol["Voltage [V]"].entries
        _ = sol["Current [A]"].entries
        _ = sol["Time [h]"].entries

        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]

        # Repeated access should hit cache
        for _ in range(200):
            _ = sol["Voltage [V]"].entries
            _ = sol["Current [A]"].entries
            _ = sol["Time [h]"].entries
            _ = sol["Terminal voltage [V]"].entries

        gc.collect()
        final, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        retained_kb = (final - baseline) / 1024
        # Cached access should add minimal memory
        assert retained_kb < 100, (
            f"Memory grew {retained_kb:.1f} KB after 200 variable accesses. "
            f"Expected minimal growth since variables should be cached."
        )


class TestExperimentMemory:
    """Tests for experiment-specific memory behavior."""

    def test_cycle_count_memory_scaling(self):
        """Memory should NOT scale linearly with cycle count."""
        cycle = [
            "Discharge at C/5 for 30 minutes or until 3.0 V",
            "Rest for 5 minutes",
            "Charge at C/5 for 30 minutes or until 4.2 V",
            "Rest for 5 minutes",
        ]

        # Warmup
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=pybamm.Experiment(cycle * 2))
        sim.solve()

        # Measure memory for different cycle counts
        gc.collect()
        tracemalloc.start()
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=pybamm.Experiment(cycle * 5))
        sim.solve()
        gc.collect()
        mem_5_cycles, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        gc.collect()
        tracemalloc.start()
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=pybamm.Experiment(cycle * 20))
        sim.solve()
        gc.collect()
        mem_20_cycles, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 4x more cycles should NOT cause 4x memory (that would indicate
        # each step is building its own model). Allow 1.25x for solution data growth.
        ratio = mem_20_cycles / mem_5_cycles
        assert ratio < 1.25, (
            f"Memory grew {ratio:.1f}x for 4x more cycles. "
            f"Expected sub-linear growth (<2.5x). "
            f"This may indicate termination hashing is broken (see #5453)."
        )

    def test_long_cycling_bounded_memory(self):
        """Long experiments should not cause memory blowup."""
        cycle = [
            "Discharge at C/5 for 30 minutes or until 3.0 V",
            "Rest for 5 minutes",
            "Charge at C/5 for 30 minutes or until 4.2 V",
            "Rest for 5 minutes",
        ]
        experiment = pybamm.Experiment(cycle * 10)
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)

        tracemalloc.start()
        sim.solve()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        assert peak_mb < 6, (
            f"Peak memory {peak_mb:.1f} MB during 40-step experiment is excessive."
        )

    def test_cccv_memory(self):
        """CCCV protocol should have bounded memory."""
        experiment = pybamm.Experiment(
            [
                "Discharge at 1C until 3.0 V",
                "Charge at 0.5C until 4.2 V",
                "Hold at 4.2 V until C/50",
            ]
            * 5
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)

        tracemalloc.start()
        sim.solve()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        assert peak_mb < 6, f"Peak memory {peak_mb:.1f} MB for CCCV is excessive."

    def test_gitt_memory(self):
        """GITT protocol should have bounded memory."""
        experiment = pybamm.Experiment(
            ["Discharge at C/20 for 1 hour", "Rest for 1 hour"] * 10,
            period="6 minutes",
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)

        tracemalloc.start()
        sim.solve()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        assert peak_mb < 6, f"Peak memory {peak_mb:.1f} MB for GITT is excessive."
