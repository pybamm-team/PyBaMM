"""
Regression tests for historical electrode state of health (eSOH) bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import numpy as np
import pytest

import pybamm


class TestESOHHalfCellFixes:
    """Guards for eSOH half-cell model bug fixes."""

    def test_initial_soc_with_drive_cycle_half_cell(self):
        """
        Guards against: PR #3467 - fix esoh bug

        The bug caused `initial_soc` to error when using a drive cycle for
        half-cell models. This test verifies initial_soc works correctly
        with drive cycles in half-cell configuration.
        """
        model = pybamm.lithium_ion.SPM({"working electrode": "positive"})
        param = pybamm.ParameterValues("Xu2019")

        # Create a simple drive cycle
        t_data = np.linspace(0, 3600, 100)
        I_data = np.sin(t_data / 600) * 0.5 + 0.5  # Varying current

        # Use drive cycle with initial_soc
        param.update(
            {"Current function [A]": pybamm.Interpolant(t_data, I_data, pybamm.t)}
        )

        sim = pybamm.Simulation(model, parameter_values=param)

        # Should work with initial_soc
        sol = sim.solve([0, 600], initial_soc=0.8)

        assert len(sol.t) > 0
        V = sol["Voltage [V]"].data
        assert np.all(V > 2.0)
        assert np.all(V < 5.0)

    def test_half_cell_esoh_calculation_runs(self):
        """
        Verify eSOH calculation works for half-cell models.
        """
        model = pybamm.lithium_ion.SPM({"working electrode": "positive"})
        param = pybamm.ParameterValues("Xu2019")

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600], initial_soc=0.5)

        # Should complete without eSOH errors
        assert len(sol.t) > 0

        # Capacity should be reasonable
        Q = sol["Discharge capacity [A.h]"].data
        assert Q[-1] > 0


class TestESOHTrapzFixes:
    """Guards for numpy trapz deprecation fixes."""

    def test_esoh_energy_calculation(self):
        """
        Guards against: PR #5324 - fix np trapz bug

        The bug was that np.trapz is deprecated in newer NumPy versions.
        This test verifies energy calculations in eSOH work correctly.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 3600])

        # Energy calculations should work without deprecation errors
        energy = sol["Discharge energy [W.h]"].data
        assert not np.any(np.isnan(energy))
        # Energy should accumulate during discharge
        assert energy[-1] >= energy[0]

    def test_esoh_solver_convergence(self):
        """
        Verify eSOH solver converges for standard configurations.
        """
        model = pybamm.lithium_ion.DFN()
        param = pybamm.ParameterValues("Chen2020")

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 3600], initial_soc=0.8)

        # Should solve without convergence errors
        assert len(sol.t) > 0

        # Discharge capacity should increase during discharge
        Q = sol["Discharge capacity [A.h]"].data
        assert Q[-1] > Q[0]


class TestInitialConditionsFromFixes:
    """Guards for initial_conditions_from scale evaluation fixes."""

    def test_inputs_propagate_to_ic_scale_evaluation(self):
        """
        Guards against: PR #5285 - Bugfix: inputs for `initial_conditions_from`
        scale evaluation

        The bug was that inputs were not properly propagated for scale
        evaluation when calling set_initial_conditions_from(). Specifically,
        scale.evaluate(inputs=inputs) was not receiving the inputs dict.

        This test exercises the fix by:
        1. Solving with a parameter that affects IC scales as an input
        2. Using set_initial_conditions_from with that solution and inputs
        3. Verifying the scale evaluation doesn't fail
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        # Use maximum concentration as input - this affects IC scales for
        # particle concentration variables (c_n is scaled by c_n_max)
        c_n_max = param["Maximum concentration in negative electrode [mol.m-3]"]
        param.update(
            {
                "Maximum concentration in negative electrode [mol.m-3]": "[input]",
            }
        )

        sim = pybamm.Simulation(model, parameter_values=param)

        # First solve to get a solution
        sol1 = sim.solve(
            [0, 300],
            inputs={"Maximum concentration in negative electrode [mol.m-3]": c_n_max},
            initial_soc=0.8,
        )

        # Now use set_initial_conditions_from with the solution and inputs
        # This is where the bug manifested - inputs weren't passed to scale.evaluate()
        model2 = pybamm.lithium_ion.SPM()
        sim2 = pybamm.Simulation(model2, parameter_values=param)
        sim2.build()

        # This call would fail before the fix if inputs affected IC scales
        sim2.built_model.set_initial_conditions_from(
            sol1,
            inputs={"Maximum concentration in negative electrode [mol.m-3]": c_n_max},
        )

        # Solve from the continued state
        sol2 = sim2.solve(
            [0, 300],
            inputs={"Maximum concentration in negative electrode [mol.m-3]": c_n_max},
        )

        # Should complete without scale evaluation errors
        assert len(sol2.t) > 0

        # Verify concentration stays within physical bounds
        c_n = sol2["X-averaged negative particle concentration [mol.m-3]"].data
        assert np.all(c_n > 0)
        assert np.all(c_n < c_n_max)


class TestStartingSolutionFixes:
    """Guards for starting_solution bug fixes."""

    def test_last_state_as_starting_solution(self):
        """
        Guards against: PR #2822 / 6f81b35bb - Fix use of last-state as
        starting-solution in Simulation.solve()

        The bug was that using solution.last_state or a previous solution as
        starting_solution didn't work correctly. This test explicitly uses
        the starting_solution parameter to verify time continuity and that
        the state is properly transferred.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        # First solve: discharge for 20 minutes (1200s)
        exp1 = pybamm.Experiment(["Discharge at C/2 for 20 minutes"])
        sim1 = pybamm.Simulation(model, parameter_values=param, experiment=exp1)
        sol1 = sim1.solve(calc_esoh=False)

        # Verify last_state is accessible and has correct final time
        assert sol1.last_state is not None
        assert sol1.last_state.t[-1] == pytest.approx(1200, rel=1e-3)

        # Second solve: continue using starting_solution
        exp2 = pybamm.Experiment(["Discharge at C/2 for 10 minutes"])
        sim2 = pybamm.Simulation(model, parameter_values=param, experiment=exp2)
        sol2 = sim2.solve(calc_esoh=False, starting_solution=sol1)

        # The key test: time should continue from where sol1 ended
        # sol1 ended at 1200s, sol2 adds 600s, so final time should be 1800s
        assert sol2["Time [s]"].entries[-1] == pytest.approx(1800, rel=1e-3), (
            f"Expected final time 1800s, got {sol2['Time [s]'].entries[-1]}"
        )

        # Capacity should also continue accumulating
        Q_end_sol1 = sol1["Discharge capacity [A.h]"].data[-1]
        Q_end_sol2 = sol2["Discharge capacity [A.h]"].data[-1]
        assert Q_end_sol2 > Q_end_sol1, (
            "Capacity should continue to increase with starting_solution"
        )

    def test_starting_solution_with_experiment(self):
        """
        Verify starting_solution works with experiments.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        exp1 = pybamm.Experiment(
            [
                "Discharge at 1C for 30 minutes",
            ]
        )
        sim1 = pybamm.Simulation(model, parameter_values=param, experiment=exp1)
        sol1 = sim1.solve()

        exp2 = pybamm.Experiment(
            [
                "Rest for 10 minutes",
                "Discharge at 0.5C for 30 minutes",
            ]
        )
        sim2 = pybamm.Simulation(model, parameter_values=param, experiment=exp2)
        sol2 = sim2.solve(starting_solution=sol1)

        # Should complete without errors
        assert len(sol2.t) > 0


class TestESOHCompositeElectrode:
    """Tests for eSOH with composite electrodes."""

    def test_esoh_composite_electrode_runs(self):
        """
        Verify eSOH calculation works with composite electrodes.
        """
        model = pybamm.lithium_ion.SPM({"particle phases": ("2", "1")})
        param = pybamm.ParameterValues("Chen2020_composite")

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 3600], initial_soc=0.8)

        # Should complete successfully
        assert len(sol.t) > 0

        # Discharge capacity should increase during discharge
        Q = sol["Discharge capacity [A.h]"].data
        assert Q[-1] > Q[0]

    def test_esoh_composite_capacity_tracking(self):
        """
        Verify capacity tracking works for composite electrodes.
        """
        model = pybamm.lithium_ion.DFN({"particle phases": ("2", "1")})
        param = pybamm.ParameterValues("Chen2020_composite")

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 1800])

        # Capacity should be tracked correctly
        Q = sol["Discharge capacity [A.h]"].data
        assert not np.any(np.isnan(Q))
        assert Q[-1] > 0

        # Both phases should contribute to capacity
        c_primary = sol[
            "X-averaged negative primary particle concentration [mol.m-3]"
        ].data
        c_secondary = sol[
            "X-averaged negative secondary particle concentration [mol.m-3]"
        ].data

        # Both should change during discharge
        # Flatten to get scalar values (handles multi-dimensional arrays)
        c_prim_start = np.mean(c_primary[0])
        c_prim_end = np.mean(c_primary[-1])
        c_sec_start = np.mean(c_secondary[0])
        c_sec_end = np.mean(c_secondary[-1])
        assert not np.isclose(c_prim_end, c_prim_start)
        assert not np.isclose(c_sec_end, c_sec_start)


class TestESOHInputParameters:
    """Tests for eSOH with input parameters."""

    def test_esoh_with_varying_inputs(self):
        """
        Verify eSOH works when solving multiple times with different inputs.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        param.update({"Current function [A]": "[input]"})

        sim = pybamm.Simulation(model, parameter_values=param)

        # Solve with different currents at same initial SOC
        sol_1C = sim.solve(
            [0, 600], inputs={"Current function [A]": 5.0}, initial_soc=0.8
        )
        sol_2C = sim.solve(
            [0, 600], inputs={"Current function [A]": 10.0}, initial_soc=0.8
        )

        # Both should work
        assert len(sol_1C.t) > 0
        assert len(sol_2C.t) > 0

        # Higher C-rate should discharge more capacity in same time
        Q_1C = sol_1C["Discharge capacity [A.h]"].data[-1]
        Q_2C = sol_2C["Discharge capacity [A.h]"].data[-1]
        assert Q_2C > Q_1C
