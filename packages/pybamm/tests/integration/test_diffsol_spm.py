"""Integration test: SPM solved via Rust diffsol BDF solver.

End-to-end test for the DiffsolSolver. Builds an SPM model,
solves it via the Rust diffsol BDF backend, and compares
against the CasADi solver for parity.
"""

import numpy as np
import pytest

import pybamm


class TestDiffsolSPM:
    """Integration tests for DiffsolSolver with SPM model."""

    def test_spm_basic_solve(self):
        """Verify DiffsolSolver can solve a basic SPM discharge."""
        model = pybamm.lithium_ion.SPM()
        solver = pybamm.DiffsolSolver(rtol=1e-6, atol=1e-6)
        sim = pybamm.Simulation(model, solver=solver)

        t_eval = np.linspace(0, 3600, 100)
        solution = sim.solve(t_eval)

        assert solution is not None
        assert solution.termination == "final time"
        assert len(solution.t) > 0

        voltage = solution["Voltage [V]"]
        assert voltage is not None
        # SPM voltage should be in a reasonable range
        assert voltage.entries.min() > 2.5
        assert voltage.entries.max() < 4.5

    def test_spm_voltage_trajectory(self):
        """Verify voltage trajectory is physically reasonable."""
        model = pybamm.lithium_ion.SPM()
        solver = pybamm.DiffsolSolver(rtol=1e-6, atol=1e-6)
        sim = pybamm.Simulation(model, solver=solver)

        t_eval = np.linspace(0, 1800, 100)
        sol = sim.solve(t_eval)

        voltage = sol["Voltage [V]"].entries
        assert voltage[0] > 3.5, "Initial voltage should be > 3.5V"
        assert voltage[-1] > 3.0, "Final voltage at 1800s should be > 3V"
        assert voltage[-1] < voltage[0], "Voltage should decrease during discharge"

    def test_diffsol_matches_casadi(self):
        """Verify DiffsolSolver produces results close to CasadiSolver.

        SPM carries an algebraic ``Voltage [V]`` equation (the voltage-as-a-state
        default), so this is a DAE. The diffsol and CasADi BDF implementations
        agree to ~1e-7 on the DAE path, so the parity tolerance is looser than
        the solver tolerance (1e-8) while still well below the 1e-4 used by the
        other cross-implementation parity tests in this module.
        """
        model_ds = pybamm.lithium_ion.SPM()
        model_cs = pybamm.lithium_ion.SPM()
        model_cs.convert_to_format = "casadi"

        t_eval = np.linspace(0, 3600, 100)

        sol_ds = pybamm.Simulation(
            model_ds, solver=pybamm.DiffsolSolver(rtol=1e-8, atol=1e-8)
        ).solve(t_eval)
        sol_cs = pybamm.Simulation(
            model_cs, solver=pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        ).solve(t_eval)

        v_ds = sol_ds["Voltage [V]"].entries
        v_cs = sol_cs["Voltage [V]"].entries

        np.testing.assert_allclose(
            v_ds,
            v_cs,
            rtol=1e-6,
            atol=1e-7,
            err_msg="DiffsolSolver voltage differs from CasadiSolver",
        )

    def test_diffsol_explicit_nonlinear_solver_no_crash(self):
        """Verify DiffsolSolver(root_method='nonlinear_solver') doesn't crash on construction."""
        solver = pybamm.DiffsolSolver(root_method="nonlinear_solver")
        assert solver.root_method is not None
        assert solver.root_tol == 1e-6

    def test_diffsol_calc_ic_parameter_exists(self):
        """Verify calc_ic parameter is accepted and creates _internal_initialisation property."""
        solver = pybamm.DiffsolSolver(calc_ic=True)
        assert solver._internal_initialisation is True

        solver_default = pybamm.DiffsolSolver()
        assert solver_default._internal_initialisation is False

    def test_diffsol_dfn_default_solves(self):
        """Verify DiffsolSolver() solves DFN (DAE) model without explicit root_method."""
        model = pybamm.lithium_ion.DFN()
        solver = pybamm.DiffsolSolver(rtol=1e-6, atol=1e-6)
        sim = pybamm.Simulation(model, solver=solver)

        t_eval = np.linspace(0, 100, 10)
        solution = sim.solve(t_eval)

        assert solution is not None
        assert solution.termination == "final time"

        voltage = solution["Voltage [V]"]
        assert voltage.entries.min() > 2.5
        assert voltage.entries.max() < 4.5

    def test_diffsol_explicit_casadi_dfn(self):
        """Verify DiffsolSolver(root_method='casadi') works for DFN."""
        model = pybamm.lithium_ion.DFN()
        solver = pybamm.DiffsolSolver(rtol=1e-6, atol=1e-6, root_method="casadi")
        sim = pybamm.Simulation(model, solver=solver)

        t_eval = np.linspace(0, 100, 10)
        solution = sim.solve(t_eval)

        assert solution is not None
        assert solution.termination == "final time"

    def test_diffsol_calc_ic_native_spm(self):
        """Verify DiffsolSolver(calc_ic=True) works on SPM (DAE via voltage state)."""
        model = pybamm.lithium_ion.SPM()
        solver = pybamm.DiffsolSolver(rtol=1e-6, atol=1e-6, calc_ic=True)
        sim = pybamm.Simulation(model, solver=solver)

        t_eval = np.linspace(0, 100, 10)
        solution = sim.solve(t_eval)

        assert solution is not None
        assert solution.termination == "final time"

    def test_diffsol_spm_unchanged_after_dae_fix(self):
        """Regression test: SPM should still work after DAE IC fix."""
        model = pybamm.lithium_ion.SPM()
        solver = pybamm.DiffsolSolver()
        sim = pybamm.Simulation(model, solver=solver)

        t_eval = np.linspace(0, 3600, 100)
        solution = sim.solve(t_eval)

        assert solution is not None
        assert solution.termination == "final time"
        voltage = solution["Voltage [V]"]
        assert voltage.entries.min() > 2.5
        assert voltage.entries.max() < 4.5

    def test_diffsol_output_variables(self):
        """DiffsolSolver with output_variables should request output rows."""
        model = pybamm.lithium_ion.SPM()
        solver = pybamm.DiffsolSolver(output_variables=["Voltage [V]"])
        sim = pybamm.Simulation(model, solver=solver)

        sol = sim.solve([0, 3600])

        # Should be able to access the requested variable
        voltage = sol["Voltage [V]"].data
        assert len(voltage) > 0
        assert voltage[0] > 3.0  # Reasonable voltage range
        assert voltage[0] < 5.0

    def test_diffsol_output_variables_with_event(self):
        """Event detection should work with output-only solve path."""
        model = pybamm.lithium_ion.SPM()
        # Add a voltage cutoff event
        model.events.append(
            pybamm.Event(
                "Voltage cutoff",
                model.variables["Voltage [V]"] - 3.2,
                pybamm.EventType.TERMINATION,
            )
        )

        solver = pybamm.DiffsolSolver(output_variables=["Voltage [V]"])
        sim = pybamm.Simulation(model, solver=solver)

        sol = sim.solve([0, 7200])  # Long enough to hit cutoff

        # Should terminate early due to event
        assert sol.termination.startswith("event")
        assert sol.t[-1] < 7200
        # Voltage at end should be near cutoff
        voltage_end = sol["Voltage [V]"].data[-1]
        assert abs(voltage_end - 3.2) < 0.01


class TestDiffsolCubicOCP:
    """SPM with cubic data-interpolant OCPs solves on the diffsol path."""

    @staticmethod
    def _cubic_ocp_parameter_values():
        # Sample Chen2020's closed-form OCPs into data and rebuild them as CUBIC
        # interpolants, so the model carries 1D cubic interpolants.
        pv = pybamm.ParameterValues("Chen2020")
        sto = np.linspace(0.0, 1.0, 200)
        un_cb = pv["Negative electrode OCP [V]"]
        up_cb = pv["Positive electrode OCP [V]"]
        un_data = np.array([float(un_cb(pybamm.Scalar(s)).evaluate()) for s in sto])
        up_data = np.array([float(up_cb(pybamm.Scalar(s)).evaluate()) for s in sto])

        def make_cubic_ocp(x_data, y_data, name):
            def ocp(stoich):
                return pybamm.Interpolant(
                    x_data, y_data, stoich, name=name, interpolator="cubic"
                )

            return ocp

        pv["Negative electrode OCP [V]"] = make_cubic_ocp(sto, un_data, "Un")
        pv["Positive electrode OCP [V]"] = make_cubic_ocp(sto, up_data, "Up")
        return pv

    def test_cubic_ocp_diffsol_vs_casadi(self):
        model = pybamm.lithium_ion.SPM()
        model.events = []
        pv = self._cubic_ocp_parameter_values()
        t_eval = np.linspace(0, 1800, 50)

        # Pass t_interp so IDAKLU outputs exactly on t_eval (not internal steps).
        ref_model = model.new_copy()
        ref_model.convert_to_format = "casadi"
        sol_casadi = pybamm.Simulation(
            ref_model,
            parameter_values=pv,
            solver=pybamm.IDAKLUSolver(),
        ).solve(t_eval, t_interp=t_eval)

        sol_diffsol = pybamm.Simulation(
            model.new_copy(),
            parameter_values=pv,
            solver=pybamm.DiffsolSolver(),
        ).solve(t_eval)

        np.testing.assert_allclose(
            sol_diffsol["Voltage [V]"].data,
            sol_casadi["Voltage [V]"].data,
            rtol=1e-4,
            atol=1e-6,
            err_msg="Diffsol cubic-OCP voltage differs from CasADi",
        )


class TestDiffsolECMNDInterpolants:
    """Thevenin ECM with 3D/2D data tables solves on the diffsol path."""

    def test_ecm_diffsol_vs_casadi(self):
        model = pybamm.equivalent_circuit.Thevenin()
        model.events = []
        pv = pybamm.ParameterValues("ECM_Example")
        t_eval = np.linspace(0, 600, 50)

        # Pass t_interp so IDAKLU outputs exactly on t_eval (not internal steps).
        ref_model = model.new_copy()
        ref_model.convert_to_format = "casadi"
        sol_casadi = pybamm.Simulation(
            ref_model,
            parameter_values=pv,
            solver=pybamm.IDAKLUSolver(),
        ).solve(t_eval, t_interp=t_eval)

        sol_diffsol = pybamm.Simulation(
            model.new_copy(),
            parameter_values=pv,
            solver=pybamm.DiffsolSolver(),
        ).solve(t_eval)

        np.testing.assert_allclose(
            sol_diffsol["Voltage [V]"].data,
            sol_casadi["Voltage [V]"].data,
            rtol=1e-4,
            atol=1e-6,
            err_msg="Diffsol ECM ND-interpolant voltage differs from CasADi",
        )


class TestDiffsolExperiment:
    """Experiments re-build the rust model per step and stitch segment
    solutions; diffsol had no coverage of either."""

    def test_two_step_experiment_matches_idaklu(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at 1C for 5 minutes",
                "Rest for 5 minutes",
            ]
        )

        def run(solver):
            sim = pybamm.Simulation(
                pybamm.lithium_ion.SPM(), experiment=experiment, solver=solver
            )
            return sim.solve()

        sol_diffsol = run(pybamm.DiffsolSolver(rtol=1e-8, atol=1e-8))
        sol_idaklu = run(pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8))

        # A flat instruction list makes one cycle per instruction.
        assert len(sol_diffsol.cycles) == 2
        assert sol_diffsol.t[-1] == pytest.approx(600.0)
        # Compare on diffsol's own grid: its observation is grid-aligned, so
        # off-grid points would measure interpolation error, not solver error.
        t_common = sol_diffsol.t
        np.testing.assert_allclose(
            sol_diffsol["Voltage [V]"](t_common),
            sol_idaklu["Voltage [V]"](t_common),
            rtol=1e-5,
            atol=1e-6,
        )
