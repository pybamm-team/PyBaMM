"""Integration test: SPM solved via Rust-backed IDAKLU.

Gate test for Rust core IDAKLU integration. Builds SPM model,
converts to Rust, solves via IDAKLU, compares against CasADi.
"""

import numpy as np
import pytest

import pybamm


class TestRustIDAKLUSPM:
    """Gate test: Full SPM simulation via Rust IDAKLU backend."""

    @pytest.fixture
    def spm_model_no_events(self):
        """SPM model with events removed for fixed-time solve."""
        model = pybamm.lithium_ion.SPM()
        model.events = []
        return model

    @pytest.fixture
    def parameter_values(self):
        return pybamm.ParameterValues("Chen2020")

    def test_spm_rust_vs_casadi_solve(self, spm_model_no_events, parameter_values):
        """Compare full SPM solve: Rust backend vs CasADi backend."""
        t_eval = np.linspace(0, 3600, 100)

        # CasADi reference
        model_casadi = spm_model_no_events
        model_casadi.convert_to_format = "casadi"
        sim_casadi = pybamm.Simulation(
            model_casadi,
            parameter_values=parameter_values,
            solver=pybamm.IDAKLUSolver(),
        )
        sol_casadi = sim_casadi.solve(t_eval)

        # Rust backend
        model_rust = model_casadi.new_copy()
        model_rust.convert_to_format = "rust"
        sim_rust = pybamm.Simulation(
            model_rust,
            parameter_values=parameter_values,
            solver=pybamm.IDAKLUSolver(),
        )
        sol_rust = sim_rust.solve(t_eval)

        np.testing.assert_allclose(
            sol_rust.y,
            sol_casadi.y,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Solution trajectories differ",
        )

    def test_spm_rust_voltage_trajectory(self, spm_model_no_events, parameter_values):
        """Verify voltage trajectory is physically reasonable."""
        model = spm_model_no_events
        model.convert_to_format = "rust"
        # Use 1800s (half discharge) so the battery stays above 2.5V cutoff
        t_eval = np.linspace(0, 1800, 100)

        sim = pybamm.Simulation(
            model, parameter_values=parameter_values, solver=pybamm.IDAKLUSolver()
        )
        sol = sim.solve(t_eval)
        voltage = sol["Voltage [V]"].data

        assert voltage[0] > 4.0, "Initial voltage should be > 4V"
        assert voltage[-1] > 3.0, "Final voltage at 1800s should be > 3V"
        assert voltage[-1] < voltage[0], "Voltage should decrease"


class TestRustIDAKLUEvents:
    """Verify termination events fire correctly via the Rust IDAKLU backend."""

    def test_event_termination_matches_casadi(self):
        """A voltage cutoff event must terminate the Rust solve at the same
        time, on the same event, as the CasADi backend."""
        # High discharge current so the "Minimum voltage [V]" event fires well
        # before the final time.
        parameter_values = pybamm.ParameterValues("Marquis2019")
        parameter_values["Current function [A]"] = 5.0
        t_eval = [0, 3600]
        t_interp = np.linspace(0, 3600, 1000)

        def solve(fmt):
            model = pybamm.lithium_ion.SPM()  # has events by default
            model.convert_to_format = fmt
            solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
            sim = pybamm.Simulation(
                model, parameter_values=parameter_values, solver=solver
            )
            return sim.solve(t_eval=t_eval, t_interp=t_interp)

        sol_casadi = solve("casadi")
        sol_rust = solve("rust")

        # The event actually fired (did not run to the final time).
        assert sol_rust.t[-1] < 3600.0
        assert str(sol_rust.termination).startswith("event")
        # Same event identified, and same termination time as CasADi.
        assert sol_rust.termination == sol_casadi.termination
        np.testing.assert_allclose(sol_rust.t[-1], sol_casadi.t[-1], rtol=1e-6)
        np.testing.assert_allclose(
            sol_rust["Voltage [V]"].entries[-1],
            sol_casadi["Voltage [V]"].entries[-1],
            rtol=1e-6,
            atol=1e-8,
        )


class TestRustCubicOCP:
    """SPM with cubic data-interpolant OCPs solves on the Rust paths."""

    @staticmethod
    def _cubic_ocp_parameter_values():
        # Sample Chen2020's closed-form OCPs into data, rebuild as CUBIC
        # interpolants so the model carries 1D cubic interpolants.
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

    def test_cubic_ocp_idaklu_rust_vs_casadi(self):
        model = pybamm.lithium_ion.SPM()
        model.events = []
        pv = self._cubic_ocp_parameter_values()
        t_eval = np.linspace(0, 1800, 50)

        model_casadi = model.new_copy()
        model_casadi.convert_to_format = "casadi"
        sol_casadi = pybamm.Simulation(
            model_casadi,
            parameter_values=pv,
            solver=pybamm.IDAKLUSolver(),
        ).solve(t_eval)

        model_rust = model.new_copy()
        model_rust.convert_to_format = "rust"
        sol_rust = pybamm.Simulation(
            model_rust,
            parameter_values=pv,
            solver=pybamm.IDAKLUSolver(),
        ).solve(t_eval)

        np.testing.assert_allclose(
            sol_rust["Voltage [V]"].data,
            sol_casadi["Voltage [V]"].data,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Rust cubic-OCP voltage differs from CasADi",
        )


class TestRustECMNDInterpolants:
    """Thevenin ECM with 3D (r0/r1/c1) and 2D (dUdT) data tables solves on
    the Rust evaluator path and matches CasADi."""

    def test_ecm_idaklu_rust_vs_casadi(self):
        model = pybamm.equivalent_circuit.Thevenin()
        model.events = []
        pv = pybamm.ParameterValues("ECM_Example")
        t_eval = np.linspace(0, 600, 50)

        model_casadi = model.new_copy()
        model_casadi.convert_to_format = "casadi"
        sol_casadi = pybamm.Simulation(
            model_casadi,
            parameter_values=pv,
            solver=pybamm.IDAKLUSolver(),
        ).solve(t_eval)

        model_rust = model.new_copy()
        model_rust.convert_to_format = "rust"
        sol_rust = pybamm.Simulation(
            model_rust,
            parameter_values=pv,
            solver=pybamm.IDAKLUSolver(),
        ).solve(t_eval)

        np.testing.assert_allclose(
            sol_rust["Voltage [V]"].data,
            sol_casadi["Voltage [V]"].data,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Rust ECM ND-interpolant voltage differs from CasADi",
        )
