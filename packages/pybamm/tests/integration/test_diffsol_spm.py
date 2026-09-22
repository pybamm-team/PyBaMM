"""Integration tests: SPM solved via the Rust diffsol BDF solver."""

import numpy as np
import pytest

import pybamm


class TestDiffsolSPM:
    def test_spm_basic_solve(self):
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
        assert voltage.entries.min() > 2.5
        assert voltage.entries.max() < 4.5

    def test_spm_voltage_trajectory(self):
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
        solver = pybamm.DiffsolSolver(root_method="nonlinear_solver")
        assert solver.root_method is not None
        assert solver.root_tol == 1e-6

    def test_diffsol_dfn_default_solves(self):
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
        model = pybamm.lithium_ion.DFN()
        solver = pybamm.DiffsolSolver(rtol=1e-6, atol=1e-6, root_method="casadi")
        sim = pybamm.Simulation(model, solver=solver)

        t_eval = np.linspace(0, 100, 10)
        solution = sim.solve(t_eval)

        assert solution is not None
        assert solution.termination == "final time"

    def test_diffsol_calc_ic_native_spm(self):
        model = pybamm.lithium_ion.SPM()
        solver = pybamm.DiffsolSolver(rtol=1e-6, atol=1e-6, calc_ic=True)
        sim = pybamm.Simulation(model, solver=solver)

        t_eval = np.linspace(0, 100, 10)
        solution = sim.solve(t_eval)

        assert solution is not None
        assert solution.termination == "final time"

    def test_diffsol_spm_unchanged_after_dae_fix(self):
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
        model = pybamm.lithium_ion.SPM()
        solver = pybamm.DiffsolSolver(output_variables=["Voltage [V]"])
        sim = pybamm.Simulation(model, solver=solver)

        sol = sim.solve([0, 3600])

        voltage = sol["Voltage [V]"].data
        assert len(voltage) > 0
        assert voltage[0] > 3.0
        assert voltage[0] < 5.0

    def test_diffsol_output_variables_with_event(self):
        model = pybamm.lithium_ion.SPM()
        model.events.append(
            pybamm.Event(
                "Voltage cutoff",
                model.variables["Voltage [V]"] - 3.2,
                pybamm.EventType.TERMINATION,
            )
        )

        solver = pybamm.DiffsolSolver(output_variables=["Voltage [V]"])
        sim = pybamm.Simulation(model, solver=solver)

        sol = sim.solve([0, 7200])

        assert sol.termination.startswith("event")
        assert sol.t[-1] < 7200
        voltage_end = sol["Voltage [V]"].data[-1]
        assert abs(voltage_end - 3.2) < 0.01


class TestDiffsolCubicOCP:
    @staticmethod
    def _cubic_ocp_parameter_values():
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
        # Off diffsol's grid this would measure interpolation error, not solver error.
        t_common = sol_diffsol.t
        np.testing.assert_allclose(
            sol_diffsol["Voltage [V]"](t_common),
            sol_idaklu["Voltage [V]"](t_common),
            rtol=1e-5,
            atol=1e-6,
        )


class TestDiffsolCurrentSteps:
    def test_a_pulsed_dfn_matches_idaklu(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values["Current function [A]"] = lambda t: (
            5.0 * ((t < 60) + (t >= 120) * (t < 180))
        )
        # Odd seconds stay off the corners, where the solvers' knots may differ.
        t_interp = np.arange(1.0, 240.0, 2.0)

        def run(solver):
            sim = pybamm.Simulation(
                pybamm.lithium_ion.DFN(),
                parameter_values=parameter_values,
                solver=solver,
            )
            return sim.solve([0, 240], t_interp=t_interp)

        sol_diffsol = run(pybamm.DiffsolSolver())
        sol_idaklu = run(pybamm.IDAKLUSolver())

        assert sol_diffsol.termination == "final time"
        np.testing.assert_allclose(
            sol_diffsol["Voltage [V]"](t_interp),
            sol_idaklu["Voltage [V]"](t_interp),
            atol=1e-4,
        )
