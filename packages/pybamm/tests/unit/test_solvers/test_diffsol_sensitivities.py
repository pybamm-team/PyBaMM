"""Parity tests: diffsol native state sensitivities vs CasADi-IDAKLU oracle."""

import numpy as np
import pytest

import pybamm
from pybamm.solvers.observation import NativeObservation

# Parity assertion tolerances (mandated by task brief; do not weaken).
_PARITY_RTOL = 1e-5
_PARITY_ATOL = 1e-8

# 0D outputs: 1e-9 is the loosest tolerance at which BDF/IDA cross-integrator
# trajectory differences stay within the parity bounds above.
_SCALAR_SOLVER_TOL = 1e-9

# D_n's |p * dV/dp| ~ 1e-4 makes 1e-9's ~6e-6 oracle error a 1.7x parity margin;
# 1e-11 moves the oracle close enough that the assertion measures native error.
_TWO_PARAM_SOLVER_TOL = 1e-11

# Spatial outputs: concentration Jacobian (~246) amplifies cross-integrator
# differences; 1e-12 is required to stay within the parity bounds.
_SPATIAL_SOLVER_TOL = 1e-12


class TestDiffsolSensitivities:
    def _solve_both(
        self, output_name, calc, extra_inputs=None, solver_tol=_SCALAR_SOLVER_TOL
    ):
        params = pybamm.ParameterValues("Chen2020")
        params["Current function [A]"] = pybamm.InputParameter("I")
        inputs = {"I": 0.5}
        if extra_inputs:
            for pybamm_name, (input_name, value) in extra_inputs.items():
                params[pybamm_name] = pybamm.InputParameter(input_name)
                inputs[input_name] = value
        t_eval = np.linspace(0, 100, 15)

        m_native = pybamm.lithium_ion.SPM()
        m_native.events = []
        m_native.convert_to_format = "rust"
        sol_n = pybamm.Simulation(
            m_native,
            parameter_values=params,
            solver=pybamm.DiffsolSolver(rtol=solver_tol, atol=solver_tol),
        ).solve(t_eval, inputs=inputs, calculate_sensitivities=calc)

        m_casadi = pybamm.lithium_ion.SPM()
        m_casadi.events = []
        m_casadi.convert_to_format = "casadi"
        # t_interp=t_eval forces IDAKLUSolver to output exactly at t_eval so
        # both solutions share the same time grid for comparison.
        sol_c = pybamm.Simulation(
            m_casadi,
            parameter_values=params,
            solver=pybamm.IDAKLUSolver(rtol=solver_tol, atol=solver_tol),
        ).solve(t_eval, inputs=inputs, calculate_sensitivities=calc, t_interp=t_eval)
        return sol_n[output_name], sol_c[output_name]

    def test_sensitivities_match_casadi_0d(self):
        var_n, var_c = self._solve_both("Terminal voltage [V]", ["I"])
        sens_n, sens_c = var_n.sensitivities, var_c.sensitivities
        assert set(sens_n) == set(sens_c)
        # Non-vacuous: a real multi-point, nonzero sensitivity (not collapsed/all-zero).
        assert sens_c["I"].size > 1 and np.any(sens_c["I"] != 0)
        for key in sens_c:
            np.testing.assert_allclose(
                sens_n[key], sens_c[key], rtol=_PARITY_RTOL, atol=_PARITY_ATOL
            )

    def test_sensitivities_match_casadi_spatial(self):
        name = "Negative particle concentration [mol.m-3]"
        var_n, var_c = self._solve_both(name, ["I"], solver_tol=_SPATIAL_SOLVER_TOL)
        assert var_c.entries.shape[0] > 1
        sens_n, sens_c = var_n.sensitivities, var_c.sensitivities
        assert set(sens_n) == set(sens_c)
        for key in sens_c:
            assert sens_n[key].shape == sens_c[key].shape
            np.testing.assert_allclose(
                sens_n[key], sens_c[key], rtol=_PARITY_RTOL, atol=_PARITY_ATOL
            )

    def test_sensitivities_all_block_two_params(self):
        var_n, var_c = self._solve_both(
            "Terminal voltage [V]",
            ["D_n", "I"],
            extra_inputs={"Negative particle diffusivity [m2.s-1]": ("D_n", 3.3e-14)},
            solver_tol=_TWO_PARAM_SOLVER_TOL,
        )
        a_n, a_c = var_n.sensitivities["all"], var_c.sensitivities["all"]
        assert a_n.shape == a_c.shape and a_n.shape[1] == 2
        assert np.any(var_n.sensitivities["D_n"] != 0)
        # "all" column 0 aligns with the named block (assembly-order check)...
        np.testing.assert_allclose(
            a_n[:, 0], var_n.sensitivities["D_n"], rtol=_PARITY_RTOL, atol=_PARITY_ATOL
        )
        # ...and each named block matches the oracle, not just itself.
        for k in ("D_n", "I"):
            np.testing.assert_allclose(
                var_n.sensitivities[k],
                var_c.sensitivities[k],
                rtol=_PARITY_RTOL,
                atol=_PARITY_ATOL,
            )
        np.testing.assert_allclose(a_n, a_c, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)

    def test_state_sensitivity_matches_finite_difference(self):
        # Finite-difference diffsol's OWN values: no cross-integrator trajectory
        # noise, so the sensitivity formula is validated without ultra-tight tols.
        params = pybamm.ParameterValues("Chen2020")
        params["Current function [A]"] = pybamm.InputParameter("I")
        t_eval = np.linspace(0, 100, 15)
        name = "Terminal voltage [V]"
        i0, h = 0.5, 1e-4

        def voltage(current):
            m = pybamm.lithium_ion.SPM()
            m.events = []
            m.convert_to_format = "rust"
            sol = pybamm.Simulation(
                m,
                parameter_values=params,
                solver=pybamm.DiffsolSolver(rtol=1e-8, atol=1e-8),
            ).solve(t_eval, inputs={"I": current})
            return sol[name].entries.ravel()

        m = pybamm.lithium_ion.SPM()
        m.events = []
        m.convert_to_format = "rust"
        sol = pybamm.Simulation(
            m,
            parameter_values=params,
            solver=pybamm.DiffsolSolver(rtol=1e-8, atol=1e-8),
        ).solve(t_eval, inputs={"I": i0}, calculate_sensitivities=["I"])
        analytic = sol[name].sensitivities["I"].ravel()
        fd = (voltage(i0 + h) - voltage(i0 - h)) / (2 * h)
        assert np.any(analytic != 0)
        np.testing.assert_allclose(analytic, fd, rtol=2e-3, atol=1e-6)

    def test_output_variable_sensitivities_match_casadi(self):
        # output_variables + calculate_sensitivities routes diffsol through the
        # native outputs-and-sensitivities request, oracled by CasADi-IDAKLU.
        params = pybamm.ParameterValues("Chen2020")
        params["Current function [A]"] = pybamm.InputParameter("I")
        inputs = {"I": 0.5}
        t_eval = np.linspace(0, 100, 15)
        name = "Terminal voltage [V]"

        m = pybamm.lithium_ion.SPM()
        m.events = []
        m.convert_to_format = "rust"
        solver = pybamm.DiffsolSolver(
            rtol=_SCALAR_SOLVER_TOL, atol=_SCALAR_SOLVER_TOL, output_variables=[name]
        )
        sol_n = pybamm.Simulation(m, parameter_values=params, solver=solver).solve(
            t_eval, inputs=inputs, calculate_sensitivities=["I"]
        )

        m_c = pybamm.lithium_ion.SPM()
        m_c.events = []
        m_c.convert_to_format = "casadi"
        sol_c = pybamm.Simulation(
            m_c,
            parameter_values=params,
            solver=pybamm.IDAKLUSolver(
                rtol=_SCALAR_SOLVER_TOL,
                atol=_SCALAR_SOLVER_TOL,
                output_variables=[name],
            ),
        ).solve(t_eval, inputs=inputs, calculate_sensitivities=["I"], t_interp=t_eval)

        np.testing.assert_allclose(
            sol_n[name].sensitivities["I"],
            sol_c[name].sensitivities["I"],
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
        )

    def test_output_variable_sensitivities_subset_of_inputs_match_casadi(self):
        # Two input parameters (I, D_n) but sensitivities requested for one (I): the
        # native output path must return just that block, not one per input.
        params = pybamm.ParameterValues("Chen2020")
        params["Current function [A]"] = pybamm.InputParameter("I")
        params["Negative particle diffusivity [m2.s-1]"] = pybamm.InputParameter("D_n")
        inputs = {"I": 0.5, "D_n": 3.3e-14}
        t_eval = np.linspace(0, 100, 15)
        name = "Terminal voltage [V]"

        m = pybamm.lithium_ion.SPM()
        m.events = []
        m.convert_to_format = "rust"
        solver = pybamm.DiffsolSolver(
            rtol=_SCALAR_SOLVER_TOL, atol=_SCALAR_SOLVER_TOL, output_variables=[name]
        )
        sol_n = pybamm.Simulation(m, parameter_values=params, solver=solver).solve(
            t_eval, inputs=inputs, calculate_sensitivities=["I"]
        )

        # Oracle is the IDAKLU STATE path: its output path mislabels columns when
        # calculate_sensitivities is a strict subset, and dV/dI is the same either way.
        m_c = pybamm.lithium_ion.SPM()
        m_c.events = []
        m_c.convert_to_format = "casadi"
        sol_c = pybamm.Simulation(
            m_c,
            parameter_values=params,
            solver=pybamm.IDAKLUSolver(
                rtol=_SCALAR_SOLVER_TOL, atol=_SCALAR_SOLVER_TOL
            ),
        ).solve(t_eval, inputs=inputs, calculate_sensitivities=["I"], t_interp=t_eval)

        assert set(sol_n[name].sensitivities) == {"I", "all"}
        np.testing.assert_allclose(
            sol_n[name].sensitivities["I"],
            sol_c[name].sensitivities["I"],
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
        )

    def test_state_sensitivities_subset_of_inputs_match_casadi(self):
        # With sensitivities requested for I alone, the state path must label the
        # single block "I" and match CasADi regardless of the surplus input.
        params = pybamm.ParameterValues("Chen2020")
        params["Current function [A]"] = pybamm.InputParameter("I")
        params["Negative particle diffusivity [m2.s-1]"] = pybamm.InputParameter("D_n")
        inputs = {"I": 0.5, "D_n": 3.3e-14}
        t_eval = np.linspace(0, 100, 15)
        name = "Terminal voltage [V]"

        m = pybamm.lithium_ion.SPM()
        m.events = []
        m.convert_to_format = "rust"
        sol_n = pybamm.Simulation(
            m,
            parameter_values=params,
            solver=pybamm.DiffsolSolver(
                rtol=_SCALAR_SOLVER_TOL, atol=_SCALAR_SOLVER_TOL
            ),
        ).solve(t_eval, inputs=inputs, calculate_sensitivities=["I"])

        m_c = pybamm.lithium_ion.SPM()
        m_c.events = []
        m_c.convert_to_format = "casadi"
        sol_c = pybamm.Simulation(
            m_c,
            parameter_values=params,
            solver=pybamm.IDAKLUSolver(
                rtol=_SCALAR_SOLVER_TOL, atol=_SCALAR_SOLVER_TOL
            ),
        ).solve(t_eval, inputs=inputs, calculate_sensitivities=["I"], t_interp=t_eval)

        assert set(sol_n[name].sensitivities) == {"I", "all"}
        np.testing.assert_allclose(
            sol_n[name].sensitivities["I"],
            sol_c[name].sensitivities["I"],
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
        )

    @staticmethod
    def _build_time_integral_model(fmt):
        # A genuine ExplicitTimeIntegral whose post_sum_node is the squaring. ``c``
        # is a state, so the integral discretises to StateVector(0:1).
        c = pybamm.Variable("c")
        c2 = pybamm.Variable("c2")
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        integral = pybamm.ExplicitTimeIntegral(c, 0) ** 2
        model = pybamm.BaseModel(name="time_integral_model")
        model.rhs = {c: b * -a * c, c2: -2 * c2}
        model.initial_conditions = {c: 1, c2: 1}
        model.variables["integral"] = integral
        model.variables["c"] = c
        model.convert_to_format = fmt
        pybamm.Discretisation().process_model(model)
        return model

    def test_time_integral_sensitivities_match_casadi(self):
        # Confirm the variable is a time-integral with a non-None post_sum_node.
        model_check = self._build_time_integral_model("casadi")
        var = model_check.get_processed_variable_or_event("integral")
        time_integral = pybamm.ProcessedVariableTimeIntegral.from_pybamm_var(var, 2)
        assert time_integral is not None
        assert time_integral.post_sum_node is not None

        times = np.linspace(0, 1, 15)
        inputs = {"a": 0.7, "b": 1.0}

        # The squaring post-sum amplifies BDF-vs-IDA trajectory differences, so pin
        # both solvers at the tight spatial tolerance.
        m_native = self._build_time_integral_model("rust")
        sol_n = pybamm.DiffsolSolver(
            rtol=_SPATIAL_SOLVER_TOL, atol=_SPATIAL_SOLVER_TOL
        ).solve(
            m_native,
            t_eval=[times[0], times[-1]],
            t_interp=times,
            inputs=inputs,
            calculate_sensitivities=["a", "b"],
        )
        assert isinstance(sol_n.observation, NativeObservation)

        m_casadi = self._build_time_integral_model("casadi")
        sol_c = pybamm.IDAKLUSolver(
            rtol=_SPATIAL_SOLVER_TOL, atol=_SPATIAL_SOLVER_TOL
        ).solve(
            m_casadi,
            t_eval=[times[0], times[-1]],
            t_interp=times,
            inputs=inputs,
            calculate_sensitivities=["a", "b"],
        )

        sens_n, sens_c = (
            sol_n["integral"].sensitivities,
            sol_c["integral"].sensitivities,
        )
        assert set(sens_n) == set(sens_c)
        for key in sens_c:
            assert sens_n[key].shape == sens_c[key].shape
            np.testing.assert_allclose(
                sens_n[key], sens_c[key], rtol=_PARITY_RTOL, atol=_PARITY_ATOL
            )

    def test_output_variables_time_integral_matches_the_full_state_solve(self):
        # The rows of a time-integral output carry its integrand; the postfix sum
        # runs in the shared outputs assembly, so diffsol reads the same value and
        # the same sensitivities as its own full-state solve.
        times = np.linspace(0, 1, 15)
        inputs = {"a": 0.7, "b": 1.0}
        solve_kwargs = {
            "t_eval": [times[0], times[-1]],
            "t_interp": times,
            "inputs": inputs,
            "calculate_sensitivities": ["a", "b"],
        }

        sol_out = pybamm.DiffsolSolver(
            rtol=_SPATIAL_SOLVER_TOL,
            atol=_SPATIAL_SOLVER_TOL,
            output_variables=["integral"],
        ).solve(self._build_time_integral_model("rust"), **solve_kwargs)
        sol_full = pybamm.DiffsolSolver(
            rtol=_SPATIAL_SOLVER_TOL, atol=_SPATIAL_SOLVER_TOL
        ).solve(self._build_time_integral_model("rust"), **solve_kwargs)

        np.testing.assert_allclose(
            np.asarray(sol_out["integral"].entries),
            np.asarray(sol_full["integral"].entries),
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
        )
        sens_out, sens_full = (
            sol_out["integral"].sensitivities,
            sol_full["integral"].sensitivities,
        )
        for name in inputs:
            np.testing.assert_allclose(
                sens_out[name],
                sens_full[name],
                rtol=_PARITY_RTOL,
                atol=_PARITY_ATOL,
                err_msg=f"sensitivity mismatch for param '{name}'",
            )

    @staticmethod
    def _build_two_integrals_model(fmt):
        # Two time-integral variables with the SAME n_inner (scalar) but DIFFERENT
        # post-sum operations (square vs cube), to catch post-sum cache aliasing.
        c = pybamm.Variable("c")
        c2 = pybamm.Variable("c2")
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        model = pybamm.BaseModel(name="two_integrals_model")
        model.rhs = {c: b * -a * c, c2: -2 * c2}
        model.initial_conditions = {c: 1, c2: 1}
        model.variables["sq"] = pybamm.ExplicitTimeIntegral(c, 0) ** 2
        model.variables["cube"] = pybamm.ExplicitTimeIntegral(c, 0) ** 3
        model.variables["c"] = c
        model.convert_to_format = fmt
        pybamm.Discretisation().process_model(model)
        return model

    def test_distinct_time_integrals_same_n_inner_dont_alias(self):
        # Both post-sum fns share n_inner=1; the cache must key them apart by
        # variable name, else "sq" and "cube" swap post-sum jacobians.
        times = np.linspace(0, 1, 15)
        inputs = {"a": 0.7, "b": 1.0}

        sol_n = pybamm.DiffsolSolver(
            rtol=_SPATIAL_SOLVER_TOL, atol=_SPATIAL_SOLVER_TOL
        ).solve(
            self._build_two_integrals_model("rust"),
            t_eval=[times[0], times[-1]],
            t_interp=times,
            inputs=inputs,
            calculate_sensitivities=["a", "b"],
        )
        sol_c = pybamm.IDAKLUSolver(
            rtol=_SPATIAL_SOLVER_TOL, atol=_SPATIAL_SOLVER_TOL
        ).solve(
            self._build_two_integrals_model("casadi"),
            t_eval=[times[0], times[-1]],
            t_interp=times,
            inputs=inputs,
            calculate_sensitivities=["a", "b"],
        )

        for var in ("sq", "cube"):
            sens_n, sens_c = sol_n[var].sensitivities, sol_c[var].sensitivities
            assert set(sens_n) == set(sens_c)
            for key in sens_c:
                np.testing.assert_allclose(
                    sens_n[key], sens_c[key], rtol=_PARITY_RTOL, atol=_PARITY_ATOL
                )

    def test_sensitivities_with_event_termination_match_casadi(self):
        params = pybamm.ParameterValues("Chen2020")
        params["Current function [A]"] = pybamm.InputParameter("I")
        inputs = {"I": 5.0}  # high current trips the lower voltage cutoff early
        t_eval = np.linspace(0, 3600, 50)
        name = "Terminal voltage [V]"

        m = pybamm.lithium_ion.SPM()  # events retained (no m.events = [])
        m.convert_to_format = "rust"
        # t_interp pins output to the t_eval grid so both solvers compare aligned
        # time points up to the event.
        sol_n = pybamm.Simulation(
            m,
            parameter_values=params,
            solver=pybamm.DiffsolSolver(
                rtol=_SCALAR_SOLVER_TOL, atol=_SCALAR_SOLVER_TOL
            ),
        ).solve(t_eval, inputs=inputs, calculate_sensitivities=["I"], t_interp=t_eval)
        assert sol_n.termination.startswith("event")

        m_c = pybamm.lithium_ion.SPM()
        m_c.convert_to_format = "casadi"
        sol_c = pybamm.Simulation(
            m_c,
            parameter_values=params,
            solver=pybamm.IDAKLUSolver(
                rtol=_SCALAR_SOLVER_TOL, atol=_SCALAR_SOLVER_TOL
            ),
            # t_interp forces IDAKLU to output exactly at t_eval so both solutions
            # share the same time grid for comparison.
        ).solve(t_eval, inputs=inputs, calculate_sensitivities=["I"], t_interp=t_eval)
        assert sol_c.termination.startswith("event")

        # t_interp pins both to one grid, so allow at most a single grid-point
        # difference in the event window and require it to be non-trivial.
        len_n = sol_n[name].sensitivities["I"].shape[0]
        len_c = sol_c[name].sensitivities["I"].shape[0]
        assert abs(len_n - len_c) <= 1, (
            f"solvers disagree on event window length: diffsol={len_n}, idaklu={len_c}"
        )
        n = min(len_n, len_c)
        assert n > 1
        np.testing.assert_allclose(
            sol_n[name].sensitivities["I"][:n],
            sol_c[name].sensitivities["I"][:n],
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
        )

    def test_output_variable_sensitivities_with_event_termination_match_casadi(self):
        # Exercises an outputs-and-sensitivities request under event truncation. The single
        # input is the sole requested sensitivity, so IDAKLU's output path oracles it.
        params = pybamm.ParameterValues("Chen2020")
        params["Current function [A]"] = pybamm.InputParameter("I")
        inputs = {"I": 5.0}
        t_eval = np.linspace(0, 3600, 50)
        name = "Terminal voltage [V]"

        m = pybamm.lithium_ion.SPM()  # events retained
        m.convert_to_format = "rust"
        sol_n = pybamm.Simulation(
            m,
            parameter_values=params,
            solver=pybamm.DiffsolSolver(
                rtol=_SCALAR_SOLVER_TOL,
                atol=_SCALAR_SOLVER_TOL,
                output_variables=[name],
            ),
        ).solve(t_eval, inputs=inputs, calculate_sensitivities=["I"], t_interp=t_eval)
        assert sol_n.termination.startswith("event")

        m_c = pybamm.lithium_ion.SPM()
        m_c.convert_to_format = "casadi"
        sol_c = pybamm.Simulation(
            m_c,
            parameter_values=params,
            solver=pybamm.IDAKLUSolver(
                rtol=_SCALAR_SOLVER_TOL,
                atol=_SCALAR_SOLVER_TOL,
                output_variables=[name],
            ),
        ).solve(t_eval, inputs=inputs, calculate_sensitivities=["I"], t_interp=t_eval)

        len_n = sol_n[name].sensitivities["I"].shape[0]
        len_c = sol_c[name].sensitivities["I"].shape[0]
        assert abs(len_n - len_c) <= 1, (
            f"solvers disagree on event window length: diffsol={len_n}, idaklu={len_c}"
        )
        n = min(len_n, len_c)
        assert n > 1
        np.testing.assert_allclose(
            sol_n[name].sensitivities["I"][:n],
            sol_c[name].sensitivities["I"][:n],
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
        )

    def test_output_variable_sensitivities_two_params_match_casadi(self):
        # Locking test: insertion order (I, D_n) differs from sorted
        # calculate_sensitivities order (D_n, I), exposing column mislabelling.
        params = pybamm.ParameterValues("Chen2020")
        params["Current function [A]"] = pybamm.InputParameter("I")
        params["Negative particle diffusivity [m2.s-1]"] = pybamm.InputParameter("D_n")
        # Insertion order: I first, D_n second — intentionally != sorted order.
        inputs = {"I": 0.5, "D_n": 3.3e-14}
        t_eval = np.linspace(0, 100, 15)
        name = "Terminal voltage [V]"

        m = pybamm.lithium_ion.SPM()
        m.events = []
        m.convert_to_format = "rust"
        solver = pybamm.DiffsolSolver(
            rtol=_TWO_PARAM_SOLVER_TOL,
            atol=_TWO_PARAM_SOLVER_TOL,
            output_variables=[name],
        )
        sol_n = pybamm.Simulation(m, parameter_values=params, solver=solver).solve(
            t_eval, inputs=inputs, calculate_sensitivities=["D_n", "I"]
        )

        m_c = pybamm.lithium_ion.SPM()
        m_c.events = []
        m_c.convert_to_format = "casadi"
        sol_c = pybamm.Simulation(
            m_c,
            parameter_values=params,
            solver=pybamm.IDAKLUSolver(
                rtol=_TWO_PARAM_SOLVER_TOL,
                atol=_TWO_PARAM_SOLVER_TOL,
                output_variables=[name],
            ),
        ).solve(
            t_eval, inputs=inputs, calculate_sensitivities=["D_n", "I"], t_interp=t_eval
        )

        for param_name in ("I", "D_n"):
            np.testing.assert_allclose(
                sol_n[name].sensitivities[param_name],
                sol_c[name].sensitivities[param_name],
                rtol=_PARITY_RTOL,
                atol=_PARITY_ATOL,
                err_msg=f"sensitivity mislabelled for param '{param_name}'",
            )

    def test_discrete_time_sum_guard(self):
        # A DiscreteTimeSum solved on times that do not match the discrete sum
        # times must raise when its sensitivities are accessed natively.
        data_times = np.linspace(0, 1, 10)
        ref = pybamm.BaseModel(name="ref")
        c_ref = pybamm.Variable("c")
        ref.rhs = {c_ref: -2 * c_ref}
        ref.initial_conditions = {c_ref: 1}
        ref.variables["c"] = c_ref
        pybamm.Discretisation().process_model(ref)
        data_values = (
            pybamm.IDAKLUSolver()
            .solve(ref, t_eval=[data_times[0], data_times[-1]], t_interp=data_times)[
                "c"
            ]
            .entries
        )
        data = pybamm.DiscreteTimeData(data_times, data_values, "test_data")

        c = pybamm.Variable("c")
        c2 = pybamm.Variable("c2")
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        model = pybamm.BaseModel(name="dts_model")
        model.rhs = {c: b * -a * c, c2: -2 * c2}
        model.initial_conditions = {c: 1, c2: 1}
        model.variables["data_comparison"] = pybamm.DiscreteTimeSum((c - data) ** 2)
        model.variables["c"] = c
        model.convert_to_format = "rust"
        pybamm.Discretisation().process_model(model)

        # Solve on a mismatched grid (t_interp != data_times) so the guard fires.
        sol = pybamm.DiffsolSolver().solve(
            model,
            t_eval=[0, 1],
            t_interp=np.linspace(0, 1, 5),
            inputs={"a": 0.7, "b": 1.0},
            calculate_sensitivities=["a", "b"],
        )
        assert isinstance(sol.observation, NativeObservation)
        with pytest.raises(pybamm.SolverError, match="discrete times"):
            _ = sol["data_comparison"].sensitivities

    def test_calculate_sensitivities_rejects_vector_width_input_parameter(self):
        # sens_param_indices seeds one scalar direction per named parameter; a
        # width>1 parameter must raise instead of silently under-seeding (idaklu guard).
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        b = pybamm.InputParameter("b", expected_size=2)
        model.rhs = {u: -(pybamm.Index(b, 0) + pybamm.Index(b, 1)) * u}
        model.initial_conditions = {u: 1}
        model.convert_to_format = "rust"
        pybamm.Discretisation().process_model(model)
        with pytest.raises(
            pybamm.SolverError, match=r"vector-width input parameters.*'b'"
        ):
            pybamm.DiffsolSolver().solve(
                model,
                t_eval=np.linspace(0, 1, 10),
                inputs={"b": np.array([0.2, 0.3])},
                calculate_sensitivities=["b"],
            )

    def _chen_spm_params(self, extra):
        params = pybamm.ParameterValues("Chen2020")
        params["Current function [A]"] = pybamm.InputParameter("I")
        for pybamm_name, input_name in extra.items():
            params[pybamm_name] = pybamm.InputParameter(input_name)
        return params

    def _solve(self, fmt, tol, params, inputs, t_eval, calc, t_interp=None):
        model = pybamm.lithium_ion.SPM()
        model.events = []
        model.convert_to_format = fmt
        solver = (
            pybamm.DiffsolSolver(rtol=tol, atol=tol)
            if fmt == "rust"
            else pybamm.IDAKLUSolver(rtol=tol, atol=tol)
        )
        kwargs = {"inputs": inputs, "calculate_sensitivities": calc}
        # Both backends must land on the same grid or the arrays are not
        # comparable; IDAKLU additionally needs t_interp to avoid its own knots.
        if t_interp is not None:
            kwargs["t_interp"] = t_interp
        elif fmt == "casadi":
            kwargs["t_interp"] = t_eval
        return pybamm.Simulation(model, parameter_values=params, solver=solver).solve(
            t_eval, **kwargs
        )

    def test_weakly_influential_parameter_sensitivity_is_accurate(self):
        # D_n has |p * dV/dp| ~ 1e-4; a state-sized atol floor swamps its column.
        # Assert against a converged reference: a same-tol oracle shares the defect.
        params = self._chen_spm_params(
            {"Negative particle diffusivity [m2.s-1]": "D_n"}
        )
        inputs = {"D_n": 3.3e-14, "I": 0.5}
        t_eval = np.linspace(0, 100, 15)
        calc = ["D_n", "I"]

        reference = self._solve("casadi", 1e-12, params, inputs, t_eval, calc)
        native = self._solve("rust", 1e-9, params, inputs, t_eval, calc)

        ref = np.asarray(reference["Terminal voltage [V]"].sensitivities["D_n"]).ravel()
        got = np.asarray(native["Terminal voltage [V]"].sensitivities["D_n"]).ravel()
        assert np.any(ref != 0)
        # Pre-fix this column lands at 2.1e-05; the converged value is ~1.6e-08.
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=0.0)

    def test_early_transient_sensitivity_is_accurate(self):
        # Error peaks near t=0.1s where |s| is a few percent of its peak; a uniform
        # grid samples this weakly, so use a log-spaced early grid.
        params = self._chen_spm_params(
            {"Positive electrode active material volume fraction": "eps_p"}
        )
        inputs = {"I": 5.0, "eps_p": 0.665}
        grid = np.unique(
            np.concatenate([np.array([0.0]), np.logspace(-3, np.log10(30.0), 25)])
        )
        span = np.array([0.0, 30.0])
        calc = ["I", "eps_p"]

        reference = self._solve(
            "casadi", 1e-11, params, inputs, span, calc, t_interp=grid
        )
        native = self._solve("rust", 1e-6, params, inputs, span, calc, t_interp=grid)

        for key in calc:
            ref = np.asarray(
                reference["Terminal voltage [V]"].sensitivities[key]
            ).ravel()
            got = np.asarray(native["Terminal voltage [V]"].sensitivities[key]).ravel()
            peak = np.max(np.abs(ref))
            assert peak > 0
            # Pre-fix (this span): I=4.99e-04, eps_p=1.04e-03 peak-normalised.
            # Not the design doc's 3000 s-horizon figures; those don't apply here.
            assert np.max(np.abs(got - ref)) / peak < 1e-6, (
                f"early-time {key} sensitivity error too large"
            )

    def test_parameter_entering_initial_conditions_is_seeded(self):
        # diffsol once held dy0/dp at zero, reading 5x low at t=0 and drifting
        # from there; both IDAKLU backends seed it from jacp_initial_conditions.
        name = "Maximum concentration in negative electrode [mol.m-3]"
        params = self._chen_spm_params({name: "c_n_max"})
        inputs = {"I": 0.5, "c_n_max": 33133.0}
        t_eval = np.linspace(0, 100, 15)
        calc = ["I", "c_n_max"]

        reference = self._solve("casadi", 1e-9, params, inputs, t_eval, calc)
        native = self._solve("rust", 1e-9, params, inputs, t_eval, calc)

        ref = np.asarray(
            reference["Terminal voltage [V]"].sensitivities["c_n_max"]
        ).ravel()
        got = np.asarray(
            native["Terminal voltage [V]"].sensitivities["c_n_max"]
        ).ravel()
        # Non-vacuous: the t=0 value is the one an unseeded dy0/dp gets wrong.
        assert abs(ref[0]) > 0
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=0.0)

    def test_dy0_dp_seed_does_not_smear_across_columns(self):
        # I never reaches y0, so its seed column is zero while c_n_max's is not;
        # a seed written to the wrong column would move this one too.
        name = "Maximum concentration in negative electrode [mol.m-3]"
        params = self._chen_spm_params({name: "c_n_max"})
        inputs = {"I": 0.5, "c_n_max": 33133.0}
        t_eval = np.linspace(0, 100, 15)
        calc = ["I", "c_n_max"]

        reference = self._solve("casadi", 1e-9, params, inputs, t_eval, calc)
        native = self._solve("rust", 1e-9, params, inputs, t_eval, calc)

        ref = np.asarray(reference["Terminal voltage [V]"].sensitivities["I"]).ravel()
        got = np.asarray(native["Terminal voltage [V]"].sensitivities["I"]).ravel()
        assert np.any(ref != 0)
        np.testing.assert_allclose(got, ref, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)

    @pytest.mark.parametrize("parameter_set", ["default", "Chen2020"])
    def test_dfn_sensitivities_solve(self, parameter_set):
        # DFN is a DAE (algebraic block ~100 states). Sensitivity error control
        # broke this entirely once; both parameter sets must stay solvable.
        model = pybamm.lithium_ion.DFN()
        params = (
            model.default_parameter_values.copy()
            if parameter_set == "default"
            else pybamm.ParameterValues("Chen2020")
        )
        sens = {
            "Current function [A]": "I",
            "Positive electrode active material volume fraction": "eps_p",
        }
        inputs = {}
        for pybamm_name, input_name in sens.items():
            inputs[input_name] = float(params[pybamm_name])
            params[pybamm_name] = pybamm.InputParameter(input_name)
        model.convert_to_format = "rust"

        grid = np.linspace(0.0, 600.0, 50)
        solution = pybamm.Simulation(
            model,
            parameter_values=params,
            solver=pybamm.DiffsolSolver(rtol=1e-6, atol=1e-6),
        ).solve(
            [float(grid[0]), float(grid[-1])],
            t_interp=grid,
            inputs=inputs,
            calculate_sensitivities=sorted(sens.values()),
        )

        assert solution.solver_statistics.sens_error_control_relaxed is False
        for key in sorted(sens.values()):
            block = np.asarray(solution["Voltage [V]"].sensitivities[key]).ravel()
            assert block.size > 1
            assert np.all(np.isfinite(block))
            assert np.any(block != 0)

    def test_state_sensitivities_subset_selects_the_extreme_scale_input(self):
        # Guards the subset mapping against applying I's parameter scale to the
        # D_n column: the included parameter is the extreme-magnitude one.
        params = self._chen_spm_params(
            {"Negative particle diffusivity [m2.s-1]": "D_n"}
        )
        inputs = {"D_n": 3.3e-14, "I": 0.5}
        t_eval = np.linspace(0, 100, 15)

        native = self._solve(
            "rust", _TWO_PARAM_SOLVER_TOL, params, inputs, t_eval, ["D_n"]
        )
        casadi = self._solve(
            "casadi", _TWO_PARAM_SOLVER_TOL, params, inputs, t_eval, ["D_n"]
        )

        sens_n = native["Terminal voltage [V]"].sensitivities
        sens_c = casadi["Terminal voltage [V]"].sensitivities
        assert "D_n" in sens_n
        assert "I" not in sens_n
        assert np.any(np.asarray(sens_n["D_n"]) != 0)
        np.testing.assert_allclose(
            sens_n["D_n"], sens_c["D_n"], rtol=_PARITY_RTOL, atol=_PARITY_ATOL
        )


class TestDiffsolSensAtolFactor:
    def test_default_sens_atol_factor(self):
        solver = pybamm.DiffsolSolver()
        assert solver._sens_atol_factor == pytest.approx(1e-3)

    def test_sens_atol_factor_is_configurable(self):
        solver = pybamm.DiffsolSolver(sens_atol_factor=1e-2)
        assert solver._sens_atol_factor == pytest.approx(1e-2)

    @pytest.mark.parametrize(
        "bad", [0, -1.0, float("nan"), float("inf"), "not-a-number", None]
    )
    def test_sens_atol_factor_rejects_invalid(self, bad):
        with pytest.raises(pybamm.SolverError, match=r"sens_atol_factor"):
            pybamm.DiffsolSolver(sens_atol_factor=bad)

    def test_sens_atol_factor_reaches_rust(self):
        # Pins the factor actually reaching the Rust binding (not just stored
        # on self): a tighter floor forces more BDF steps under error control.
        params = pybamm.ParameterValues("Chen2020")
        params["Current function [A]"] = pybamm.InputParameter("I")
        inputs = {"I": 0.5}
        t_eval = np.linspace(0, 100, 15)

        def solve(sens_atol_factor):
            m = pybamm.lithium_ion.SPM()
            m.events = []
            m.convert_to_format = "rust"
            return pybamm.Simulation(
                m,
                parameter_values=params,
                solver=pybamm.DiffsolSolver(
                    rtol=1e-9, atol=1e-9, sens_atol_factor=sens_atol_factor
                ),
            ).solve(t_eval, inputs=inputs, calculate_sensitivities=["I"])

        sol_default = solve(1e-3)
        sol_relaxed = solve(1.0)
        assert (
            sol_default.solver_statistics.number_of_steps
            > sol_relaxed.solver_statistics.number_of_steps
        )


class TestFlattenY0Sens:
    """``dy0/dp`` normalisation, the seed threading that fixed diffsol's
    unseeded initial-condition sensitivities.

    Producers of ``model.y0S_list`` disagree on shape: ``jacp`` hands over a
    list of ``(n, 1)`` columns, a step restart a tuple of bare ``(n,)`` ones.
    Both must reach the Rust solver as one column-major block.
    """

    @staticmethod
    def _flatten(*args, **kwargs):
        return pybamm.solvers.diffsol_solver._flatten_y0_sens(*args, **kwargs)

    def test_a_list_of_column_vectors_is_column_major(self):
        blocks = [np.array([[1.0], [2.0], [3.0]]), np.array([[4.0], [5.0], [6.0]])]
        np.testing.assert_array_equal(
            self._flatten(blocks, 3, 2), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )

    def test_a_tuple_of_bare_columns_is_the_same_block(self):
        # The shape a step restart produces, via full_sens[:, i].
        blocks = (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        np.testing.assert_array_equal(
            self._flatten(blocks, 3, 2), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )

    def test_a_single_matrix_is_the_same_block(self):
        matrix = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        np.testing.assert_array_equal(
            self._flatten(matrix, 3, 2), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )

    def test_a_casadi_column_is_densified(self):
        casadi = pybamm.import_optional_dependency("casadi")
        blocks = [casadi.DM([1.0, 2.0]), casadi.DM([3.0, 4.0])]
        np.testing.assert_array_equal(self._flatten(blocks, 2, 2), [1.0, 2.0, 3.0, 4.0])

    def test_an_empty_seed_is_the_all_zero_case(self):
        assert self._flatten([], 3, 2).size == 0

    def test_a_wrong_shape_is_a_solver_error(self):
        # Silently reshaping would attach one parameter's seed to another.
        with pytest.raises(pybamm.SolverError, match=r"\(3, 2\) was expected"):
            self._flatten([np.array([1.0, 2.0, 3.0])], 3, 2)

    def test_a_wrong_state_count_is_a_solver_error(self):
        with pytest.raises(pybamm.SolverError, match=r"\(3, 1\) was expected"):
            self._flatten([np.array([1.0, 2.0])], 3, 1)


class TestDiffsolSensitivitiesAcrossExperimentSteps:
    """A step restart re-seeds ``dy0/dp`` from the previous segment's terminal
    sensitivities, in a different shape from the ``jacp`` path."""

    @staticmethod
    def _solve(convert_to_format):
        parameter_values = pybamm.ParameterValues("Chen2020")
        name = "Negative particle diffusivity [m2.s-1]"
        parameter_values[name] = pybamm.InputParameter("D_n")
        model = pybamm.lithium_ion.SPM()
        model.convert_to_format = convert_to_format
        solver = (
            pybamm.DiffsolSolver(rtol=1e-8, atol=1e-8)
            if convert_to_format == "rust"
            else pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
        )
        return pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            experiment=pybamm.Experiment(
                ["Discharge at 1C for 200 seconds", "Rest for 100 seconds"],
                period="20 seconds",
            ),
            solver=solver,
        ).solve(inputs={"D_n": 3.3e-14}, calculate_sensitivities=["D_n"])

    def test_stepped_sensitivities_match_casadi(self):
        native = self._solve("rust")
        reference = self._solve("casadi")
        assert len(native.all_ts) == 2

        got = np.asarray(native["Voltage [V]"].sensitivities["D_n"]).ravel()
        ref = np.asarray(reference["Voltage [V]"].sensitivities["D_n"]).ravel()
        assert np.any(ref != 0)
        assert got.shape == ref.shape
        # Scaled by |p|: dV/dD_n is ~1e10, so a bare rtol reads as noise.
        scale = np.abs(ref).max()
        np.testing.assert_allclose(got / scale, ref / scale, rtol=0.0, atol=1e-4)
