import logging
import shutil

import numpy as np
import pytest

import pybamm
from pybamm.codegen.compilation import _CACHE


class TestIDAKLUSolver:
    def test_on_spme(self):
        model = pybamm.lithium_ion.SPMe()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        t_eval = np.linspace(0, 3600, 100)
        solution = pybamm.IDAKLUSolver().solve(model, t_eval)
        np.testing.assert_array_less(1, solution.t.size)

    def test_debug_logs_during_solve(self, caplog):
        model = pybamm.lithium_ion.SPM()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # Length 3 so solver hits an interior breakpoint and logs "reinitializing"
        t_eval = np.array([0.0, 0.5, 1.0])
        with caplog.at_level(logging.DEBUG, logger="pybamm.logger"):
            pybamm.IDAKLUSolver().solve(model, t_eval)

        requirements = [
            "Integrating from",
            "t =",
            "reinitializing",
            "Integration complete",
        ]
        for requirement in requirements:
            assert requirement in caplog.text, (
                f"Requirement {requirement} not found in log"
            )

    def test_on_spme_sensitivities(self):
        param_name = "Current function [A]"
        param_value = 0.15652
        param = pybamm.ParameterValues("Marquis2019")
        model = pybamm.lithium_ion.SPMe()
        geometry = model.default_geometry
        param.update({param_name: "[input]"})
        inputs = {param_name: param_value}
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        t_interp = np.linspace(0, 3500, 100)
        t_eval = [t_interp[0], t_interp[-1]]
        solver = pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-10)
        solution = solver.solve(
            model,
            t_eval,
            inputs=inputs,
            calculate_sensitivities=True,
            t_interp=t_interp,
        )
        np.testing.assert_array_less(1, solution.t.size)

        # evaluate the sensitivities using idas
        dyda_ida = solution.sensitivities[param_name]

        # evaluate the sensitivities using finite difference
        h = 1e-5
        sol_plus = solver.solve(
            model, t_eval, inputs={param_name: param_value + 0.5 * h}, t_interp=t_interp
        )
        sol_neg = solver.solve(
            model, t_eval, inputs={param_name: param_value - 0.5 * h}, t_interp=t_interp
        )
        dyda_fd = (sol_plus.y - sol_neg.y) / h
        dyda_fd = dyda_fd.transpose().reshape(-1, 1)

        np.testing.assert_allclose(
            dyda_ida,
            dyda_fd,
            rtol=1e-2,
            atol=1e-3,
        )

    def test_changing_grid(self):
        model = pybamm.lithium_ion.SPM()

        # load parameter values and geometry
        geometry = model.default_geometry
        param = model.default_parameter_values

        # Process parameters
        param.process_model(model)
        param.process_geometry(geometry)

        # Calculate time for each solver and each number of grid points
        t_eval = np.linspace(0, 3600, 100)
        for npts in [100, 200]:
            # discretise
            var_pts = {
                spatial_var: npts for spatial_var in ["x_n", "x_s", "x_p", "r_n", "r_p"]
            }
            mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            model_disc = disc.process_model(model, inplace=False)
            solver = pybamm.IDAKLUSolver()

            # solve
            solver.solve(model_disc, t_eval)

    def test_interpolation(self):
        model = pybamm.BaseModel()
        u1 = pybamm.Variable("u1")
        u2 = pybamm.Variable("u2")
        u3 = pybamm.Variable("u3")
        v = pybamm.Variable("v")
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b", expected_size=2)
        model.rhs = {u1: a * v, u2: pybamm.Index(b, 0), u3: pybamm.Index(b, 1)}
        model.algebraic = {v: 1 - v}
        model.initial_conditions = {u1: 0, u2: 0, u3: 0, v: 1}

        disc = pybamm.Discretisation()
        model_disc = disc.process_model(model, inplace=False)

        a_value = 0.1
        b_value = np.array([[0.2], [0.3]])
        inputs = {"a": a_value, "b": b_value}

        # Calculate time for each solver and each number of grid points
        t0 = 0
        tf = 3600
        t_eval_dense = np.linspace(t0, tf, 1000)
        t_eval_sparse = [t0, tf]

        t_interp_dense = np.linspace(t0, tf, 800)
        t_interp_sparse = [t0, tf]
        solver = pybamm.IDAKLUSolver()

        # solve
        # 1. dense t_eval + adaptive time stepping
        sol1 = solver.solve(model_disc, t_eval_dense, inputs=inputs)
        np.testing.assert_array_less(len(t_eval_dense), len(sol1.t))

        # 2. sparse t_eval + adaptive time stepping
        sol2 = solver.solve(model_disc, t_eval_sparse, inputs=inputs)
        np.testing.assert_array_less(len(sol2.t), len(sol1.t))

        # 3. dense t_eval + dense t_interp
        sol3 = solver.solve(
            model_disc, t_eval_dense, t_interp=t_interp_dense, inputs=inputs
        )
        t_combined = np.concatenate((sol3.t, t_interp_dense))
        t_combined = np.unique(t_combined)
        t_combined.sort()
        np.testing.assert_allclose(sol3.t, t_combined)

        # 4. sparse t_eval + sparse t_interp
        sol4 = solver.solve(
            model_disc, t_eval_sparse, t_interp=t_interp_sparse, inputs=inputs
        )
        np.testing.assert_allclose(sol4.t, np.array([t0, tf]))

        sols = [sol1, sol2, sol3, sol4]
        for sol in sols:
            # test that y[0] = to true solution
            true_solution = a_value * sol.t
            np.testing.assert_allclose(sol.y[0], true_solution)

            # test that y[1:3] = to true solution
            true_solution = b_value * sol.t
            np.testing.assert_allclose(sol.y[1:3], true_solution)

    def test_with_experiments(self):
        summary_vars = []
        sols = []
        for out_vars in [True, False]:
            model = pybamm.lithium_ion.SPM()

            if out_vars:
                output_variables = [
                    "Discharge capacity [A.h]",  # 0D variables
                    "Time [s]",
                    "Current [A]",
                    "Voltage [V]",
                    "Pressure [Pa]",  # 1D variable
                    "Positive particle effective diffusivity [m2.s-1]",  # 2D variable
                ]
            else:
                output_variables = None

            solver = pybamm.IDAKLUSolver(output_variables=output_variables)

            experiment = pybamm.Experiment(
                [
                    (
                        "Charge at 1C until 4.2 V",
                        "Hold at 4.2 V until C/50",
                        "Rest for 1 hour",
                    )
                ]
            )

            sim = pybamm.Simulation(
                model,
                experiment=experiment,
                solver=solver,
            )

            sol = sim.solve()
            sols.append(sol)
            summary_vars.append(sol.summary_variables)

        # check computed variables are propagated successfully
        np.testing.assert_array_equal(
            sols[0]["Pressure [Pa]"].data, sols[1]["Pressure [Pa]"].data
        )
        np.testing.assert_allclose(
            sols[0]["Voltage [V]"].data, sols[1]["Voltage [V]"].data
        )

        # check summary variables are the same if output variables are specified
        for var in model.summary_variables:
            assert summary_vars[0][var] == summary_vars[1][var]

        # check variables are accessible for each cycle
        np.testing.assert_allclose(
            sols[0].cycles[-1]["Current [A]"].data,
            sols[1].cycles[-1]["Current [A]"].data,
        )

    def test_multiple_initial_conditions_against_independent_solves(self):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        u0 = pybamm.InputParameter("u0")
        v0 = pybamm.InputParameter("v0")
        model.rhs = {u: -u, v: -2 * v}
        model.initial_conditions = {u: u0, v: v0}
        model.variables = {"u": u, "v": v}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        t_eval = np.array([0, 1])
        solver = pybamm.IDAKLUSolver()

        inputs = [{"u0": 3, "v0": 4}, {"u0": 5, "v0": 6}]

        multi_sols = solver.solve(
            model,
            t_eval,
            inputs=inputs,
        )
        assert isinstance(multi_sols, list) and len(multi_sols) == 2
        np.testing.assert_equal([sol["u"](0) for sol in multi_sols], [3, 5])

        indep_sols = []
        for ic in inputs:
            sol_indep = solver.solve(model, t_eval, inputs=ic)
            indep_sols.append(sol_indep)

        for idx in (0, 1):
            sol_vec = multi_sols[idx]
            sol_ind = indep_sols[idx]

            np.testing.assert_allclose(sol_vec.t, sol_ind.t)
            np.testing.assert_allclose(sol_vec.y, sol_ind.y)

    def test_outvars_with_experiments_multi_simulation(self):
        model = pybamm.lithium_ion.SPM()

        experiment = pybamm.Experiment(
            [
                "Discharge at C/2 for 10 minutes",
                "Rest for 10 minutes",
            ]
        )

        solver = pybamm.IDAKLUSolver(
            output_variables=[
                "Discharge capacity [A.h]",  # 0D variables
                "Time [s]",
                "Current [A]",
                "Voltage [V]",
                "Pressure [Pa]",  # 1D variable
                "Positive particle effective diffusivity [m2.s-1]",  # 2D variable
            ]
        )

        sim = pybamm.Simulation(model, experiment=experiment, solver=solver)
        solution = sim.solve()

        sim2 = pybamm.Simulation(model, experiment=experiment, solver=solver)

        new_sol1 = sim2.solve(starting_solution=solution.copy())
        new_sol2 = sim2.solve(starting_solution=solution.copy().last_state)

        n_initial = len(solution.t)
        np.testing.assert_array_equal(
            new_sol1["Voltage [V]"].entries[n_initial - 1 :],
            new_sol2["Voltage [V]"].entries,
        )


@pytest.mark.skipif(
    not (shutil.which("gcc") or shutil.which("cc")),
    reason="No C compiler available for AOT compilation",
)
class TestIDAKLUEventTimeSensitivities:
    """Sensitivities across moving event times against central differences."""

    var_pts = {"x_n": 4, "x_s": 3, "x_p": 4, "r_n": 8, "r_p": 8}

    @staticmethod
    def _central_difference(run, inputs, name, read, relative_step=1e-5):
        step = relative_step * inputs[name]
        plus, minus = (
            read(run({**inputs, name: inputs[name] + sign * step}, False))
            for sign in (1, -1)
        )
        return (plus - minus) / (2 * step)

    @pytest.mark.parametrize(
        "output_variables", [None, ["Discharge capacity [A.h]", "Voltage [V]"]]
    )
    def test_spm_discharge_to_cutoff(self, output_variables):
        names = ["Current function [A]", "Contact resistance [Ohm]"]
        inputs = {"Current function [A]": 5.0, "Contact resistance [Ohm]": 0.01}

        def run(inputs, sensitivities, output_variables=None):
            parameter_values = pybamm.ParameterValues("Chen2020")
            parameter_values.update({name: "[input]" for name in names})
            simulation = pybamm.Simulation(
                pybamm.lithium_ion.SPM({"contact resistance": "true"}),
                parameter_values=parameter_values,
                var_pts=self.var_pts,
                solver=pybamm.IDAKLUSolver(
                    rtol=1e-9, atol=1e-11, output_variables=output_variables or []
                ),
            )
            return simulation.solve(
                [0, 3 * 3600], inputs=inputs, calculate_sensitivities=sensitivities
            )

        solution = run(inputs, names, output_variables)
        assert solution.termination == "event: Minimum voltage [V]"
        last_state = solution.last_state
        for name in names:
            np.testing.assert_allclose(
                solution.t_event_sensitivities[name],
                self._central_difference(run, inputs, name, lambda s: s.t[-1:]),
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                last_state["Discharge capacity [A.h]"].sensitivities[name],
                self._central_difference(
                    run,
                    inputs,
                    name,
                    lambda s: s["Discharge capacity [A.h]"].entries[-1:],
                ),
                rtol=1e-5,
            )
            # The voltage at the cut-off is the cut-off, whatever the inputs
            np.testing.assert_allclose(
                last_state["Voltage [V]"].sensitivities[name], 0.0, atol=1e-8
            )

    @pytest.mark.parametrize("mode", ["legacy", "unified"])
    @pytest.mark.parametrize(
        "output_variables",
        [None, ["Voltage [V]", "Discharge capacity [A.h]"]],
    )
    def test_experiment_after_event_matches_finite_differences(
        self, mode, output_variables
    ):
        names = [
            "Negative particle diffusivity [m2.s-1]",
            "Positive electrode active material volume fraction",
        ]
        inputs = {names[0]: 3.3e-14, names[1]: 0.665}
        experiment = pybamm.Experiment(
            [
                "Discharge at 1C until 3.5 V",
                "Rest for 10 minutes",
                "Charge at C/2 for 10 minutes",
            ]
        )

        def run(inputs, sensitivities, output_variables=None):
            parameter_values = pybamm.ParameterValues("Chen2020")
            parameter_values.update({name: "[input]" for name in names})
            simulation = pybamm.Simulation(
                pybamm.lithium_ion.SPM(),
                parameter_values=parameter_values,
                var_pts=self.var_pts,
                solver=pybamm.IDAKLUSolver(
                    rtol=1e-10, atol=1e-12, output_variables=output_variables or []
                ),
                experiment=experiment,
                experiment_model_mode=mode,
            )
            return simulation.solve(
                inputs=inputs, calculate_sensitivities=sensitivities
            )

        solution = run(inputs, names, output_variables)
        steps = [step for cycle in solution.cycles for step in cycle.steps]
        assert len(steps) == 3
        assert "[experiment]" in steps[0].termination
        # Solution times away from the step transitions, which move with the inputs
        transitions = np.array([step.t[-1] for step in steps[:-1]])
        t = solution.t[
            np.min(np.abs(solution.t[:, np.newaxis] - transitions), axis=1) > 10
        ]
        t = t[t < solution.t[-1] - 10]
        t = np.concatenate([t[t < transitions[0]][::10], t[t > transitions[0]]])
        assert np.sum(t > transitions[0]) > 5
        for name in names:
            np.testing.assert_allclose(
                steps[0].t_event_sensitivities[name],
                self._central_difference(
                    run, inputs, name, lambda s: s.cycles[0].steps[0].t[-1:]
                ),
                rtol=1e-5,
            )
            for variable in ["Voltage [V]", "Discharge capacity [A.h]"]:
                sensitivity = np.interp(
                    t,
                    solution.t,
                    np.asarray(solution[variable].sensitivities[name]).ravel(),
                )
                finite_difference = self._central_difference(
                    run, inputs, name, lambda s, variable=variable: s[variable](t)
                )
                np.testing.assert_allclose(
                    sensitivity,
                    finite_difference,
                    rtol=1e-4,
                    atol=1e-4 * np.max(np.abs(finite_difference)),
                )

    def test_drive_cycle_after_event_matches_finite_differences(self):
        name = "Negative particle diffusivity [m2.s-1]"
        inputs = {name: 3.3e-14}
        time = np.arange(0, 601, 10.0)
        experiment = pybamm.Experiment(
            [
                "Discharge at 1C until 3.6 V",
                pybamm.step.current(
                    np.column_stack([time, 2.0 + 1.5 * np.sin(time / 40)]),
                    duration=600,
                ),
            ]
        )

        def run(inputs, sensitivities):
            parameter_values = pybamm.ParameterValues("Chen2020")
            parameter_values.update({name: "[input]"})
            simulation = pybamm.Simulation(
                pybamm.lithium_ion.SPM(),
                parameter_values=parameter_values,
                var_pts=self.var_pts,
                solver=pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-12),
                experiment=experiment,
            )
            return simulation.solve(
                inputs=inputs, calculate_sensitivities=sensitivities
            )

        solution = run(inputs, [name])
        t_start = solution.cycles[0].steps[0].t[-1]
        # Solution times between the drive cycle's knots, where its current has kinks
        offset = (solution.t - t_start) % 10
        t = solution.t[(solution.t > t_start + 5) & (offset > 2) & (offset < 8)]
        t = t[t < solution.t[-1] - 5]
        for variable in ["Voltage [V]", "Current [A]", "Discharge capacity [A.h]"]:
            sensitivity = np.interp(
                t,
                solution.t,
                np.asarray(solution[variable].sensitivities[name]).ravel(),
            )
            finite_difference = self._central_difference(
                run, inputs, name, lambda s, variable=variable: s[variable](t)
            )
            np.testing.assert_allclose(
                sensitivity,
                finite_difference,
                rtol=1e-4,
                atol=1e-4 * np.max(np.abs(finite_difference)),
            )


class TestIDAKLUSolverAOTCompilation:
    @staticmethod
    def _build_model():
        model = pybamm.lithium_ion.SPMe()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        return model

    def test_spme_solve_and_observe_voltage_matches_vm(self):
        t_eval = np.linspace(0, 3600, 100)

        model_int = self._build_model()
        sol_int = pybamm.IDAKLUSolver().solve(model_int, t_eval)

        model_aot = self._build_model()
        sol_aot = pybamm.IDAKLUSolver(options={"compile": True}).solve(
            model_aot, t_eval
        )

        np.testing.assert_array_less(1, sol_aot.t.size)
        t = np.union1d(sol_int.t, sol_aot.t)
        t = t[(t >= sol_aot.t[0]) & (t <= sol_aot.t[-1])]

        v_int = sol_int["Voltage [V]"](t)
        v_aot = sol_aot["Voltage [V]"](t)
        np.testing.assert_allclose(v_aot, v_int, rtol=1e-5, atol=1e-5)

        assert sol_aot.options["compile"] is True
        assert sol_int.options["compile"] is False

    def test_spme_observe_uses_aot_external(self):
        snapshot = dict(_CACHE)
        _CACHE.clear()
        try:
            model = self._build_model()
            t_eval = np.linspace(0, 3600, 50)
            sol = pybamm.IDAKLUSolver(options={"compile": True}).solve(model, t_eval)

            v = sol["Voltage [V]"](t_eval)
            assert v.shape == t_eval.shape

            assert len(_CACHE) >= 2
            assert all(
                fn.class_name() == "External"
                for fn_list in _CACHE.values()
                for fn in fn_list
            )
        finally:
            _CACHE.clear()
            _CACHE.update(snapshot)

    def test_invalid_compile_raises(self):
        with pytest.raises(pybamm.SolverError, match="compile must be a bool"):
            pybamm.IDAKLUSolver(options={"compile": "garbage"})
