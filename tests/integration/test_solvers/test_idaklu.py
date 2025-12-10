import numpy as np
import pytest

import pybamm


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

        # check computed variables are propegated sucessfully
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

    @pytest.mark.parametrize(
        "model_cls, make_ics",
        [
            (pybamm.lithium_ion.SPM, lambda y0: [y0, 2 * y0]),
            (
                pybamm.lithium_ion.DFN,
                lambda y0: [y0, y0 * (1 + 0.01 * np.ones_like(y0))],
            ),
        ],
    )
    def test_multiple_initial_conditions_against_independent_solves(
        self, model_cls, make_ics
    ):
        model = model_cls()
        geom = model.default_geometry
        pv = model.default_parameter_values
        pv.process_model(model)
        pv.process_geometry(geom)
        mesh = pybamm.Mesh(geom, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        t_eval = np.array([0, 1])
        solver = pybamm.IDAKLUSolver()

        base_sol = solver.solve(model, t_eval)
        y0_base = base_sol.y[:, 0]

        ics = make_ics(y0_base)
        inputs = [{}] * len(ics)

        multi_sols = solver.solve(
            model,
            t_eval,
            inputs=inputs,
            initial_conditions=ics,
        )
        assert isinstance(multi_sols, list) and len(multi_sols) == 2

        indep_sols = []
        for ic in ics:
            sol_indep = solver.solve(
                model, t_eval, inputs=[{}], initial_conditions=[ic]
            )
            if isinstance(sol_indep, list):
                sol_indep = sol_indep[0]
            indep_sols.append(sol_indep)

        if model_cls is pybamm.lithium_ion.SPM:
            rtol, atol = 1e-8, 1e-10
        else:
            rtol, atol = 1e-6, 1e-8

        for idx in (0, 1):
            sol_vec = multi_sols[idx]
            sol_ind = indep_sols[idx]

            np.testing.assert_allclose(sol_vec.t, sol_ind.t, rtol=1e-12, atol=0)
            np.testing.assert_allclose(sol_vec.y, sol_ind.y, rtol=rtol, atol=atol)

            if model_cls is pybamm.lithium_ion.SPM:
                np.testing.assert_allclose(
                    sol_vec.y[:, 0], ics[idx], rtol=1e-8, atol=1e-10
                )

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

        np.testing.assert_array_equal(
            new_sol1["Voltage [V]"].entries[63:], new_sol2["Voltage [V]"].entries
        )
