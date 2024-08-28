import pytest
import pybamm
import numpy as np


@pytest.mark.skipif(not pybamm.have_idaklu(), reason="idaklu solver is not installed")
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
        np.testing.assert_array_almost_equal(sol3.t, t_combined)

        # 4. sparse t_eval + sparse t_interp
        sol4 = solver.solve(
            model_disc, t_eval_sparse, t_interp=t_interp_sparse, inputs=inputs
        )
        np.testing.assert_array_almost_equal(sol4.t, np.array([t0, tf]))

        sols = [sol1, sol2, sol3, sol4]
        for sol in sols:
            # test that y[0] = to true solution
            true_solution = a_value * sol.t
            np.testing.assert_array_almost_equal(sol.y[0], true_solution)

            # test that y[1:3] = to true solution
            true_solution = b_value * sol.t
            np.testing.assert_array_almost_equal(sol.y[1:3], true_solution)
