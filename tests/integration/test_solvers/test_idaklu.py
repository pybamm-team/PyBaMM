import pybamm
import numpy as np
import sys
import unittest


@unittest.skipIf(not pybamm.have_idaklu(), "idaklu solver is not installed")
class TestIDAKLUSolver(unittest.TestCase):
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
        param_name = "Negative electrode conductivity [S.m-1]"
        neg_electrode_cond = 100.0
        model = pybamm.lithium_ion.SPMe()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.update({param_name: "[input]"})
        inputs = {param_name: neg_electrode_cond}
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        t_eval = np.linspace(0, 3600, 100)
        solver = pybamm.IDAKLUSolver()
        solution = solver.solve(
            model, t_eval,
            inputs=inputs,
            calculate_sensitivities=True,
        )
        np.testing.assert_array_less(1, solution.t.size)

        # evaluate the sensitivities using idas
        dyda_ida = solution.sensitivities[param_name]

        # evaluate the sensitivities using finite difference
        h = 1e-6
        sol_plus = solver.solve(
            model, t_eval,
            inputs={param_name: neg_electrode_cond + 0.5 * h}
        )
        sol_neg = solver.solve(
            model, t_eval,
            inputs={param_name: neg_electrode_cond - 0.5 * h}
        )
        dyda_fd = (sol_plus.y - sol_neg.y) / h

        np.testing.assert_array_almost_equal(
            dyda_ida, dyda_fd
        )

    def test_set_tol_by_variable(self):
        model = pybamm.lithium_ion.SPMe()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        t_eval = np.linspace(0, 3600, 100)
        solver = pybamm.IDAKLUSolver()

        variable_tols = {"Porosity times concentration": 1e-3}
        solver.set_atol_by_variable(variable_tols, model)

        solver.solve(model, t_eval)

    def test_changing_grid(self):
        model = pybamm.lithium_ion.SPM()
        solver = pybamm.IDAKLUSolver()

        # load parameter values and geometry
        geometry = model.default_geometry
        param = model.default_parameter_values

        # Process parameters
        param.process_model(model)
        param.process_geometry(geometry)

        # Calculate time for each solver and each number of grid points
        var = pybamm.standard_spatial_vars
        t_eval = np.linspace(0, 3600, 100)
        for npts in [100, 200]:
            # discretise
            var_pts = {
                spatial_var: npts
                for spatial_var in [var.x_n, var.x_s, var.x_p, var.r_n, var.r_p]
            }
            mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            model_disc = disc.process_model(model, inplace=False)

            # solve
            solver.solve(model_disc, t_eval)


if __name__ == "__main__":
    print("Add -v for more debug output")

    pybamm.set_logging_level('INFO')
    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
