import pybamm
import numpy as np
import sys
from tests import TestCase
import unittest


@unittest.skipIf(not pybamm.have_idaklu(), "idaklu solver is not installed")
class TestIDAKLUSolver(TestCase):
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
        t_eval = np.linspace(0, 3500, 100)
        solver = pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-10)
        solution = solver.solve(
            model,
            t_eval,
            inputs=inputs,
            calculate_sensitivities=True,
        )
        np.testing.assert_array_less(1, solution.t.size)

        # evaluate the sensitivities using idas
        dyda_ida = solution.sensitivities[param_name]

        # evaluate the sensitivities using finite difference
        h = 1e-5
        sol_plus = solver.solve(
            model, t_eval, inputs={param_name: param_value + 0.5 * h}
        )
        sol_neg = solver.solve(
            model, t_eval, inputs={param_name: param_value - 0.5 * h}
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

    def test_with_output_variables(self):
        # Construct a model and solve for all vairables, then test
        # the 'output_variables' option for each variable in turn, confirming
        # equivalence

        # construct model
        model = pybamm.lithium_ion.DFN()
        geometry = model.default_geometry
        param = model.default_parameter_values
        input_parameters = {}
        param.update({key: "[input]" for key in input_parameters})
        param.process_model(model)
        param.process_geometry(geometry)
        var_pts = {"x_n": 50, "x_s": 50, "x_p": 50, "r_n": 5, "r_p": 5}
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        t_eval = np.linspace(0, 3600, 100)

        options = {
            'linear_solver': 'SUNLinSol_KLU',
            'jacobian': 'sparse',
            'num_threads': 4,
        }

        # Select all output_variables (NB: This can be very slow to run all solves)
        # output_variables = [m for m, (k, v) in
        #                     zip(model.variable_names(), model.variables.items())
        #                     if not isinstance(v, pybamm.ExplicitTimeIntegral)]

        # Use a selection of variables (of different types)
        output_variables = [
            "Voltage [V]",
            "Time [min]",
            "Current [A]",
            "r_n [m]",
            "x [m]",
            "Gradient of negative electrolyte potential [V.m-1]",
            "Negative particle flux [mol.m-2.s-1]",
            "Discharge capacity [A.h]",
            "Throughput capacity [A.h]",
        ]

        solver_all = pybamm.IDAKLUSolver(
            atol=1e-8, rtol=1e-8,
            options=options,
        )
        sol_all = solver_all.solve(
            model,
            t_eval,
            inputs=input_parameters,
            calculate_sensitivities=True,
        )

        for varname in output_variables:
            print(varname)
            solver = pybamm.IDAKLUSolver(
                atol=1e-8, rtol=1e-8,
                options=options,
                output_variables=[varname],
            )

            sol = solver.solve(
                model,
                t_eval,
                inputs=input_parameters,
                calculate_sensitivities=True,
            )

            # Compare output to sol_all
            self.assertTrue(np.allclose(sol[varname].data, sol_all[varname].data))


if __name__ == "__main__":
    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
