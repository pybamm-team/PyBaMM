#
# Test function control submodel
#
import numpy as np
import pybamm
import unittest


class TestFunctionControl(unittest.TestCase):
    def test_constant_current(self):
        def constant_current(variables):
            I = variables["Current [A]"]
            return I + 1

        # load models
        models = [
            pybamm.lithium_ion.SPM(),
            pybamm.lithium_ion.SPM({"operating mode": constant_current}),
        ]

        # load parameter values and process models and geometry
        params = [model.default_parameter_values for model in models]

        # First model: 1A charge
        params[0]["Current function [A]"] = -1
        params[1]["Current function [A]"] = -1

        # set parameters and discretise models
        for i, model in enumerate(models):
            # create geometry
            geometry = model.default_geometry
            params[i].process_model(model)
            params[i].process_geometry(geometry)
            mesh = pybamm.Mesh(
                geometry, model.default_submesh_types, model.default_var_pts
            )
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            disc.process_model(model)

        # solve model
        solutions = [None] * len(models)
        t_eval = np.linspace(0, 3600, 100)
        for i, model in enumerate(models):
            solutions[i] = model.default_solver.solve(model, t_eval)

        np.testing.assert_array_almost_equal(
            solutions[0]["Discharge capacity [A.h]"].entries,
            solutions[0]["Current [A]"].entries * solutions[0]["Time [h]"].entries,
        )
        np.testing.assert_array_almost_equal(
            solutions[0]["Terminal voltage [V]"](solutions[0].t),
            solutions[1]["Terminal voltage [V]"](solutions[0].t),
        )

    def test_constant_voltage(self):
        def constant_voltage(variables):
            V = variables["Terminal voltage [V]"]
            return V - 4.1

        # load models
        models = [
            pybamm.lithium_ion.SPM({"operating mode": "voltage"}),
            pybamm.lithium_ion.SPM({"operating mode": constant_voltage}),
        ]

        # load parameter values and process models and geometry
        params = [model.default_parameter_values for model in models]

        # First model: 4.1V charge
        params[0]["Voltage function [V]"] = 4.1

        # set parameters and discretise models
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 30, var.r_n: 10, var.r_p: 10}
        for i, model in enumerate(models):
            # create geometry
            geometry = model.default_geometry
            params[i].process_model(model)
            params[i].process_geometry(geometry)
            mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            disc.process_model(model)

        # solve model
        solutions = [None] * len(models)
        t_eval = np.linspace(0, 3600, 100)
        for i, model in enumerate(models):
            solutions[i] = model.default_solver.solve(model, t_eval)

        V0 = solutions[0]["Terminal voltage [V]"].entries
        V1 = solutions[1]["Terminal voltage [V]"].entries
        np.testing.assert_array_almost_equal(V0, V1)

        I0 = solutions[0]["Current [A]"].entries
        I1 = solutions[1]["Current [A]"].entries
        np.testing.assert_array_almost_equal(abs((I1 - I0) / I0), 0, decimal=1)

    def test_constant_power(self):
        def constant_power(variables):
            I = variables["Current [A]"]
            V = variables["Terminal voltage [V]"]
            return I * V - 4

        # load models
        models = [
            pybamm.lithium_ion.SPM({"operating mode": "power"}),
            pybamm.lithium_ion.SPM({"operating mode": constant_power}),
        ]

        # load parameter values and process models and geometry
        params = [model.default_parameter_values for model in models]

        # First model: 4W charge
        params[0]["Power function [W]"] = 4

        # set parameters and discretise models
        for i, model in enumerate(models):
            # create geometry
            geometry = model.default_geometry
            params[i].process_model(model)
            params[i].process_geometry(geometry)
            mesh = pybamm.Mesh(
                geometry, model.default_submesh_types, model.default_var_pts
            )
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            disc.process_model(model)

        # solve model
        solutions = [None] * len(models)
        t_eval = np.linspace(0, 3600, 100)
        for i, model in enumerate(models):
            solutions[i] = model.default_solver.solve(model, t_eval)

        V0 = solutions[0]["Terminal voltage [V]"].entries
        V1 = solutions[1]["Terminal voltage [V]"].entries
        np.testing.assert_array_equal(V0, V1)

        I0 = solutions[0]["Current [A]"].entries
        I1 = solutions[1]["Current [A]"].entries
        np.testing.assert_array_equal(I0, I1)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
