#
# Test function control submodel
#
import numpy as np
import pybamm
import tests
import unittest


class TestFunctionControl(unittest.TestCase):
    def test_constant_current(self):
        # load models
        models = [
            pybamm.lithium_ion.SPM(),
            pybamm.lithium_ion.SPM({"operating mode": "custom"}),
        ]

        # load parameter values and process models and geometry
        params = [model.default_parameter_values for model in models]

        # First model: 1A charge
        params[0]["Typical current [A]"] = -1

        # Second model: 1C charge via a function
        def constant_current(I, V):
            return I + 1

        params[1]["External circuit function"] = constant_current

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
        t_eval = np.linspace(0, 1, 100)
        for i, model in enumerate(models):
            solutions[i] = model.default_solver.solve(model, t_eval)

        V0 = pybamm.ProcessedVariable(
            models[0].variables["Terminal voltage [V]"],
            solutions[0].t,
            solutions[0].y,
            mesh,
        ).entries
        V1 = pybamm.ProcessedVariable(
            models[1].variables["Terminal voltage [V]"],
            solutions[1].t,
            solutions[1].y,
            mesh,
        ).entries
        np.testing.assert_array_equal(V0, V1)

    def test_constant_voltage(self):
        # load models
        models = [
            pybamm.lithium_ion.SPM({"operating mode": "voltage"}),
            pybamm.lithium_ion.SPM({"operating mode": "custom"}),
        ]

        # load parameter values and process models and geometry
        params = [model.default_parameter_values for model in models]

        # First model: 4.1V charge
        params[0]["Voltage function"] = 4.1

        # Second model: 4.1V charge via a function
        def constant_voltage(I, V):
            return V - 4.1

        params[1]["External circuit function"] = constant_voltage

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
        t_eval = np.linspace(0, 1, 100)
        for i, model in enumerate(models):
            solutions[i] = model.default_solver.solve(model, t_eval)

        V0 = pybamm.ProcessedVariable(
            models[0].variables["Terminal voltage [V]"],
            solutions[0].t,
            solutions[0].y,
            mesh,
        ).entries
        V1 = pybamm.ProcessedVariable(
            models[1].variables["Terminal voltage [V]"],
            solutions[1].t,
            solutions[1].y,
            mesh,
        ).entries
        np.testing.assert_array_equal(V0, V1)

        I0 = pybamm.ProcessedVariable(
            models[0].variables["Current [A]"], solutions[0].t, solutions[0].y, mesh
        ).entries
        I1 = pybamm.ProcessedVariable(
            models[1].variables["Current [A]"], solutions[1].t, solutions[1].y, mesh
        ).entries
        np.testing.assert_array_equal(I0, I1)

    def test_constant_power(self):
        # load models
        models = [
            pybamm.lithium_ion.SPM({"operating mode": "power"}),
            pybamm.lithium_ion.SPM({"operating mode": "custom"}),
        ]

        # load parameter values and process models and geometry
        params = [model.default_parameter_values for model in models]

        # First model: 4W charge
        params[0]["Power function"] = 4

        # Second model: 4W charge via a function
        def constant_power(I, V):
            return I * V - 4

        params[1]["External circuit function"] = constant_power

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
        t_eval = np.linspace(0, 1, 100)
        for i, model in enumerate(models):
            solutions[i] = model.default_solver.solve(model, t_eval)

        V0 = pybamm.ProcessedVariable(
            models[0].variables["Terminal voltage [V]"],
            solutions[0].t,
            solutions[0].y,
            mesh,
        ).entries
        V1 = pybamm.ProcessedVariable(
            models[1].variables["Terminal voltage [V]"],
            solutions[1].t,
            solutions[1].y,
            mesh,
        ).entries
        np.testing.assert_array_equal(V0, V1)

        I0 = pybamm.ProcessedVariable(
            models[0].variables["Current [A]"], solutions[0].t, solutions[0].y, mesh
        ).entries
        I1 = pybamm.ProcessedVariable(
            models[1].variables["Current [A]"], solutions[1].t, solutions[1].y, mesh
        ).entries
        np.testing.assert_array_equal(I0, I1)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
