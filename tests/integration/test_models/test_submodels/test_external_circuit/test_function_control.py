#
# Test function control submodel
#
import numpy as np
import pybamm
import unittest


class TestFunctionControl(unittest.TestCase):
    def test_constant_current(self):
        class ConstantCurrent:
            num_switches = 0

            def __call__(self, variables):
                I = variables["Current [A]"]
                return I - 1

        # load models
        models = [
            pybamm.lithium_ion.SPM(),
            pybamm.lithium_ion.SPM({"operating mode": ConstantCurrent()}),
        ]

        # load parameter values and process models and geometry
        params = [model.default_parameter_values for model in models]

        # First model: 1A charge
        params[0]["Typical current [A]"] = 1

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
        )(solutions[0].t)
        V1 = pybamm.ProcessedVariable(
            models[1].variables["Terminal voltage [V]"],
            solutions[1].t,
            solutions[1].y,
            mesh,
        )(solutions[1].t)
        pv0 = pybamm.post_process_variables(
            models[0].variables, solutions[0].t, solutions[0].y, mesh
        )
        pv1 = pybamm.post_process_variables(
            models[1].variables, solutions[1].t, solutions[1].y, mesh
        )
        import ipdb

        ipdb.set_trace()
        np.testing.assert_array_equal(V0, V1)

    def test_constant_voltage(self):
        class ConstantVoltage:
            num_switches = 0

            def __call__(self, variables):
                V = variables["Terminal voltage [V]"]
                return V - 4.1

        # load models
        # test the DFN for this one as it has a particular implementation of constant
        # voltage
        models = [
            pybamm.lithium_ion.DFN({"operating mode": "voltage"}),
            pybamm.lithium_ion.DFN({"operating mode": ConstantVoltage()}),
        ]

        # load parameter values and process models and geometry
        params = [model.default_parameter_values for model in models]

        # First model: 4.1V charge
        params[0]["Voltage function"] = 4.1

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
        class ConstantPower:
            num_switches = 0

            def __call__(self, variables):
                I = variables["Current [A]"]
                V = variables["Terminal voltage [V]"]
                return I * V - 4

        # load models
        models = [
            pybamm.lithium_ion.SPM({"operating mode": "power"}),
            pybamm.lithium_ion.SPM({"operating mode": ConstantPower()}),
        ]

        # load parameter values and process models and geometry
        params = [model.default_parameter_values for model in models]

        # First model: 4W charge
        params[0]["Power function"] = 4

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
