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
            solutions[0]["Voltage [V]"](solutions[0].t),
            solutions[1]["Voltage [V]"](solutions[0].t),
            decimal=5,
        )

    def test_constant_voltage(self):
        def constant_voltage(variables):
            V = variables["Voltage [V]"]
            return V - 4.08

        # load models
        models = [
            pybamm.lithium_ion.SPM({"operating mode": "voltage"}),
            pybamm.lithium_ion.SPM({"operating mode": constant_voltage}),
        ]

        # load parameter values and process models and geometry
        params = [model.default_parameter_values for model in models]

        # First model: 4.08V charge
        params[0].update({"Voltage function [V]": 4.08}, check_already_exists=False)

        # set parameters and discretise models
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 30, "r_n": 10, "r_p": 10}
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

        V0 = solutions[0]["Voltage [V]"].entries
        V1 = solutions[1]["Voltage [V]"].entries
        np.testing.assert_array_almost_equal(V0, V1)

        I0 = solutions[0]["Current [A]"].entries
        I1 = solutions[1]["Current [A]"].entries
        np.testing.assert_array_almost_equal(abs((I1 - I0) / I0), 0, decimal=1)

    def test_constant_power(self):
        def constant_power(variables):
            I = variables["Current [A]"]
            V = variables["Voltage [V]"]
            return I * V - 4

        # load models
        # use DFN since only DFN allows "explicit power"
        models = [
            pybamm.lithium_ion.DFN({"operating mode": "power"}),
            pybamm.lithium_ion.DFN({"operating mode": "explicit power"}),
            pybamm.lithium_ion.DFN({"operating mode": constant_power}),
        ]

        # set parameters and discretise models
        solutions = [None] * len(models)
        t_eval = np.linspace(0, 3600, 100)
        for i, model in enumerate(models):
            # create geometry
            geometry = model.default_geometry
            param = model.default_parameter_values
            param.update({"Power function [W]": 4}, check_already_exists=False)
            param.process_model(model)
            param.process_geometry(geometry)
            mesh = pybamm.Mesh(
                geometry, model.default_submesh_types, model.default_var_pts
            )
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            disc.process_model(model)
            solutions[i] = model.default_solver.solve(model, t_eval)

        for var in ["Voltage [V]", "Current [A]"]:
            for sol in solutions[1:]:
                np.testing.assert_array_almost_equal(
                    solutions[0][var].data, sol[var].data
                )

    def test_constant_resistance(self):
        def constant_resistance(variables):
            I = variables["Current [A]"]
            V = variables["Voltage [V]"]
            return V / I - 2

        # load models
        # use DFN since only DFN allows "explicit resistance"
        models = [
            pybamm.lithium_ion.DFN({"operating mode": "resistance"}),
            pybamm.lithium_ion.DFN({"operating mode": "explicit resistance"}),
            pybamm.lithium_ion.DFN({"operating mode": constant_resistance}),
        ]

        # set parameters and discretise models
        solutions = [None] * len(models)
        t_eval = np.linspace(0, 3600, 100)
        for i, model in enumerate(models):
            # create geometry
            geometry = model.default_geometry
            param = model.default_parameter_values
            param.update({"Resistance function [Ohm]": 2}, check_already_exists=False)
            param.process_model(model)
            param.process_geometry(geometry)
            mesh = pybamm.Mesh(
                geometry, model.default_submesh_types, model.default_var_pts
            )
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            disc.process_model(model)
            solutions[i] = model.default_solver.solve(model, t_eval)

        for var in ["Voltage [V]", "Current [A]"]:
            for sol in solutions[1:]:
                np.testing.assert_array_almost_equal(
                    solutions[0][var].data, sol[var].data
                )

    def test_cccv(self):
        # load models
        model = pybamm.lithium_ion.SPM({"operating mode": "CCCV"})

        # load parameter values and process models and geometry
        param = model.default_parameter_values

        param.update(
            {"CCCV current function [A]": -0.5, "Voltage function [V]": 4.2},
            check_already_exists=False,
        )

        # set parameters and discretise models
        # create geometry
        geometry = model.default_geometry
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # solve model
        t_eval = np.linspace(0, 3600, 100)
        model.default_solver.solve(model, t_eval)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
