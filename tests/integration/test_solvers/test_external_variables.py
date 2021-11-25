#
# Test solvers with external variables
#
import pybamm
import numpy as np
import os
import unittest


class TestExternalVariables(unittest.TestCase):
    def test_on_dfn(self):
        e_height = 0.25

        model = pybamm.lithium_ion.DFN()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.update({"Negative electrode conductivity [S.m-1]": "[input]"})
        param.process_model(model)
        param.process_geometry(geometry)
        inputs = {"Negative electrode conductivity [S.m-1]": e_height}
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 10, "r_p": 10}
        spatial_methods = model.default_spatial_methods

        solver = pybamm.CasadiSolver()
        sim = pybamm.Simulation(
            model=model,
            geometry=geometry,
            parameter_values=param,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
            solver=solver,
        )
        sim.solve(t_eval=np.linspace(0, 3600, 100), inputs=inputs)

    def test_external_variables_SPMe_thermal(self):
        model_options = {"thermal": "lumped", "external submodels": ["thermal"]}
        model = pybamm.lithium_ion.SPMe(model_options)
        sim = pybamm.Simulation(model)
        t_eval = np.linspace(0, 100, 3)
        T_av = 0
        for i in np.arange(1, len(t_eval) - 1):
            dt = t_eval[i + 1] - t_eval[i]
            external_variables = {"Volume-averaged cell temperature": T_av}
            T_av += 1
            sim.step(dt, external_variables=external_variables)
        V = sim.solution["Terminal voltage [V]"].data
        np.testing.assert_array_less(np.diff(V), 0)
        # test generate with external variable
        sim.built_model.generate("test.c", ["Volume-averaged cell temperature"])
        os.remove("test.c")


if __name__ == "__main__":
    import sys

    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
