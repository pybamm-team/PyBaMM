#
# Test basic half-cell model with different parameter values
#
import pybamm

import numpy as np
import unittest


class TestBasicHalfCellModels(unittest.TestCase):
    def test_runs_Xu2019(self):
        options = {"working electrode": "positive"}
        model = pybamm.lithium_ion.BasicDFNHalfCell(options=options)

        # create geometry
        geometry = model.default_geometry

        # load parameter values
        param = pybamm.ParameterValues("Xu2019")

        param["Current function [A]"] = 2.5e-3

        # process model and geometry
        param.process_model(model)
        param.process_geometry(geometry)

        # set mesh
        var_pts = model.default_var_pts
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # solve model
        t_eval = np.linspace(0, 7200, 1000)
        solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3)
        solver.solve(model, t_eval)

    def test_runs_Chen2020(self):
        # load model
        options = {"working electrode": "positive"}
        model = pybamm.lithium_ion.BasicDFNHalfCell(options=options)

        # create geometry
        geometry = model.default_geometry

        # load parameter values
        param = pybamm.ParameterValues({
            "chemistry": "lithium_ion",
            "cell": "LGM50_Chen2020",
            "negative electrode": "graphite_Chen2020",
            "separator": "separator_Chen2020",
            "positive electrode": "nmc_Chen2020",
            "electrolyte": "lipf6_Nyman2008",
            "experiment": "1C_discharge_from_full_Chen2020",
            "sei": "example",
            "lithium plating": "okane2020_Li_plating",
        })

        param["Current function [A]"] = 2.5

        # process model and geometry
        param.process_model(model)
        param.process_geometry(geometry)

        # set mesh
        var_pts = model.default_var_pts
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # solve model
        t_eval = np.linspace(0, 7200, 1000)
        solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3)
        solver.solve(model, t_eval)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
