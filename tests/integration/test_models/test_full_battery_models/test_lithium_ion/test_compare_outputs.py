#
# Tests for the surface formulation
#
import pybamm
import numpy as np
import unittest
from tests import StandardOutputComparison


class TestCompareOutputs(unittest.TestCase):
    def test_compare_outputs_surface_form(self):
        # load models
        options = [
            {"surface form": cap} for cap in [False, "differential", "algebraic"]
        ]
        model_combos = [
            ([pybamm.lithium_ion.SPM(opt) for opt in options]),
            ([pybamm.lithium_ion.SPMe(opt) for opt in options]),
            ([pybamm.lithium_ion.DFN(opt) for opt in options]),
        ]

        for models in model_combos:
            # load parameter values (same for all models)
            param = models[0].default_parameter_values
            param.update({"Current function [A]": 1})
            for model in models:
                param.process_model(model)

            # set mesh
            var = pybamm.standard_spatial_vars
            var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 5, var.r_p: 5}

            # discretise models
            discs = {}
            for model in models:
                geometry = model.default_geometry
                param.process_geometry(geometry)
                mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
                disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
                disc.process_model(model)
                discs[model] = disc

            # solve model
            solutions = []
            t_eval = np.linspace(0, 3600, 100)
            for model in models:
                solution = pybamm.CasadiSolver().solve(model, t_eval)
                solutions.append(solution)

            # compare outputs
            comparison = StandardOutputComparison(solutions)
            comparison.test_all(skip_first_timestep=True)

    def test_compare_outputs_thermal(self):
        # load models - for the default params we expect x-full and lumped to
        # agree as the temperature is practically independent of x
        options = [{"thermal": opt} for opt in ["lumped", "x-full"]]
        options.append({"thermal": "lumped", "cell_geometry": "pouch"})

        model_combos = [
            ([pybamm.lithium_ion.SPM(opt) for opt in options]),
            ([pybamm.lithium_ion.SPMe(opt) for opt in options]),
            ([pybamm.lithium_ion.DFN(opt) for opt in options]),
        ]

        for models in model_combos:
            # load parameter values (same for all models)
            param = models[0].default_parameter_values

            # for x-full, cooling is only implemented on the surfaces
            # so set other forms of cooling to zero for comparison.
            param.update(
                {
                    "Negative current collector"
                    + " surface heat transfer coefficient [W.m-2.K-1]": 5,
                    "Positive current collector"
                    + " surface heat transfer coefficient [W.m-2.K-1]": 5,
                    "Negative tab heat transfer coefficient [W.m-2.K-1]": 0,
                    "Positive tab heat transfer coefficient [W.m-2.K-1]": 0,
                    "Edge heat transfer coefficient [W.m-2.K-1]": 0,
                }
            )
            for model in models:
                param.process_model(model)

            # set mesh
            var = pybamm.standard_spatial_vars
            var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 5, var.r_p: 5}

            # discretise models
            discs = {}
            for model in models:
                geometry = model.default_geometry
                param.process_geometry(geometry)
                mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
                disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
                disc.process_model(model)
                discs[model] = disc

            # solve model
            solutions = []
            t_eval = np.linspace(0, 3600, 100)
            for model in models:
                solution = pybamm.CasadiSolver().solve(model, t_eval)
                solutions.append(solution)

            # compare outputs
            comparison = StandardOutputComparison(solutions)
            comparison.test_all(skip_first_timestep=True)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
