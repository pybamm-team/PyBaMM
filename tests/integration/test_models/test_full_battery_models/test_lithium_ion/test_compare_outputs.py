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
            {"surface form": cap} for cap in ["false", "differential", "algebraic"]
        ]
        model_combos = [
            ([pybamm.lithium_ion.SPM(opt) for opt in options]),
            # ([pybamm.lithium_ion.SPMe(opt) for opt in options]), # not implemented
            ([pybamm.lithium_ion.DFN(opt) for opt in options]),
        ]

        for models in model_combos:
            # load parameter values (same for all models)
            param = models[0].default_parameter_values
            for model in models:
                param.process_model(model)

            # set mesh
            var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5}

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
        options.append({"thermal": "lumped", "cell geometry": "pouch"})

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
            var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5}

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

    def test_compare_narrow_size_distribution(self):
        # The MPM should agree with the SPM when the size distributions are narrow
        # enough.
        models = [pybamm.lithium_ion.SPM(), pybamm.lithium_ion.MPM()]

        param = models[0].default_parameter_values

        # Set size distribution parameters (lognormals)
        param = pybamm.get_size_distribution_parameters(
            param,
            sd_n=0.05,  # small standard deviations
            sd_p=0.05,
            R_min_n=0.8,
            R_min_p=0.8,
            R_max_n=1.2,
            R_max_p=1.2,
        )

        # set same mesh for both models
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5, "R_n": 5, "R_p": 5}

        # solve models
        solutions = []
        for model in models:
            sim = pybamm.Simulation(
                model,
                var_pts=var_pts,
                parameter_values=param,
                solver=pybamm.CasadiSolver(mode="fast"),
            )
            solution = sim.solve([0, 3600])
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
