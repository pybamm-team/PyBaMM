#
# Tests for the asymptotic convergence of the simplified models
#
import pybamm
import numpy as np
from tests import StandardOutputComparison


class TestCompareOutputs:
    def test_compare_averages_asymptotics(self):
        """
        Check that the average value of certain variables is constant across submodels
        """
        # load models
        models = [pybamm.lead_acid.LOQS(), pybamm.lead_acid.Full()]

        # load parameter values (same for all models)
        param = models[0].default_parameter_values
        param.update({"Current function [A]": 1})
        for model in models:
            param.process_model(model)

        # set mesh
        var_pts = {"x_n": 10, "x_s": 10, "x_p": 10}

        # discretise models
        for model in models:
            geometry = model.default_geometry
            param.process_geometry(geometry)
            mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            disc.process_model(model)

        # solve model
        solutions = []
        t_eval = np.linspace(0, 3600 * 17, 100)
        for model in models:
            solution = pybamm.CasadiSolver().solve(model, t_eval)
            solutions.append(solution)

        # test averages
        comparison = StandardOutputComparison(solutions)
        comparison.test_averages()

    def test_compare_outputs_surface_form(self):
        """
        Check that the models agree with the different surface forms
        """
        # load models
        options = [
            {"surface form": cap} for cap in ["false", "differential", "algebraic"]
        ]
        model_combos = [
            ([pybamm.lead_acid.LOQS(opt) for opt in options]),
            ([pybamm.lead_acid.Full(opt) for opt in options]),
        ]

        for models in model_combos:
            # load parameter values (same for all models)
            param = models[0].default_parameter_values
            param.update({"Current function [A]": 1})
            for model in models:
                param.process_model(model)

            # set mesh
            var_pts = {"x_n": 5, "x_s": 5, "x_p": 5}

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
            t_eval = np.linspace(0, 3600 * 20, 100)
            for model in models:
                solution = pybamm.CasadiSolver().solve(model, t_eval)
                solutions.append(solution)

            # compare outputs
            comparison = StandardOutputComparison(solutions)
            comparison.test_all(skip_first_timestep=True)
