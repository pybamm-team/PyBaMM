#
# Tests for the asymptotic convergence of the simplified models
#
import pybamm
import numpy as np
import unittest
from tests import StandardOutputComparison


class TestCompareOutputs(unittest.TestCase):
    def test_compare_outputs_low_current(self):
        """
        Check that the leading-order model solution converges linearly in C_e to the
        full model solution
        """
        # load models
        models = [
            pybamm.lead_acid.LOQS(),
            pybamm.lead_acid.Composite(),
            pybamm.lead_acid.NewmanTiedemann(),
        ]

        # load parameter values (same for all models)
        param = models[0].default_parameter_values
        param.update({"Typical current": 0.01})
        for model in models:
            param.process_model(model)

        # set mesh
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10}

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
        solvers = {}
        t_eval = np.linspace(0, 1, 100)
        for i, model in enumerate(models):
            solver = model.default_solver
            solver.solve(model, t_eval)
            solvers[model] = solver

        # test averages
        comparison = StandardOutputComparison(models, param, discs, solvers)
        comparison.test_averages()

        # compare leading and first order
        comparison = StandardOutputComparison(models[:2], param, discs, solvers)
        comparison.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
