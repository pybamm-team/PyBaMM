import pybamm
import unittest
from tests import get_mesh_for_testing
import sys
import time
import numpy as np


class TestJaxBDFSolver(unittest.TestCase):
    def test_solver(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "jax"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        # Solve
        t_eval = np.linspace(0, 1, 80)
        y0 = model.concatenated_initial_conditions.evaluate()
        rhs = pybamm.EvaluatorJax(model.concatenated_rhs)
        y = pybamm.jax_bdf_integrate(rhs, y0, t_eval, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(y[0], np.exp(0.1 * t_eval),
                                   rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
