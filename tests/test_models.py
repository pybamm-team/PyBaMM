import pybamm
import pybamm.models as models

import unittest


class TestModel(unittest.TestCase):
    def test_models_shapes(self):
        param = pybamm.Parameters()
        mesh = pybamm.Mesh(param, 50)
        param.set_mesh_dependent_parameters(mesh)
        for model_name in models.KNOWN_MODELS:
            with self.subTest(model_name=model_name):
                model = pybamm.Model(model_name)
                y0 = model.initial_conditions(param, mesh)
                vars = pybamm.Variables(0, y0, model, mesh)
                operators = {
                    domain: pybamm.Operators("Finite Volumes", domain, mesh)
                    for domain in model.domains()
                }
                dydt = model.pdes_rhs(vars, param, operators)
                self.assertEqual(y0.shape, dydt.shape)

    def test_models_boundary_conditions(self):
        pass


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModel)
    unittest.TextTestRunner(verbosity=2).run(suite)
