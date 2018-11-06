from pybat_lead_acid.parameters import Parameters
from pybat_lead_acid.mesh import Mesh, UniformMesh
from pybat_lead_acid.variables import Variables
from pybat_lead_acid.spatial_operators import Operators
from pybat_lead_acid.models.model_class import Model, KNOWN_MODELS

import numpy as np
from numpy.linalg import norm

import unittest

class TestModel(unittest.TestCase):

    def test_models_shapes(self):
        param = Parameters()
        mesh = Mesh(param, 50)
        for model_name in KNOWN_MODELS:
            with self.subTest(model_name=model_name):
                model = Model(model_name)
                y0, _ = model.get_initial_conditions(param, mesh)
                vars = Variables(y0, param)
                operators = Operators("Finite Volumes", mesh)
                dydt, _ = model.get_pdes_rhs(vars, param, operators)
                self.assertEqual(y0.shape, dydt.shape)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModel)
    unittest.TextTestRunner(verbosity=2).run(suite)
