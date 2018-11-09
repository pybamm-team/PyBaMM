from pybamm.parameters import Parameters
from pybamm.mesh import Mesh, UniformMesh
from pybamm.variables import Variables
from pybamm.spatial_operators import Operators
from pybamm.models.model_class import Model, KNOWN_MODELS

import numpy as np
from numpy.linalg import norm

import unittest

class TestModel(unittest.TestCase):

    def test_models_shapes(self):
        param = Parameters()
        mesh = Mesh(param, 50)
        param.set_mesh_dependent_parameters(mesh)
        for model_name in KNOWN_MODELS:
            with self.subTest(model_name=model_name):
                model = Model(model_name)
                y0, _ = model.initial_conditions(param, mesh)
                vars = Variables(0, y0, param, mesh)
                operators = Operators("Finite Volumes", mesh)
                dydt, _ = model.pdes_rhs(vars, param, operators)
                self.assertEqual(y0.shape, dydt.shape)

    def test_models_boundary_conditions(self):
        pass

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModel)
    unittest.TextTestRunner(verbosity=2).run(suite)
