#
# Tests for the Binary Operator classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest
import numpy as np


class TestStefanMaxwellDiffusion(unittest.TestCase):
    def test_make_tree(self):
        G = pybamm.Scalar(1)
        pybamm.electrolyte.StefanMaxwellDiffusion(G)

    def test_processing_parameters(self):
        G = pybamm.Scalar(1)
        model = pybamm.electrolyte.StefanMaxwellDiffusion(G)

        param = pybamm.ParameterValues(
            "input/parameters/lithium-ion/parameters/LCO.csv"
        )

        param.process_model(model)

    def test_processing_disc(self):
        G = pybamm.Scalar(1)
        model = pybamm.electrolyte.StefanMaxwellDiffusion(G)

        param = pybamm.ParameterValues(
            "input/parameters/lithium-ion/parameters/LCO.csv"
        )

        param.process_model(model)

        mesh = pybamm.FiniteVolumeMacroMesh(param, 2)
        disc = pybamm.FiniteVolumeDiscretisation(mesh)
        disc.process_model(model)

    def test_solving(self):
        G = pybamm.Scalar(0.001)
        model = pybamm.electrolyte.StefanMaxwellDiffusion(G)

        param = pybamm.ParameterValues(
            "input/parameters/lithium-ion/parameters/LCO.csv"
        )

        param.process_model(model)

        mesh = pybamm.FiniteVolumeMacroMesh(param, 2)
        disc = pybamm.FiniteVolumeDiscretisation(mesh)
        disc.process_model(model)

        # c_e, rhs = model.rhs.items()
        # print("\nmodel rhs: ", model.rhs)
        # print("model rhs type: ", type(model.rhs))
        # print("model initial conditions: ", model.initial_conditions)
        # print("model boundary conditions: ", model.boundary_conditions)

        y0 = model.initial_conditions
        np.testing.assert_array_equal(y0, np.ones_like(mesh["whole cell"].nodes))
        print("model rhs is: ", model.rhs.evaluate(None, y0))

        # Solve
        # solver = pybamm.ScipySolver(tol=1e-8, method="RK45")
        # t_eval = mesh["time"]
        # print("This is the solver: ", solver)
        # print("This is t_eval: ", t_eval)
        # solver.solve(model, t_eval)
