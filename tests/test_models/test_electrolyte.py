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

        y0 = model.concatenated_initial_conditions
        np.testing.assert_array_equal(y0, np.ones_like(mesh["whole cell"].nodes))

        # Solve
        solver = pybamm.ScipySolver(tol=1e-8, method="RK45")
        t_eval = mesh["time"]
        solver.solve(model, t_eval)
