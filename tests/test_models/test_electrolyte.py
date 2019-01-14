#
# Tests for the Binary Operator classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


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
        self.assertEqual(param["R"], 8.314)

        param.process_model(model)

    def test_processing_disc(self):
        G = pybamm.Scalar(1)
        model = pybamm.electrolyte.StefanMaxwellDiffusion(G)

        param = pybamm.ParameterValues(
            "input/parameters/lithium-ion/parameters/LCO.csv"
        )
        self.assertEqual(param["R"], 8.314)

        param.process_model(model)

        print(model.initial_conditions)

        # mesh = pybamm.FiniteVolumeMacroMesh(param, 2)
        # disc = pybamm.FiniteVolumeDiscretisation(mesh)
        # disc.process_model(model)
