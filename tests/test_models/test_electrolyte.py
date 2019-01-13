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
            base_parameters={
                "epsilon_s": 0.01,
                "F": 0.02,
                "t_plus": 0.03,
                "Ln": 0.04,
                "Ls": 0.05,
                "Lp": 0.06,
                "I_typ": 1.1,
                "cn_max": 2.2,
                "De_typ": 3.3,
                "ce_typ": 6.6,
                "b": 4.4,
                "ce0": 5.5,
            }
        )
        # just going to skip for now.
        # param.process_model(model)
