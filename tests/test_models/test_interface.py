#
# Tests for the electrode-electrolyte interface equations
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import unittest


class TestHomogeneousReaction(unittest.TestCase):
    def test_set_parameters(self):
        param = pybamm.ParameterValues(
            "input/parameters/lithium-ion/parameters/LCO.csv"
        )
        rxn = pybamm.interface.homogeneous_reaction()

        processed_rxn = param.process_symbol(rxn)

        # rxn (a concatenation of functions of scalars and parameters) should get
        # discretised to a concantenation of functions of scalars
        self.assertIsInstance(processed_rxn, pybamm.Concatenation)
        self.assertFalse(
            any(
                [
                    isinstance(x, pybamm.Parameter)
                    for x in processed_rxn.children[0].pre_order()
                ]
            )
        )
        self.assertFalse(
            any(
                [
                    isinstance(x, pybamm.Parameter)
                    for x in processed_rxn.children[2].pre_order()
                ]
            )
        )
        self.assertIsInstance(processed_rxn.children[1], pybamm.Scalar)
        self.assertEqual(processed_rxn.children[0].domain, ["negative electrode"])
        self.assertEqual(processed_rxn.children[1].domain, ["separator"])
        self.assertEqual(processed_rxn.children[2].domain, ["positive electrode"])

        # test values
        ln = param.process_symbol(pybamm.standard_parameters.ln)
        ln * 1 / ln
        lp = param.process_symbol(pybamm.standard_parameters.lp)
        self.assertEqual(processed_rxn.children[0].evaluate() * ln.evaluate(), 1)
        self.assertEqual(processed_rxn.children[2].evaluate() * lp.evaluate(), -1)

    def test_discretisation(self):
        param = pybamm.ParameterValues(
            "input/parameters/lithium-ion/parameters/LCO.csv"
        )
        mesh = pybamm.FiniteVolumeMacroMesh(param, 2)
        disc = pybamm.FiniteVolumeDiscretisation(mesh)

        rxn = pybamm.interface.homogeneous_reaction()

        param_rxn = param.process_symbol(rxn)
        processed_rxn = disc.process_symbol(param_rxn)

        # processed_rxn should be a vector with the right shape
        self.assertIsInstance(processed_rxn, pybamm.Vector)
        self.assertEqual(processed_rxn.shape, mesh["whole cell"].nodes.shape)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
