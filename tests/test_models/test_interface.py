#
# Tests for the electrode-electrolyte interface equations
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

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
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        param = pybamm.ParameterValues(
            "input/parameters/lithium-ion/parameters/LCO.csv"
        )
        geometry = pybamm.Geometry1DMacro()
        param.process_geometry(geometry)

        mesh_type = pybamm.Mesh
        submesh_pts = {
            "negative electrode": {"x": 40},
            "separator": {"x": 25},
            "positive electrode": {"x": 35},
        }
        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
        }

        disc = pybamm.FiniteVolumeDiscretisation(mesh_type, submesh_pts, submesh_types)
        disc._mesh = mesh_type(geometry, submesh_types, submesh_pts)
        rxn = pybamm.interface.homogeneous_reaction()

        param_rxn = param.process_symbol(rxn)
        processed_rxn = disc.process_symbol(param_rxn)

        combined_submeshes = disc.mesh.combine_submeshes(*whole_cell)
        # processed_rxn should be a vector with the right shape
        self.assertIsInstance(processed_rxn, pybamm.Vector)
        self.assertEqual(processed_rxn.shape, combined_submeshes.nodes.shape)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
