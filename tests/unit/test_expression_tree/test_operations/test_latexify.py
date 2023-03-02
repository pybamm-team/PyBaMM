"""
Tests for the latexify.py
"""
import os
import platform
import unittest
import uuid

import pybamm
from pybamm.expression_tree.operations.latexify import Latexify

model_dfn = pybamm.lithium_ion.DFN()
func_dfn = str(model_dfn.latexify())

model_spme = pybamm.lithium_ion.SPMe()
func_spme = str(model_spme.latexify())


class TestLatexify(unittest.TestCase):
    def test_latexify(self):
        # Test docstring
        self.assertEqual(pybamm.BaseModel.latexify.__doc__, Latexify.__doc__)

        # Test model name
        self.assertIn("Single Particle Model with electrolyte Equations", func_spme)

        # Test newline=False
        self.assertIn(r"\\", str(model_spme.latexify(newline=False)))

        # Test voltage equation name
        self.assertIn("Voltage [V]", func_spme)

        # Test partial derivative in boundary conditions
        self.assertIn("partial r}", func_spme)

        # Test boundary conditions range
        self.assertIn("quad r =", func_spme)

        # Test derivative in equations
        self.assertIn("frac{d}{d t}", func_spme)

        # Test rhs geometry ranges
        self.assertIn("quad 0 < r < ", func_spme)

        # Test initial conditions
        self.assertIn("; t=0", func_spme)

        # Test DFN algebraic lhs
        self.assertIn("0 =", func_dfn)

        # Test concatenation cases
        try:
            self.assertIn("begin{cases}", func_spme)
            self.assertIn("end{cases}", func_spme)

        except AssertionError:
            for eqn in model_spme.rhs.values():
                concat_displays = model_spme._get_concat_displays(eqn)
                if concat_displays:
                    self.assertIn("begin{cases}", str(concat_displays))
                    self.assertIn("end{cases}", str(concat_displays))
                    break

        # Test parameters and variables
        self.assertIn("Parameters and Variables", func_spme)
        self.assertIn("coefficient", func_spme)
        self.assertIn("diffusivity", func_spme)

    @unittest.skipIf(platform.system() in ["Windows", "Darwin"], "Only run for Linux")
    def test_sympy_preview(self):
        # Test sympy preview
        for ext in ["png", "jpg", "pdf"]:
            filename = f"{uuid.uuid4()}.{ext}"
            model_spme.latexify(filename)
            os.remove(filename)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
