"""
Tests for the latexify.py
"""

import os
import platform
import unittest
import uuid

import pybamm


class TestLatexify(unittest.TestCase):
    def test_latexify(self):
        model_dfn = pybamm.lithium_ion.DFN()
        func_dfn = str(model_dfn.latexify())

        model_spme = pybamm.lithium_ion.SPMe()
        func_spme = str(model_spme.latexify())

        # Test model name
        self.assertIn("Single Particle Model with electrolyte Equations", func_spme)

        # Test newline=False
        self.assertIn(r"\\", str(model_spme.latexify(newline=False)))

        # Test voltage equation name
        self.assertIn("Voltage [V]", func_spme)

        # Test derivative in boundary conditions
        self.assertIn(r"\nabla", func_spme)

        # Test boundary conditions range
        self.assertIn("r =", func_spme)

        # Test derivative in equations
        self.assertIn("frac{d}{d t}", func_spme)

        # Test rhs geometry ranges
        self.assertIn("0 < r < ", func_spme)

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

    def test_latexify_other_variables(self):
        model_spme = pybamm.lithium_ion.SPMe()
        func_spme = str(
            model_spme.latexify(
                output_variables=["Electrolyte concentration [mol.m-3]"]
            )
        )
        self.assertIn("Electrolyte concentration [mol.m-3]", func_spme)

        # Default behavior when voltage is not in the model variables
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: 0}
        model.initial_conditions = {var: 0}
        func = str(model.latexify())
        self.assertNotIn("Voltage [V]", func)

    @unittest.skipIf(platform.system() in ["Windows", "Darwin"], "Only run for Linux")
    def test_sympy_preview(self):
        # Test sympy preview
        model_spme = pybamm.lithium_ion.SPMe()

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
