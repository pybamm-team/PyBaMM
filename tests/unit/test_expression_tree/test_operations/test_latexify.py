"""
Tests for the latexify.py
"""

import pytest
import os
import platform
import uuid

import pybamm


class TestLatexify:
    def test_latexify(self):
        model_dfn = pybamm.lithium_ion.DFN()
        func_dfn = str(model_dfn.latexify())

        model_spme = pybamm.lithium_ion.SPMe()
        func_spme = str(model_spme.latexify())

        # Test model name
        assert "Single Particle Model with electrolyte Equations" in func_spme

        # Test newline=False
        assert r"\\" in str(model_spme.latexify(newline=False))

        # Test voltage equation name
        assert "Voltage [V]" in func_spme

        # Test derivative in boundary conditions
        assert r"\nabla" in func_spme

        # Test boundary conditions range
        assert "r =" in func_spme

        # Test derivative in equations
        assert "frac{d}{d t}" in func_spme

        # Test rhs geometry ranges
        assert "0 < r < " in func_spme

        # Test initial conditions
        assert "; t=0" in func_spme

        # Test DFN algebraic lhs
        assert "0 =" in func_dfn

        # Test concatenation cases
        try:
            assert "begin{cases}" in func_spme
            assert "end{cases}" in func_spme

        except AssertionError:
            for eqn in model_spme.rhs.values():
                concat_displays = model_spme._get_concat_displays(eqn)
                if concat_displays:
                    assert "begin{cases}" in str(concat_displays)
                    assert "end{cases}" in str(concat_displays)
                    break

        # Test parameters and variables
        assert "Parameters and Variables" in func_spme
        assert "coefficient" in func_spme
        assert "diffusivity" in func_spme

    def test_latexify_other_variables(self):
        model_spme = pybamm.lithium_ion.SPMe()
        func_spme = str(
            model_spme.latexify(
                output_variables=["Electrolyte concentration [mol.m-3]"]
            )
        )
        assert "Electrolyte concentration [mol.m-3]" in func_spme

        # Default behavior when voltage is not in the model variables
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: 0}
        model.initial_conditions = {var: 0}
        func = str(model.latexify())
        assert "Voltage [V]" not in func

    @pytest.mark.skipif(
        platform.system() in ["Windows", "Darwin"], reason="Only run for Linux"
    )
    def test_sympy_preview(self):
        # Test sympy preview
        model_spme = pybamm.lithium_ion.SPMe()

        for ext in ["png", "jpg", "pdf"]:
            filename = f"{uuid.uuid4()}.{ext}"
            model_spme.latexify(filename)
            os.remove(filename)
