"""
Tests for the latexify.py
"""

import shutil

import pytest

import pybamm


class TestLatexify:
    @pytest.fixture
    def ode_model(self):
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: 0}
        model.initial_conditions = {var: 0}
        return model

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

    def test_latexify_other_variables(self, ode_model):
        model_spme = pybamm.lithium_ion.SPMe()
        func_spme = str(
            model_spme.latexify(
                output_variables=["Electrolyte concentration [mol.m-3]"]
            )
        )
        assert "Electrolyte concentration [mol.m-3]" in func_spme

        # Default behavior when voltage is not in the model variables
        func = str(ode_model.latexify())
        assert "Voltage [V]" not in func

    @pytest.mark.parametrize(
        ("filename", "expected_kwargs"),
        [
            ("eqn.tex", {"outputTexFile": "eqn.tex"}),
            (
                "eqn.pdf",
                {
                    "output": "pdf",
                    "viewer": "file",
                    "filename": "eqn.pdf",
                    "euler": False,
                },
            ),
            (
                "eqn.png",
                {
                    "viewer": "file",
                    "filename": "eqn.png",
                    "dvioptions": ["-D", "900"],
                    "euler": False,
                },
            ),
            (
                "eqn.jpg",
                {
                    "viewer": "file",
                    "filename": "eqn.jpg",
                    "dvioptions": ["-D", "900"],
                    "euler": False,
                },
            ),
        ],
    )
    def test_sympy_preview(self, mocker, ode_model, filename, expected_kwargs):
        preview = mocker.patch("sympy.preview")
        ode_model.latexify(filename)
        preview.assert_called_once()
        assert preview.call_args.kwargs == expected_kwargs

    def test_sympy_preview_resolution_fallback(self, mocker, ode_model):
        preview = mocker.patch("sympy.preview", side_effect=[RuntimeError, None])
        with pytest.warns(UserWarning, match="output resolution to default"):
            ode_model.latexify("eqn.png")
        assert preview.call_count == 2
        assert "dvioptions" not in preview.call_args.kwargs

    @pytest.mark.parametrize(
        ("ext", "converters"),
        [
            ("png", ["dvipng"]),
            ("jpg", ["dvipng"]),
            ("pdf", ["dvipdfmx", "dvipdfm", "dvipdf"]),
        ],
    )
    def test_sympy_preview_renders(self, tmp_path, ext, converters):
        if shutil.which("latex") is None or not any(map(shutil.which, converters)):
            pytest.skip(f"Rendering .{ext} needs LaTeX with one of {converters}")
        model_spme = pybamm.lithium_ion.SPMe()
        filename = tmp_path / f"equations.{ext}"
        model_spme.latexify(str(filename))
        assert filename.stat().st_size > 0

    def test_latexify_with_lithium_plating(self):
        model = pybamm.lithium_ion.SPM(options={"lithium plating": "irreversible"})
        # Should not raise KeyError
        model.latexify()
