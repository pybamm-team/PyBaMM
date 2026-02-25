#
# Tests for ParameterValues JSON serialisation (to_json / from_json)
#

import json

import numpy as np
import pytest

import pybamm
from pybamm.expression_tree.operations.serialise import (
    convert_symbol_to_json,
)
from pybamm.parameters.parameter_values import (
    convert_parameter_values_to_json,
    convert_symbols_in_dict,
)


def _roundtrip(pv):
    """Serialize then deserialize a ParameterValues object."""
    return pybamm.ParameterValues.from_json(pv.to_json())


def _assert_evaluate_equal(pv_original, pv_loaded, name, inputs):
    """Assert that a FunctionParameter evaluates identically
    before and after roundtrip."""
    fp = pybamm.FunctionParameter(name, inputs)
    orig = pv_original.process_symbol(fp).evaluate()
    loaded = pv_loaded.process_symbol(fp).evaluate()
    assert not np.any(np.isnan(orig)), f"'{name}' original has NaN"
    np.testing.assert_array_equal(orig, loaded)


def _sv(start, n=5):
    """Create a StateVector spanning *n* elements starting at *start*."""
    return pybamm.StateVector(slice(start, start + n))


def _assert_evaluate_equal_array(pv_original, pv_loaded, name, inputs, y):
    """Like _assert_evaluate_equal but evaluates with a state vector
    array *y* so that the comparison covers vector-valued outputs."""
    fp = pybamm.FunctionParameter(name, inputs)
    orig = pv_original.process_symbol(fp).evaluate(y=y)
    loaded = pv_loaded.process_symbol(fp).evaluate(y=y)
    assert not np.any(np.isnan(orig)), f"'{name}' original has NaN"
    np.testing.assert_array_equal(orig, loaded)


def _assert_evaluate_close(pv_original, pv_loaded, name, inputs, y):
    """Like _assert_evaluate_equal_array but allows machine-epsilon
    differences.  Use for real parameter sets where the original
    callable and the reconstructed ExpressionFunctionParameter may
    follow slightly different floating-point evaluation orders."""
    fp = pybamm.FunctionParameter(name, inputs)
    orig = pv_original.process_symbol(fp).evaluate(y=y)
    loaded = pv_loaded.process_symbol(fp).evaluate(y=y)
    assert not np.any(np.isnan(orig)), f"'{name}' original has NaN"
    np.testing.assert_allclose(orig, loaded, rtol=1e-14)


# ------------------------------------------------------------------ #
# Unit tests: roundtrip for each value type
# ------------------------------------------------------------------ #
class TestRoundtripValueTypes:
    def test_empty_parameter_values(self):
        pv = pybamm.ParameterValues({})
        pv2 = _roundtrip(pv)
        assert len(pv2) == 0

    def test_numeric_int(self):
        pv = pybamm.ParameterValues({"a": 42})
        pv2 = _roundtrip(pv)
        assert pv2["a"] == 42

    def test_numeric_float(self):
        pv = pybamm.ParameterValues({"a": 3.14})
        pv2 = _roundtrip(pv)
        assert pv2["a"] == 3.14

    def test_numeric_negative(self):
        pv = pybamm.ParameterValues({"a": -42, "b": -0.001})
        pv2 = _roundtrip(pv)
        assert pv2["a"] == -42
        assert pv2["b"] == -0.001

    def test_numeric_extreme_values(self):
        pv = pybamm.ParameterValues({"tiny": 1e-300, "huge": 1e300, "zero": 0.0})
        pv2 = _roundtrip(pv)
        assert pv2["tiny"] == 1e-300
        assert pv2["huge"] == 1e300
        assert pv2["zero"] == 0.0

    def test_string_converted_to_float(self):
        pv = pybamm.ParameterValues({"a": "2.718"})
        pv2 = _roundtrip(pv)
        assert pv2["a"] == 2.718

    def test_input_parameter(self):
        pv = pybamm.ParameterValues({"a [m]": "[input]"})
        pv2 = _roundtrip(pv)
        assert isinstance(pv2["a [m]"], pybamm.InputParameter)

    def test_simple_callable_one_arg(self):
        def double(x):
            return 2 * x

        pv = pybamm.ParameterValues({"func": double})
        pv2 = _roundtrip(pv)

        x = pybamm.Scalar(5)
        _assert_evaluate_equal(pv, pv2, "func", {"x": x})

        y = np.array([[1], [2], [3], [4], [5]])
        _assert_evaluate_equal_array(pv, pv2, "func", {"x": _sv(0)}, y)

    def test_callable_two_args(self):
        def diffusivity(sto, T):
            return 3.9 * pybamm.exp(-sto) * pybamm.exp(-T / 300)

        pv = pybamm.ParameterValues({"D": diffusivity})
        pv2 = _roundtrip(pv)

        sto = pybamm.Scalar(0.5)
        T = pybamm.Scalar(300)
        _assert_evaluate_equal(pv, pv2, "D", {"sto": sto, "T": T})

        n = 5
        y = np.vstack(
            [
                np.linspace(0.2, 0.8, n).reshape(-1, 1),
                np.full((n, 1), 300),
            ]
        )
        _assert_evaluate_equal_array(
            pv, pv2, "D", {"sto": _sv(0, n), "T": _sv(n, n)}, y
        )

    def test_callable_with_keyword_args(self):
        def func_no_kwargs(x):
            return 2 * x

        def func_with_kwargs(x, y=1):
            return 2 * x

        x = pybamm.Scalar(2)
        fp = pybamm.FunctionParameter("func", {"x": x})

        for func in [func_no_kwargs, func_with_kwargs]:
            pv = pybamm.ParameterValues({"func": func})
            assert pv.evaluate(fp) == 4.0

            pv2 = _roundtrip(pv)
            assert pv2.evaluate(fp) == 4.0

    def test_callable_returning_interpolant_1d(self):
        x_data = np.linspace(0, 1, 50)
        y_data = 2 * x_data**2

        def ocp(sto):
            return pybamm.Interpolant(x_data, y_data, sto, name="ocp")

        pv = pybamm.ParameterValues({"OCP [V]": ocp})
        pv2 = _roundtrip(pv)

        sto = pybamm.Scalar(0.5)
        _assert_evaluate_equal(pv, pv2, "OCP [V]", {"sto": sto})

        y = np.linspace(0.1, 0.9, 5).reshape(-1, 1)
        _assert_evaluate_equal_array(pv, pv2, "OCP [V]", {"sto": _sv(0)}, y)

    def test_callable_returning_interpolant_2d(self):
        x1 = np.linspace(0, 1, 10)
        x2 = np.linspace(280, 320, 10)
        X1, X2 = np.meshgrid(x1, x2, indexing="ij")
        y_data = X1 * X2

        def diff(c_s, T):
            return pybamm.Interpolant(
                (x1, x2),
                y_data,
                [c_s, T],
                name="diff_2d",
            )

        pv = pybamm.ParameterValues({"D": diff})
        pv2 = _roundtrip(pv)

        c_s = pybamm.Scalar(0.5)
        T = pybamm.Scalar(300)
        _assert_evaluate_equal(pv, pv2, "D", {"c_s": c_s, "T": T})

        n = 5
        y = np.vstack(
            [
                np.linspace(0.2, 0.8, n).reshape(-1, 1),
                np.linspace(285, 315, n).reshape(-1, 1),
            ]
        )
        _assert_evaluate_equal_array(
            pv,
            pv2,
            "D",
            {"c_s": _sv(0, n), "T": _sv(n, n)},
            y,
        )

    def test_callable_using_parameter_internally(self):
        def volume_change(sto):
            omega = pybamm.Parameter("partial molar volume [m3.mol-1]")
            c_max = pybamm.Parameter("max conc [mol.m-3]")
            return omega * c_max * sto

        pv = pybamm.ParameterValues(
            {
                "vol change": volume_change,
                "partial molar volume [m3.mol-1]": 1e-6,
                "max conc [mol.m-3]": 50000,
            }
        )
        pv2 = _roundtrip(pv)

        sto = pybamm.Scalar(0.5)
        _assert_evaluate_equal(pv, pv2, "vol change", {"sto": sto})

    def test_mixed_parameter_dict(self):
        def my_func(x):
            return 3 * x

        pv = pybamm.ParameterValues(
            {
                "scalar_int": 1,
                "scalar_float": 2.5,
                "callable": my_func,
                "input_param [A]": "[input]",
            }
        )
        pv2 = _roundtrip(pv)

        assert pv2["scalar_int"] == 1
        assert pv2["scalar_float"] == 2.5
        assert isinstance(pv2["input_param [A]"], pybamm.InputParameter)

        x = pybamm.Scalar(4)
        _assert_evaluate_equal(pv, pv2, "callable", {"x": x})

    def test_expression_scalar_plus_parameter(self):
        expr = 1 + pybamm.Parameter("k")
        pv = pybamm.ParameterValues({"expr": expr, "k": 5})
        pv2 = _roundtrip(pv)
        assert pv2.evaluate(pv2["expr"]) == 6

    def test_expression_arithmetic_tree(self):
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        expr = (2 * a + b) / (a - b + 1)
        pv = pybamm.ParameterValues({"expr": expr, "a": 10, "b": 3})
        pv2 = _roundtrip(pv)
        expected = (2 * 10 + 3) / (10 - 3 + 1)
        assert pv2.evaluate(pv2["expr"]) == expected

    def test_expression_with_exp_and_log(self):
        x = pybamm.Parameter("x")
        expr = pybamm.exp(x) + pybamm.log(x + 1)
        pv = pybamm.ParameterValues({"expr": expr, "x": 2})
        pv2 = _roundtrip(pv)
        expected = np.exp(2) + np.log(3)
        assert pv2.evaluate(pv2["expr"]) == expected

    def test_expression_with_min_max(self):
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        expr = pybamm.minimum(a, b) + pybamm.maximum(a, b)
        pv = pybamm.ParameterValues({"expr": expr, "a": 3, "b": 7})
        pv2 = _roundtrip(pv)
        assert pv2.evaluate(pv2["expr"]) == 10

    def test_callable_returning_expression(self):
        def my_func(x):
            return pybamm.exp(-x) + pybamm.Parameter("offset")

        pv = pybamm.ParameterValues({"f": my_func, "offset": 0.5})
        pv2 = _roundtrip(pv)
        x = pybamm.Scalar(1)
        _assert_evaluate_equal(pv, pv2, "f", {"x": x})

    def test_scalar_object_as_value(self):
        pv = pybamm.ParameterValues({"a": pybamm.Scalar(7.5)})
        pv2 = _roundtrip(pv)
        assert isinstance(pv2["a"], pybamm.Scalar)
        assert pv2["a"].evaluate() == 7.5

    def test_callable_returning_constant(self):
        def const(x):
            return pybamm.Scalar(42)

        pv = pybamm.ParameterValues({"f": const})
        pv2 = _roundtrip(pv)
        _assert_evaluate_equal(pv, pv2, "f", {"x": pybamm.Scalar(999)})

    def test_expression_with_time(self):
        expr = 1 + pybamm.t
        pv = pybamm.ParameterValues({"expr": expr})
        pv2 = _roundtrip(pv)
        result = pv2["expr"].evaluate(t=5)
        assert result == 6

    def test_expression_with_trig_and_abs(self):
        x = pybamm.Parameter("x")
        expr = pybamm.sin(x) + pybamm.cos(x) + abs(x)
        pv = pybamm.ParameterValues({"expr": expr, "x": 1.0})
        pv2 = _roundtrip(pv)
        expected = np.sin(1) + np.cos(1) + np.abs(1)
        assert pv2.evaluate(pv2["expr"]) == expected

    def test_double_roundtrip(self):
        def my_func(x):
            return 2 * x + pybamm.Parameter("b")

        pv = pybamm.ParameterValues({"f": my_func, "b": 10})
        pv2 = _roundtrip(_roundtrip(pv))

        x = pybamm.Scalar(3)
        _assert_evaluate_equal(pv, pv2, "f", {"x": x})
        assert pv2["b"] == 10


# ------------------------------------------------------------------ #
# Integration tests: nested / composite structures
# ------------------------------------------------------------------ #
class TestRoundtripNestedStructures:
    def test_function_calling_another_function(self):
        def inner(x):
            return x**2

        def outer(x):
            return 3 * inner(x) + 1

        pv = pybamm.ParameterValues({"f": outer})
        pv2 = _roundtrip(pv)

        x = pybamm.Scalar(4)
        _assert_evaluate_equal(pv, pv2, "f", {"x": x})

    def test_deeply_nested_expression(self):
        """5+ levels of nesting to stress recursive serialization."""

        def f(x):
            return pybamm.exp(
                pybamm.sin(pybamm.cos(pybamm.log(1 + abs(x))))
            ) + pybamm.Parameter("offset")

        pv = pybamm.ParameterValues({"f": f, "offset": 0.1})
        pv2 = _roundtrip(pv)

        for val in [0.5, 1.0, 2.0]:
            _assert_evaluate_equal(pv, pv2, "f", {"x": pybamm.Scalar(val)})

    def test_interpolant_inside_function_expression(self):
        x_data = np.linspace(0, 1, 50)
        y_data = np.sin(x_data)

        def scaled_ocp(sto):
            return (
                2.0 * pybamm.Interpolant(x_data, y_data, sto, name="sin_interp") + 0.5
            )

        pv = pybamm.ParameterValues({"OCP [V]": scaled_ocp})
        pv2 = _roundtrip(pv)

        sto = pybamm.Scalar(0.3)
        _assert_evaluate_equal(pv, pv2, "OCP [V]", {"sto": sto})

    def test_callable_with_expression_and_parameter(self):
        """Function that builds an expression tree using Parameter
        and arithmetic, nested inside a callable parameter value."""

        def reaction_rate(T):
            Ea = pybamm.Parameter("Ea [J.mol-1]")
            k0 = pybamm.Parameter("k0 [s-1]")
            R = 8.314
            return k0 * pybamm.exp(-Ea / (R * T))

        pv = pybamm.ParameterValues(
            {
                "rate": reaction_rate,
                "Ea [J.mol-1]": 50000,
                "k0 [s-1]": 1e6,
            }
        )
        pv2 = _roundtrip(pv)

        T = pybamm.Scalar(300)
        _assert_evaluate_equal(pv, pv2, "rate", {"T": T})

        y = np.linspace(280, 320, 5).reshape(-1, 1)
        _assert_evaluate_equal_array(pv, pv2, "rate", {"T": _sv(0)}, y)

    def test_interpolant_scaled_by_parameter(self):
        """Interpolant output multiplied by a Parameter inside a
        callable — tests nested interpolant + expression."""
        x_data = np.linspace(0, 1, 50)
        y_data = np.sin(np.pi * x_data)

        def ocp(sto):
            scale = pybamm.Parameter("scale [V]")
            return scale * pybamm.Interpolant(x_data, y_data, sto, name="sin_pi")

        pv = pybamm.ParameterValues({"OCP [V]": ocp, "scale [V]": 3.7})
        pv2 = _roundtrip(pv)

        _assert_evaluate_equal(pv, pv2, "OCP [V]", {"sto": pybamm.Scalar(0.5)})
        y = np.linspace(0.1, 0.9, 5).reshape(-1, 1)
        _assert_evaluate_equal_array(pv, pv2, "OCP [V]", {"sto": _sv(0)}, y)

    def test_expression_combining_two_interpolants(self):
        """Callable that adds two different interpolants plus a
        constant offset."""
        x_data = np.linspace(0, 1, 50)
        y1 = np.cos(x_data)
        y2 = x_data**2

        def combined(sto):
            interp1 = pybamm.Interpolant(x_data, y1, sto, name="cos")
            interp2 = pybamm.Interpolant(x_data, y2, sto, name="sq")
            return interp1 + 2 * interp2 - 0.1

        pv = pybamm.ParameterValues({"f": combined})
        pv2 = _roundtrip(pv)

        _assert_evaluate_equal(pv, pv2, "f", {"sto": pybamm.Scalar(0.4)})
        y = np.linspace(0.1, 0.9, 5).reshape(-1, 1)
        _assert_evaluate_equal_array(pv, pv2, "f", {"sto": _sv(0)}, y)

    def test_callable_with_conditional_expression(self):
        """Callable using pybamm.minimum / maximum to build a
        piecewise-like expression."""

        def clipped_func(x):
            return pybamm.maximum(
                pybamm.minimum(3 * x, pybamm.Scalar(2)), pybamm.Scalar(0)
            )

        pv = pybamm.ParameterValues({"f": clipped_func})
        pv2 = _roundtrip(pv)

        for val in [-1, 0.3, 0.5, 1, 2]:
            _assert_evaluate_equal(pv, pv2, "f", {"x": pybamm.Scalar(val)})

    def test_nested_callable_parameter_expression_mix(self):
        """Several parameter values that cross-reference each other
        through expressions and callables."""

        def rate(T):
            Ea = pybamm.Parameter("Ea")
            return pybamm.exp(-Ea / T)

        def scaled_rate(T):
            k0 = pybamm.Parameter("k0")
            return k0 * rate(T)

        pv = pybamm.ParameterValues({"f": scaled_rate, "Ea": 5000, "k0": 1e4})
        pv2 = _roundtrip(pv)

        T = pybamm.Scalar(300)
        _assert_evaluate_equal(pv, pv2, "f", {"T": T})

        y = np.linspace(280, 320, 5).reshape(-1, 1)
        _assert_evaluate_equal_array(pv, pv2, "f", {"T": _sv(0)}, y)

    def test_full_parameter_set_roundtrip_ai2020(self):
        """Ai2020 has interpolant-based OCPs, functions using
        pybamm.Parameter internally, and multi-arg callables."""
        pv = pybamm.ParameterValues("Ai2020")
        pv2 = _roundtrip(pv)

        assert set(pv.keys()) == set(pv2.keys())

        # Numeric values are preserved exactly
        for key in pv.keys():
            orig = pv[key]
            loaded = pv2[key]
            if isinstance(orig, int | float):
                assert loaded == orig, f"Numeric mismatch for '{key}'"

        # Callable parameters evaluate to the same result after
        # roundtrip — tested with arrays via StateVector to exercise
        # vector-valued paths through interpolants etc.
        n = 5
        sto_vals = np.linspace(0.2, 0.8, n).reshape(-1, 1)
        T_vals = np.full((n, 1), 300.0)
        c_e_vals = np.linspace(500, 1500, n).reshape(-1, 1)

        sto_sv = _sv(0, n)
        T_sv = _sv(n, n)
        c_e_sv = _sv(0, n)

        # 1-arg interpolant OCPs (sto only)
        y_sto = sto_vals
        for ocp_name in [
            "Negative electrode OCP [V]",
            "Positive electrode OCP [V]",
        ]:
            _assert_evaluate_close(pv, pv2, ocp_name, {"sto": _sv(0, n)}, y_sto)

        # 2-arg diffusivities (sto, T)
        y_sto_T = np.vstack([sto_vals, T_vals])
        for diff_name in [
            "Negative electrode diffusivity [m2.s-1]",
            "Positive electrode diffusivity [m2.s-1]",
        ]:
            _assert_evaluate_close(
                pv,
                pv2,
                diff_name,
                {"sto": sto_sv, "T": T_sv},
                y_sto_T,
            )

        # 4-arg exchange-current density
        c_s_surf_vals = np.full((n, 1), 20000.0)
        c_s_max_vals = np.full((n, 1), 33133.0)
        y_j0 = np.vstack(
            [
                c_e_vals,
                c_s_surf_vals,
                c_s_max_vals,
                T_vals,
            ]
        )
        _assert_evaluate_close(
            pv,
            pv2,
            "Negative electrode exchange-current density [A.m-2]",
            {
                "c_e": _sv(0, n),
                "c_s_surf": _sv(n, n),
                "c_s_max": _sv(2 * n, n),
                "T": _sv(3 * n, n),
            },
            y_j0,
        )

        # 2-arg electrolyte properties (c_e, T)
        y_ce_T = np.vstack([c_e_vals, T_vals])
        for elyte_name in [
            "Electrolyte diffusivity [m2.s-1]",
            "Electrolyte conductivity [S.m-1]",
        ]:
            _assert_evaluate_close(
                pv,
                pv2,
                elyte_name,
                {"c_e": c_e_sv, "T": T_sv},
                y_ce_T,
            )

    def test_full_parameter_set_roundtrip_chen2020_soh(self):
        """Chen2020 roundtrip verified through ElectrodeSOHSolver.
        Compares solver outputs before and after serialisation."""
        pv = pybamm.ParameterValues("Chen2020")
        param = pybamm.LithiumIonParameters()

        Q_n = pv.evaluate(param.n.Q_init)
        Q_p = pv.evaluate(param.p.Q_init)
        Q_Li = pv.evaluate(param.Q_Li_particles_init)
        inputs = {"Q_Li": Q_Li, "Q_n": Q_n, "Q_p": Q_p}

        sol_orig = pybamm.lithium_ion.ElectrodeSOHSolver(
            pv, direction=None, param=param
        ).solve(inputs)

        pv2 = _roundtrip(pv)
        sol_rt = pybamm.lithium_ion.ElectrodeSOHSolver(
            pv2, direction=None, param=param
        ).solve(inputs)

        for key in sol_orig:
            np.testing.assert_allclose(
                sol_rt[key],
                sol_orig[key],
                rtol=1e-14,
                err_msg=f"SOH mismatch for '{key}'",
            )


# ------------------------------------------------------------------ #
# File I/O tests
# ------------------------------------------------------------------ #
class TestFileIO:
    def test_roundtrip_via_string_path(self, tmp_path):
        pv = pybamm.ParameterValues({"a": 42, "b": 3.14})
        filepath = str(tmp_path / "params.json")
        pv.to_json(filepath)
        pv2 = pybamm.ParameterValues.from_json(filepath)
        assert pv2["a"] == 42
        assert pv2["b"] == 3.14

    def test_roundtrip_via_pathlib_path(self, tmp_path):
        pv = pybamm.ParameterValues({"a": 100, "b": 2.71})
        filepath = tmp_path / "params.json"
        pv.to_json(str(filepath))
        pv2 = pybamm.ParameterValues.from_json(filepath)
        assert pv2["a"] == 100
        assert pv2["b"] == 2.71

    def test_roundtrip_callable_via_file(self, tmp_path):
        def my_func(x):
            return 5 * x

        pv = pybamm.ParameterValues({"f": my_func})
        filepath = str(tmp_path / "params.json")
        pv.to_json(filepath)
        pv2 = pybamm.ParameterValues.from_json(filepath)

        x = pybamm.Scalar(3)
        _assert_evaluate_equal(pv, pv2, "f", {"x": x})

    def test_from_json_invalid_input_type(self):
        with pytest.raises(TypeError, match=r"Input must be a filename.*or a dict"):
            pybamm.ParameterValues.from_json(123)

        with pytest.raises(TypeError, match=r"Input must be a filename.*or a dict"):
            pybamm.ParameterValues.from_json([1, 2, 3])


# ------------------------------------------------------------------ #
# Helper function tests
# ------------------------------------------------------------------ #
class TestConvertSymbolsInDict:
    def test_interpolator_dict(self):
        data_dict = {
            "p": {
                "interpolator": "linear",
                "x": np.array([0, 1, 2]),
                "y": np.array([0, 1, 4]),
            },
        }
        result = convert_symbols_in_dict(data_dict)
        assert callable(result["p"])
        out = result["p"](pybamm.Scalar(1.5))
        assert isinstance(out, pybamm.Interpolant | pybamm.Scalar)

    def test_nested_dict_with_serialized_symbol(self):
        scalar = pybamm.Scalar(2.718)
        serialized = convert_symbol_to_json(scalar)
        result = convert_symbols_in_dict({"p": serialized})
        assert isinstance(result["p"], pybamm.Scalar)
        assert result["p"].value == 2.718

    def test_list_with_serialized_symbol(self):
        scalar = pybamm.Scalar(2.718)
        serialized = convert_symbol_to_json(scalar)
        result = convert_symbols_in_dict({"p": [serialized, 42]})
        assert isinstance(result["p"][0], pybamm.Scalar)
        assert result["p"][1] == 42

    def test_string_converted_to_float(self):
        result = convert_symbols_in_dict({"p": "3.14"})
        assert result["p"] == 3.14

    def test_none_returns_empty_dict(self):
        assert convert_symbols_in_dict(None) == {}


class TestConvertParameterValuesToJson:
    def test_with_callable(self, tmp_path):
        def my_function(x):
            return x * 2

        pv = pybamm.ParameterValues({"p1": 42, "p2": my_function})

        result = convert_parameter_values_to_json(pv)
        assert "p1" in result
        assert "p2" in result

        filepath = str(tmp_path / "params.json")
        convert_parameter_values_to_json(pv, filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert "p1" in data
        assert "p2" in data
