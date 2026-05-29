"""Tests for the ParameterSubstitutor class."""

import pytest

import pybamm
from pybamm.parameters.parameter_store import ParameterStore
from pybamm.parameters.parameter_substitutor import ParameterSubstitutor


class TestParameterSubstitutor:
    """Test the ParameterSubstitutor class."""

    def test_init(self):
        store = ParameterStore({"a": 1})
        processor = ParameterSubstitutor(store)
        assert processor._store is store
        assert processor._cache == {}

    def test_clear_cache(self):
        store = ParameterStore({"a": 1})
        processor = ParameterSubstitutor(store)
        # Process a symbol to populate cache
        param = pybamm.Parameter("a")
        processor.process_symbol(param)
        assert len(processor._cache) > 0
        # Clear and verify
        processor.clear_cache()
        assert processor._cache == {}

    def test_cache_property(self):
        store = ParameterStore({"a": 1})
        processor = ParameterSubstitutor(store)
        assert processor.cache == {}


class TestProcessSymbol:
    """Test the process_symbol method."""

    def test_process_parameter(self):
        store = ParameterStore({"My param": 42})
        processor = ParameterSubstitutor(store)
        param = pybamm.Parameter("My param")
        result = processor.process_symbol(param)
        assert isinstance(result, pybamm.Scalar)
        assert result.evaluate() == 42

    def test_process_parameter_caching(self):
        store = ParameterStore({"My param": 42})
        processor = ParameterSubstitutor(store)
        param = pybamm.Parameter("My param")
        result1 = processor.process_symbol(param)
        result2 = processor.process_symbol(param)
        assert result1 is result2  # Should be same object from cache

    def test_process_input_parameter(self):
        store = ParameterStore({"My param": pybamm.InputParameter("My param")})
        processor = ParameterSubstitutor(store)
        param = pybamm.Parameter("My param")
        result = processor.process_symbol(param)
        assert isinstance(result, pybamm.InputParameter)

    def test_process_scalar(self):
        store = ParameterStore({})
        processor = ParameterSubstitutor(store)
        scalar = pybamm.Scalar(42)
        result = processor.process_symbol(scalar)
        assert result is scalar  # Scalars pass through unchanged

    def test_process_number(self):
        store = ParameterStore({})
        processor = ParameterSubstitutor(store)
        result = processor.process_symbol(3.14)
        assert isinstance(result, pybamm.Scalar)
        assert result.evaluate() == 3.14

    def test_process_addition(self):
        store = ParameterStore({"a": 1, "b": 2})
        processor = ParameterSubstitutor(store)
        expr = pybamm.Parameter("a") + pybamm.Parameter("b")
        result = processor.process_symbol(expr)
        assert result.evaluate() == 3

    def test_process_multiplication(self):
        store = ParameterStore({"a": 3, "b": 4})
        processor = ParameterSubstitutor(store)
        expr = pybamm.Parameter("a") * pybamm.Parameter("b")
        result = processor.process_symbol(expr)
        assert result.evaluate() == 12

    def test_process_function_parameter_scalar(self):
        store = ParameterStore({"My func": 42})
        processor = ParameterSubstitutor(store)
        x = pybamm.Scalar(1)
        func_param = pybamm.FunctionParameter("My func", {"x": x})
        result = processor.process_symbol(func_param)
        # Function parameters with scalar values may return Scalar or multiplied result
        assert result.evaluate() == 42

    def test_process_function_parameter_callable(self):
        def my_func(x):
            return 2 * x

        store = ParameterStore({"My func": my_func})
        processor = ParameterSubstitutor(store)
        x = pybamm.Scalar(5)
        func_param = pybamm.FunctionParameter("My func", {"x": x})
        result = processor.process_symbol(func_param)
        assert result.evaluate() == 10

    def test_process_missing_parameter(self):
        store = ParameterStore({"a": 1})
        processor = ParameterSubstitutor(store)
        param = pybamm.Parameter("nonexistent")
        with pytest.raises(KeyError, match="not found"):
            processor.process_symbol(param)


class TestEvaluate:
    """Test the evaluate method."""

    def test_evaluate_constant(self):
        store = ParameterStore({"a": 42})
        processor = ParameterSubstitutor(store)
        param = pybamm.Parameter("a")
        result = processor.evaluate(param)
        assert result == 42

    def test_evaluate_expression(self):
        store = ParameterStore({"a": 3, "b": 4})
        processor = ParameterSubstitutor(store)
        expr = pybamm.Parameter("a") + pybamm.Parameter("b")
        result = processor.evaluate(expr)
        assert result == 7

    def test_evaluate_with_input(self):
        store = ParameterStore({"a": pybamm.InputParameter("a")})
        processor = ParameterSubstitutor(store)
        param = pybamm.Parameter("a")
        result = processor.evaluate(param, inputs={"a": 99})
        assert result == 99


class TestProcessGeometry:
    """Test the process_geometry method."""

    def test_process_geometry_basic(self):
        store = ParameterStore({"L_n": 0.1, "L_s": 0.2})
        processor = ParameterSubstitutor(store)

        x_n = pybamm.SpatialVariable("x_n", domain="negative electrode")
        geometry = {
            "negative electrode": {
                x_n: {
                    "min": pybamm.Scalar(0),
                    "max": pybamm.Parameter("L_n"),
                }
            }
        }

        processor.process_geometry(geometry)

        # Check that the parameter was replaced
        assert isinstance(geometry["negative electrode"][x_n]["max"], pybamm.Scalar)
        assert geometry["negative electrode"][x_n]["max"].evaluate() == 0.1


class TestNamedFunctionParameter:
    """Bind a value to a FunctionParameter by input name, not position."""

    NAME = "Electrolyte conductivity [S.m-1]"
    CONC = "Electrolyte concentration [mol.m-3]"
    TEMP = "Temperature [K]"

    def _function_parameter(self, input_order):
        children = {self.CONC: pybamm.Scalar(1000.0), self.TEMP: pybamm.Scalar(298.0)}
        inputs = {name: children[name] for name in input_order}
        return pybamm.FunctionParameter(self.NAME, inputs)

    def _evaluate(self, value, input_order):
        fp = self._function_parameter(input_order)
        return pybamm.ParameterValues({self.NAME: value}).process_symbol(fp).evaluate()

    def test_binds_named_input_ignores_others(self):
        named = pybamm.NamedFunctionParameter(lambda c: c * 2, inputs=[self.CONC])
        assert self._evaluate(named, [self.CONC, self.TEMP]) == 2000.0

    def test_order_independent(self):
        named = pybamm.NamedFunctionParameter(lambda c: c * 2, inputs=[self.CONC])
        # model declares inputs in the opposite order -> still binds to c_e
        assert self._evaluate(named, [self.TEMP, self.CONC]) == 2000.0

    def test_bare_lambda_is_positional(self):
        # contrast: a bare positional lambda binds to whatever is declared first
        bare = lambda c_e, T: c_e * 2  # noqa: E731
        assert self._evaluate(bare, [self.CONC, self.TEMP]) == 2000.0
        assert self._evaluate(bare, [self.TEMP, self.CONC]) == 596.0

    def test_json_roundtrip_preserves_named_binding(self):
        from pybamm.expression_tree.operations.serialise import (
            convert_symbol_from_json,
        )

        named = pybamm.NamedFunctionParameter(lambda c: c * 2, inputs=[self.CONC])
        pv_json = pybamm.ParameterValues({self.NAME: named}).to_json()
        reconstructed = convert_symbol_from_json(pv_json[self.NAME])
        assert self._evaluate(reconstructed, [self.CONC, self.TEMP]) == 2000.0
        assert self._evaluate(reconstructed, [self.TEMP, self.CONC]) == 2000.0

    def test_missing_input_raises(self):
        named = pybamm.NamedFunctionParameter(lambda c: c, inputs=["Not an input"])
        with pytest.raises(KeyError, match="not found among"):
            self._evaluate(named, [self.CONC, self.TEMP])

    def test_inputs_length_mismatch_raises(self):
        named = pybamm.NamedFunctionParameter(lambda c_e, T: c_e, inputs=[self.CONC])
        with pytest.raises(ValueError, match="length does not match"):
            pybamm.ParameterValues({self.NAME: named}).to_json()

    def test_non_callable_function_raises(self):
        with pytest.raises(TypeError, match="must be callable"):
            pybamm.NamedFunctionParameter(42, inputs=[self.CONC])
