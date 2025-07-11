import json
import os
from datetime import datetime

import numpy as np
import pytest

import pybamm
from pybamm.expression_tree.operations.serialise import Serialise
from pybamm.models.full_battery_models.lithium_ion.basic_dfn import BasicDFN
from pybamm.models.full_battery_models.lithium_ion.basic_spm import BasicSPM


def test_serialise_scalar():
    S = pybamm.Scalar(2.718)
    j = Serialise.convert_symbol_to_json(S)
    S2 = Serialise.convert_symbol_from_json(j)
    assert isinstance(S2, pybamm.Scalar)
    assert S2.value == pytest.approx(2.718)


def test_serialise_time():
    t = pybamm.Time()
    j = Serialise.convert_symbol_to_json(t)
    t2 = Serialise.convert_symbol_from_json(j)
    assert isinstance(t2, pybamm.Time)


def test_primary_broadcast_serialisation():
    child = pybamm.Scalar(42)
    symbol = pybamm.PrimaryBroadcast(child, "negative electrode")
    json_dict = Serialise.convert_symbol_to_json(symbol)
    symbol2 = Serialise.convert_symbol_from_json(json_dict)

    assert isinstance(symbol2, pybamm.PrimaryBroadcast)
    assert symbol2.broadcast_domain == ["negative electrode"]
    assert isinstance(symbol2.orphans[0], pybamm.Scalar)
    assert symbol2.orphans[0].value == 42


def test_interpolant_serialisation():
    x = np.linspace(0, 1, 5)
    y = np.array([0, 1, 4, 9, 16])
    child = pybamm.Variable("z")
    interp = pybamm.Interpolant(
        x, y, child, name="test_interplot", interpolator="linear"
    )
    json_dict = Serialise.convert_symbol_to_json(interp)
    interp2 = Serialise.convert_symbol_from_json(json_dict)

    assert isinstance(interp2, pybamm.Interpolant)
    assert interp2.name == "test_interplot"
    assert interp2.interpolator == "linear"
    assert isinstance(interp2.x[0], np.ndarray)
    assert isinstance(interp2.y, np.ndarray)
    assert interp2.children[0].name == "z"


def test_variable_serialisation():
    var = pybamm.Variable("var", domain="separator")
    json_dict = Serialise.convert_symbol_to_json(var)
    var2 = Serialise.convert_symbol_from_json(json_dict)

    assert isinstance(var2, pybamm.Variable)
    assert var2.name == "var"
    assert var2.domains["primary"] == ["separator"]
    assert var2.bounds[0].value == -float("inf")
    assert var2.bounds[1].value == float("inf")


def test_concatenation_variable_serialisation():
    var1 = pybamm.Variable("a", domain="negative electrode")
    var2 = pybamm.Variable("a", domain="separator")
    var3 = pybamm.Variable("a", domain="positive electrode")
    concat_var = pybamm.ConcatenationVariable(var1, var2, var3, name="conc_var")
    json_dict = Serialise.convert_symbol_to_json(concat_var)
    concat_var2 = Serialise.convert_symbol_from_json(json_dict)

    assert isinstance(concat_var2, pybamm.ConcatenationVariable)
    assert concat_var2.name == "a"
    assert len(concat_var2.children) == 3
    domains = [child.domains["primary"] for child in concat_var2.children]
    assert domains == [["negative electrode"], ["separator"], ["positive electrode"]]


def test_full_broadcast_serialisation():
    child = pybamm.Scalar(5)
    fb = pybamm.FullBroadcast(child, broadcast_domain="negative electrode")
    json_dict = Serialise.convert_symbol_to_json(fb)
    fb2 = Serialise.convert_symbol_from_json(json_dict)

    assert isinstance(fb2, pybamm.FullBroadcast)
    assert fb2.broadcast_domain == ["negative electrode"]
    assert isinstance(fb2.child, pybamm.Scalar)
    assert fb2.child.value == 5


def test_spatial_variable_serialisation():
    sv = pybamm.SpatialVariable("x", domain="negative electrode", coord_sys="cartesian")
    json_dict = Serialise.convert_symbol_to_json(sv)
    sv2 = Serialise.convert_symbol_from_json(json_dict)

    assert isinstance(sv2, pybamm.SpatialVariable)
    assert sv2.name == "x"
    assert sv2.domains["primary"] == ["negative electrode"]
    assert sv2.coord_sys == "cartesian"


def test_boundary_value_serialisation():
    var = pybamm.SpatialVariable("x", domain="electrode")
    bv = pybamm.BoundaryValue(var, "left")
    json_dict = Serialise.convert_symbol_to_json(bv)
    bv2 = Serialise.convert_symbol_from_json(json_dict)

    assert isinstance(bv2, pybamm.BoundaryValue)
    assert bv2.side == "left"
    assert isinstance(bv2.orphans[0], pybamm.SpatialVariable)
    assert bv2.orphans[0].name == "x"


def test_specific_function_not_supported():
    def dummy_func(x):
        return x

    symbol = pybamm.SpecificFunction(dummy_func, pybamm.Scalar(1))
    with pytest.raises(
        NotImplementedError, match="SpecificFunction is not supported directly"
    ):
        Serialise.convert_symbol_to_json(symbol)


def test_unary_operator_serialisation():
    expr = pybamm.Negate(pybamm.Scalar(5))
    json_dict = Serialise.convert_symbol_to_json(expr)
    expr2 = Serialise.convert_symbol_from_json(json_dict)

    assert isinstance(expr2, pybamm.Negate)
    assert isinstance(expr2.child, pybamm.Scalar)
    assert expr2.child.value == 5


def test_binary_operator_serialisation():
    expr = pybamm.Addition(pybamm.Scalar(2), pybamm.Scalar(3))
    json_dict = Serialise.convert_symbol_to_json(expr)
    expr2 = Serialise.convert_symbol_from_json(json_dict)

    assert isinstance(expr2, pybamm.Addition)
    values = [c.value for c in expr2.children]
    assert values == [2, 3]


def test_symbol_fallback_serialisation():
    var = pybamm.Variable("v", domain="electrode")
    diff = pybamm.Gradient(var)
    json_dict = Serialise.convert_symbol_to_json(diff)
    diff2 = Serialise.convert_symbol_from_json(json_dict)

    assert isinstance(diff2, pybamm.Gradient)
    assert isinstance(diff2.children[0], pybamm.Variable)
    assert diff2.children[0].name == "v"
    assert diff2.children[0].domains["primary"] == ["electrode"]


def test_unhandled_symbol_type_error():
    class NotSymbol:
        def __init__(self):
            self.name = "not_a_symbol"

    dummy = NotSymbol()
    with pytest.raises(ValueError) as e:
        Serialise.convert_symbol_to_json(dummy)

    assert "Error processing 'not_a_symbol'. Unknown symbol type:" in str(e.value)


def test_save_and_load_custom_model():
    model = pybamm.BaseModel(name="test_model")
    a = pybamm.Variable("a", domain="electrode")
    b = pybamm.Variable("b", domain="electrode")
    model.rhs = {a: b}
    model.initial_conditions = {a: pybamm.Scalar(1)}
    model.algebraic = {}
    model.boundary_conditions = {a: {"left": (pybamm.Scalar(0), "Dirichlet")}}
    model.events = [pybamm.Event("terminal", pybamm.Scalar(1) - b, "TERMINATION")]
    model.variables = {"a": a, "b": b}

    param_values = pybamm.ParameterValues({"param1": pybamm.Scalar(5)})

    # save model
    Serialise.save_custom_model(model, filename="test_model")

    # check json exists
    assert os.path.exists("test_model.json")

    # saving with defualt filename
    Serialise().save_custom_model(model)
    filename = "test_model_" + datetime.now().strftime("%Y_%m_%d-%p%I_%M") + ".json"
    assert os.path.exists(filename)
    os.remove(filename)

    # load model
    loaded_model = Serialise.load_custom_model("test_model.json")
    os.remove("test_model.json")

    assert loaded_model.name == "test_model"
    assert isinstance(loaded_model.rhs, dict)
    assert next(iter(loaded_model.rhs.keys())).name == "a"
    assert next(iter(loaded_model.rhs.values())).name == "b"


def test_plotting_serialised_models():
    models = [BasicSPM(), BasicDFN()]
    filenames = ["spm", "dfn"]

    for model, name in zip(models, filenames, strict=False):
        # Save the model
        Serialise.save_custom_model(model, filename=name)

        # Load the model
        loaded_model= Serialise.load_custom_model(
            f"{name}.json", battery_model=pybamm.lithium_ion.BaseModel()
        )

        # Simulate
        sim = pybamm.Simulation(loaded_model)
        sim.solve([0, 3600])
        sim.plot(show_plot=False)

        os.remove(f"{name}.json")
