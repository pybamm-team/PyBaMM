#
# Tests for the serialisation class
#

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pytest
from numpy import testing

import pybamm
from pybamm.expression_tree.operations.serialise import (
    SUPPORTED_SCHEMA_VERSION,
    ExpressionFunctionParameter,
    Serialise,
    convert_symbol_from_json,
    convert_symbol_to_json,
)
from pybamm.models.full_battery_models.lithium_ion.basic_dfn import BasicDFN
from pybamm.models.full_battery_models.lithium_ion.basic_spm import BasicSPM


def scalar_var_dict(mocker):
    """variable, json pair for a pybamm.Scalar instance"""
    a = pybamm.Scalar(5)
    a_dict = {
        "py/id": mocker.ANY,
        "py/object": "pybamm.expression_tree.scalar.Scalar",
        "name": "5.0",
        "id": mocker.ANY,
        "value": 5.0,
        "children": [],
    }

    return a, a_dict


def mesh_var_dict(mocker):
    """mesh, json pair for a pybamm.Mesh instance"""

    r = pybamm.SpatialVariable(
        "r", domain=["negative particle"], coord_sys="spherical polar"
    )

    geometry = {
        "negative particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
    }

    submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
    var_pts = {r: 20}

    # create mesh
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

    mesh_json = {
        "py/object": "pybamm.meshes.meshes.Mesh",
        "py/id": mocker.ANY,
        "submesh_pts": {"negative particle": {"r": 20}},
        "base_domains": ["negative particle"],
        "sub_meshes": {
            "negative particle": {
                "py/object": "pybamm.meshes.one_dimensional_submeshes.Uniform1DSubMesh",
                "py/id": mocker.ANY,
                "edges": [
                    0.0,
                    0.05,
                    0.1,
                    0.15000000000000002,
                    0.2,
                    0.25,
                    0.30000000000000004,
                    0.35000000000000003,
                    0.4,
                    0.45,
                    0.5,
                    0.55,
                    0.6000000000000001,
                    0.65,
                    0.7000000000000001,
                    0.75,
                    0.8,
                    0.8500000000000001,
                    0.9,
                    0.9500000000000001,
                    1.0,
                ],
                "coord_sys": "spherical polar",
            }
        },
    }

    return mesh, mesh_json


class TestSerialiseModels:
    def test_user_defined_model_recreaction(self):
        # Start with a base model
        model = pybamm.BaseModel()

        # Define the variables and parameters
        x = pybamm.SpatialVariable("x", domain="rod", coord_sys="cartesian")
        T = pybamm.Variable("Temperature", domain="rod")
        k = pybamm.Parameter("Thermal diffusivity")

        # Write the governing equations
        N = -k * pybamm.grad(T)  # Heat flux
        Q = 1 - pybamm.Function(np.abs, x - 1)  # Source term
        dTdt = -pybamm.div(N) + Q
        model.rhs = {T: dTdt}  # add to model

        # Add the boundary and initial conditions
        model.boundary_conditions = {
            T: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        model.initial_conditions = {T: 2 * x - x**2}

        # Add desired output variables, geometry, parameters
        model.variables = {"Temperature": T, "Heat flux": N, "Heat source": Q}
        geometry = {"rod": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(2)}}}
        param = pybamm.ParameterValues({"Thermal diffusivity": 0.75})

        # Process model and geometry
        param.process_model(model)
        param.process_geometry(geometry)

        # Pick mesh, spatial method, and discretise
        submesh_types = {"rod": pybamm.Uniform1DSubMesh}
        var_pts = {x: 30}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        spatial_methods = {"rod": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        # Solve
        solver = pybamm.ScipySolver()
        t = np.linspace(0, 1, 100)
        solution = solver.solve(model, t)

        model.save_model("heat_equation", mesh=mesh)
        new_model = pybamm.load_model("heat_equation.json")

        new_solver = pybamm.ScipySolver()
        new_solution = new_solver.solve(new_model, t)

        for x, _ in enumerate(solution.all_ys):
            np.testing.assert_allclose(
                solution.all_ys[x], new_solution.all_ys[x], rtol=1e-7, atol=1e-6
            )
        os.remove("heat_equation.json")


class TestSerialise:
    # test the symbol encoder

    def test_symbol_encoder_symbol(self, mocker):
        """test basic symbol encoder with & without children"""

        # without children
        a, a_dict = scalar_var_dict(mocker)

        a_ser_json = Serialise._SymbolEncoder().default(a)

        assert a_ser_json == a_dict

        # with children
        add = pybamm.Addition(2, 4)
        add_json = {
            "py/id": mocker.ANY,
            "py/object": "pybamm.expression_tree.binary_operators.Addition",
            "name": "+",
            "id": mocker.ANY,
            "domains": {
                "primary": [],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
            "children": [
                {
                    "py/id": mocker.ANY,
                    "py/object": "pybamm.expression_tree.scalar.Scalar",
                    "name": "2.0",
                    "id": mocker.ANY,
                    "value": 2.0,
                    "children": [],
                },
                {
                    "py/id": mocker.ANY,
                    "py/object": "pybamm.expression_tree.scalar.Scalar",
                    "name": "4.0",
                    "id": mocker.ANY,
                    "value": 4.0,
                    "children": [],
                },
            ],
        }

        add_ser_json = Serialise._SymbolEncoder().default(add)

        assert add_ser_json == add_json

    def test_symbol_encoder_explicit_time_integral(self, mocker):
        """test symbol encoder with initial conditions"""
        expr = pybamm.ExplicitTimeIntegral(pybamm.Scalar(5), pybamm.Scalar(1))

        expr_json = {
            "py/object": "pybamm.expression_tree.unary_operators.ExplicitTimeIntegral",
            "py/id": mocker.ANY,
            "name": "explicit time integral",
            "id": mocker.ANY,
            "children": [
                {
                    "py/object": "pybamm.expression_tree.scalar.Scalar",
                    "py/id": mocker.ANY,
                    "name": "5.0",
                    "id": mocker.ANY,
                    "value": 5.0,
                    "children": [],
                }
            ],
            "initial_condition": {
                "py/object": "pybamm.expression_tree.scalar.Scalar",
                "py/id": mocker.ANY,
                "name": "1.0",
                "id": mocker.ANY,
                "value": 1.0,
                "children": [],
            },
        }

        expr_ser_json = Serialise._SymbolEncoder().default(expr)

        assert expr_json == expr_ser_json

    def test_symbol_encoder_event(self, mocker):
        """test symbol encoder with event"""

        expression = pybamm.Scalar(1)
        event = pybamm.Event("my event", expression)

        event_json = {
            "py/object": "pybamm.models.event.Event",
            "py/id": mocker.ANY,
            "name": "my event",
            "event_type": ["EventType.TERMINATION", 0],
            "expression": {
                "py/object": "pybamm.expression_tree.scalar.Scalar",
                "py/id": mocker.ANY,
                "name": "1.0",
                "id": mocker.ANY,
                "value": 1.0,
                "children": [],
            },
        }

        event_ser_json = Serialise._SymbolEncoder().default(event)
        assert event_ser_json == event_json

    # test the mesh encoder
    def test_mesh_encoder(self, mocker):
        mesh, mesh_json = mesh_var_dict(mocker)

        # serialise mesh
        mesh_ser_json = Serialise._MeshEncoder().default(mesh)

        assert mesh_ser_json == mesh_json

    def test_deconstruct_pybamm_dicts(self, mocker):
        """tests serialisation of dictionaries with pybamm classes as keys"""

        x = pybamm.SpatialVariable("x", "negative electrode")

        test_dict = {"rod": {x: {"min": 0.0, "max": 2.0}}}

        ser_dict = {
            "rod": {
                "symbol_x": {
                    "py/object": "pybamm.expression_tree.independent_variable.SpatialVariable",
                    "py/id": mocker.ANY,
                    "name": "x",
                    "id": mocker.ANY,
                    "domains": {
                        "primary": ["negative electrode"],
                        "secondary": [],
                        "tertiary": [],
                        "quaternary": [],
                    },
                    "children": [],
                },
                "x": {"min": 0.0, "max": 2.0},
            }
        }

        assert Serialise()._deconstruct_pybamm_dicts(test_dict) == ser_dict

    def test_get_pybamm_class(self, mocker):
        # symbol
        _, scalar_dict = scalar_var_dict(mocker)

        scalar_class = Serialise()._get_pybamm_class(scalar_dict)

        assert isinstance(scalar_class, pybamm.Scalar)

        # mesh
        _, mesh_dict = mesh_var_dict(mocker)

        mesh_class = Serialise()._get_pybamm_class(mesh_dict)

        assert isinstance(mesh_class, pybamm.Mesh)

        with pytest.raises(AttributeError):
            unrecognised_symbol = {
                "py/id": mocker.ANY,
                "py/object": "pybamm.expression_tree.scalar.Scale",
                "name": "5.0",
                "id": mocker.ANY,
                "value": 5.0,
                "children": [],
            }
            Serialise()._get_pybamm_class(unrecognised_symbol)

    def test_reconstruct_symbol(self, mocker):
        scalar, scalar_dict = scalar_var_dict(mocker)

        new_scalar = Serialise()._reconstruct_symbol(scalar_dict)

        assert new_scalar == scalar

    def test_reconstruct_expression_tree(self):
        y = pybamm.StateVector(slice(0, 1))
        t = pybamm.t
        equation = 2 * y + t

        equation_json = {
            "py/object": "pybamm.expression_tree.binary_operators.Addition",
            "py/id": 139691619709376,
            "name": "+",
            "id": -2595875552397011963,
            "domains": {
                "primary": [],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
            "children": [
                {
                    "py/object": "pybamm.expression_tree.binary_operators.Multiplication",
                    "py/id": 139691619709232,
                    "name": "*",
                    "id": 6094209803352873499,
                    "domains": {
                        "primary": [],
                        "secondary": [],
                        "tertiary": [],
                        "quaternary": [],
                    },
                    "children": [
                        {
                            "py/object": "pybamm.expression_tree.scalar.Scalar",
                            "py/id": 139691619709040,
                            "name": "2.0",
                            "id": 1254626814648295285,
                            "value": 2.0,
                            "children": [],
                        },
                        {
                            "py/object": "pybamm.expression_tree.state_vector.StateVector",
                            "py/id": 139691619589760,
                            "name": "y[0:1]",
                            "id": 5063056989669636089,
                            "domains": {
                                "primary": [],
                                "secondary": [],
                                "tertiary": [],
                                "quaternary": [],
                            },
                            "y_slice": [{"start": 0, "stop": 1, "step": None}],
                            "evaluation_array": [True],
                            "children": [],
                        },
                    ],
                },
                {
                    "py/object": "pybamm.expression_tree.independent_variable.Time",
                    "py/id": 139692083289392,
                    "name": "time",
                    "id": -3301344124754766351,
                    "domains": {
                        "primary": [],
                        "secondary": [],
                        "tertiary": [],
                        "quaternary": [],
                    },
                    "children": [],
                },
            ],
        }

        new_equation = Serialise()._reconstruct_expression_tree(equation_json)

        assert new_equation == equation

    def test_reconstruct_mesh(self, mocker):
        mesh, mesh_dict = mesh_var_dict(mocker)

        new_mesh = Serialise()._reconstruct_mesh(mesh_dict)

        testing.assert_array_equal(
            new_mesh["negative particle"].edges, mesh["negative particle"].edges
        )
        testing.assert_array_equal(
            new_mesh["negative particle"].nodes, mesh["negative particle"].nodes
        )

        # reconstructed meshes are only used for plotting, geometry not reconstructed.
        with pytest.raises(
            AttributeError, match=r"'Mesh' object has no attribute '_geometry'"
        ):
            assert new_mesh.geometry == mesh.geometry

    def test_reconstruct_pybamm_dict(self, mocker):
        x = pybamm.SpatialVariable("x", "negative electrode")

        test_dict = {"rod": {x: {"min": 0.0, "max": 2.0}}}

        ser_dict = {
            "rod": {
                "symbol_x": {
                    "py/object": "pybamm.expression_tree.independent_variable.SpatialVariable",
                    "py/id": mocker.ANY,
                    "name": "x",
                    "id": mocker.ANY,
                    "domains": {
                        "primary": ["negative electrode"],
                        "secondary": [],
                        "tertiary": [],
                        "quaternary": [],
                    },
                    "children": [],
                },
                "x": {"min": 0.0, "max": 2.0},
            }
        }

        new_dict = Serialise()._reconstruct_pybamm_dict(ser_dict)

        assert new_dict == test_dict

        # test recreation if not passed a dict
        test_list = ["left", "right"]
        new_list = Serialise()._reconstruct_pybamm_dict(test_list)

        assert test_list == new_list

    def test_convert_options(self):
        options_dict = {
            "current collector": "uniform",
            "particle phases": ["2", "1"],
            "open-circuit potential": [["single", "current sigmoid"], "single"],
        }

        options_result = {
            "current collector": "uniform",
            "particle phases": ("2", "1"),
            "open-circuit potential": (("single", "current sigmoid"), "single"),
        }

        assert Serialise()._convert_options(options_dict) == options_result

    def test_save_load_model(self, mocker, tmp_path, request):
        os.chdir(tmp_path)
        request.addfinalizer(lambda: os.chdir(os.getcwd()))

        model = pybamm.lithium_ion.SPM(name="test_spm")
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

        # test error if not discretised
        with pytest.raises(
            NotImplementedError,
            match=r"PyBaMM can only serialise a discretised, ready-to-solve model",
        ):
            Serialise().save_model(model, filename="test_model")

        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # default save
        Serialise().save_model(model, filename="test_model")
        assert os.path.exists("test_model.json")

        # default save where filename isn't provided
        fixed_dt = datetime(2025, 12, 25, 0, 0, 0)
        mocked_dt = mocker.patch("pybamm.expression_tree.operations.serialise.datetime")
        mocked_dt.now.return_value = fixed_dt
        Serialise().save_model(model)
        filename = "test_spm_" + fixed_dt.strftime("%Y_%m_%d-%p%I_%M") + ".json"
        assert os.path.exists(filename)
        os.remove(filename)

        # default load
        new_model = Serialise().load_model("test_model.json")

        # check new model solves
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, [0, 3600])

        # check an error is raised when plotting the solution
        with pytest.raises(AttributeError):
            new_solution.plot()

        # load when specifying the battery model to use
        newest_model = Serialise().load_model("test_model.json")

        # Test for error if no model type is provided
        with open("test_model.json") as f:
            model_data = json.load(f)
            del model_data["py/object"]

        with open("test_model.json", "w") as f:
            json.dump(model_data, f)

        with pytest.raises(TypeError):
            Serialise().load_model("test_model.json")

        os.remove("test_model.json")

        # check new model solves
        newest_solver = newest_model.default_solver
        newest_solver.solve(newest_model, [0, 3600])

    def test_save_experiment_model_error(self):
        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment(["Discharge at 1C for 1 hour"])
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.solve()

        with pytest.raises(
            NotImplementedError,
            match=r"Serialising models coupled to experiments is not yet supported\.",
        ):
            sim.save_model("spm_experiment", mesh=False, variables=False)

    def test_serialised_model_plotting(self):
        # models without a mesh
        model = pybamm.BaseModel()
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables["c"] = c
        model.variables["2c"] = 2 * c

        # setup and discretise
        _ = pybamm.ScipySolver().solve(model, np.linspace(0, 1))

        assert set(model.get_processed_variables_dict().keys()) == set(
            model.variables.keys()
        )

        Serialise().save_model(
            model,
            filename="test_base_model",
        )

        new_model = Serialise().load_model("test_base_model.json")
        os.remove("test_base_model.json")

        assert set(new_model.get_processed_variables_dict().keys()) == set(
            model.variables.keys()
        )

        new_solution = pybamm.ScipySolver().solve(new_model, np.linspace(0, 1))

        # check dynamic plot loads
        new_solution.plot(["c", "2c"], show_plot=False)

        # models with a mesh ----------------
        model = pybamm.lithium_ion.SPM(name="test_spm_plotting")
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        Serialise().save_model(
            model,
            mesh=mesh,
            filename="test_plotting_model",
        )

        new_model = Serialise().load_model("test_plotting_model.json")
        os.remove("test_plotting_model.json")

        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, [0, 3600])

        # check dynamic plot loads
        new_solution.plot(show_plot=False)

    # testing custom models serilaisation and deserialisation

    def test_serialise_scalar(self):
        S = pybamm.Scalar(2.718)
        j = convert_symbol_to_json(S)
        S2 = convert_symbol_from_json(j)
        assert isinstance(S2, pybamm.Scalar)
        assert S2.value == pytest.approx(2.718)

    def test_serialise_time(self):
        t = pybamm.Time()
        j = convert_symbol_to_json(t)
        t2 = convert_symbol_from_json(j)
        assert isinstance(t2, pybamm.Time)

    def test_serialise_input_parameter(self):
        """Test InputParameter serialization and deserialization."""
        ip = pybamm.InputParameter("test_param")
        j = convert_symbol_to_json(ip)
        ip_restored = convert_symbol_from_json(j)
        assert isinstance(ip_restored, pybamm.InputParameter)
        assert ip_restored.name == "test_param"

    def test_convert_symbol_to_json_with_number_and_list(self):
        for val in (0, 3.14, -7, True):
            out = convert_symbol_to_json(val)
            assert out is val or out == val

        sample = [1, 2, 3, "foo", 4.5]
        out = convert_symbol_to_json(sample)
        assert out is sample

    def test_convert_symbol_from_json_with_primitives(self):
        assert convert_symbol_from_json(3.14) == 3.14
        assert convert_symbol_from_json(42) == 42
        assert convert_symbol_from_json(True) is True

    def test_convert_symbol_from_json_with_none(self):
        assert convert_symbol_from_json(None) is None

    def test_convert_symbol_from_json_unexpected_string(self):
        with pytest.raises(ValueError, match=r"Unexpected raw string in JSON: foo"):
            convert_symbol_from_json("foo")

    def test_numpy_array_conversion(self):
        arr = np.array([1, 2, 3])
        assert Serialise._json_encoder(arr) == [1, 2, 3]

    def test_numpy_float_conversion(self):
        val1 = np.float32(2.71)
        result1 = Serialise._json_encoder(val1)
        assert result1 == float(val1)
        assert isinstance(result1, float)

        val2 = np.float64(3.14)
        result2 = Serialise._json_encoder(val2)
        assert result2 == float(val2)
        assert isinstance(result2, float)

    def test_numpy_int_conversion(self):
        val1 = np.int32(42)
        result1 = Serialise._json_encoder(val1)
        assert result1 == int(val1)
        assert isinstance(result1, int)

        val2 = np.int64(123)
        result2 = Serialise._json_encoder(val2)
        assert result2 == int(val2)
        assert isinstance(result2, int)

    def test_unsupported_type_raises(self):
        class Dummy:
            pass

        with pytest.raises(TypeError, match=r"is not JSON serializable"):
            Serialise._json_encoder(Dummy())

    def test_create_symbol_key(self):
        var1 = pybamm.Variable("x", bounds=(0, 1))
        var2 = pybamm.Variable("x", bounds=(0, 2))

        json1 = convert_symbol_to_json(var1)
        json2 = convert_symbol_to_json(var2)

        key1 = Serialise._create_symbol_key(json1)
        key2 = Serialise._create_symbol_key(json2)

        assert isinstance(key1, str)
        assert isinstance(key2, str)
        assert key1 != key2

    def test_primary_broadcast_serialisation(self):
        child = pybamm.Scalar(42)
        symbol = pybamm.PrimaryBroadcast(child, "negative electrode")
        json_dict = convert_symbol_to_json(symbol)
        symbol2 = convert_symbol_from_json(json_dict)

        assert isinstance(symbol2, pybamm.PrimaryBroadcast)
        assert symbol2.broadcast_domain == ["negative electrode"]
        assert isinstance(symbol2.orphans[0], pybamm.Scalar)
        assert symbol2.orphans[0].value == 42

    def test_interpolant_serialisation(self):
        x = np.linspace(0, 1, 5)
        y = np.array([0, 1, 4, 9, 16])
        child = pybamm.Variable("z")
        interp = pybamm.Interpolant(
            x, y, child, name="test_interplolant", interpolator="linear"
        )
        json_dict = convert_symbol_to_json(interp)
        interp2 = convert_symbol_from_json(json_dict)

        assert isinstance(interp2, pybamm.Interpolant)
        assert interp2.name == "test_interplolant"
        assert interp2.interpolator == "linear"
        assert isinstance(interp2.x[0], np.ndarray)
        assert isinstance(interp2.y, np.ndarray)
        assert interp2.children[0].name == "z"

    def test_variable_serialisation(self):
        var = pybamm.Variable("var", domain="separator")
        json_dict = convert_symbol_to_json(var)
        var2 = convert_symbol_from_json(json_dict)

        assert isinstance(var2, pybamm.Variable)
        assert var2.name == "var"
        assert var2.domains["primary"] == ["separator"]
        assert var2.bounds[0].value == -float("inf")
        assert var2.bounds[1].value == float("inf")

    def test_coupled_variable_serialisation(self):
        # Test basic CoupledVariable serialisation
        cv = pybamm.CoupledVariable("Voltage [V]")
        json_dict = convert_symbol_to_json(cv)
        cv2 = convert_symbol_from_json(json_dict)

        assert isinstance(cv2, pybamm.CoupledVariable)
        assert cv2.name == "Voltage [V]"

    def test_coupled_variable_in_expression_serialisation(self):
        # Test CoupledVariable used in an expression
        cv = pybamm.CoupledVariable("Voltage [V]")
        expr = cv * 2
        json_dict = convert_symbol_to_json(expr)
        expr2 = convert_symbol_from_json(json_dict)

        assert isinstance(expr2, pybamm.Multiplication)
        # Find the CoupledVariable in the expression
        coupled_vars = [
            node
            for node in expr2.pre_order()
            if isinstance(node, pybamm.CoupledVariable)
        ]
        assert len(coupled_vars) == 1
        assert coupled_vars[0].name == "Voltage [V]"

    def test_concatenation_variable_serialisation(self):
        var1 = pybamm.Variable("a", domain="negative electrode")
        var2 = pybamm.Variable("a", domain="separator")
        var3 = pybamm.Variable("a", domain="positive electrode")
        concat_var = pybamm.ConcatenationVariable(var1, var2, var3, name="conc_var")
        json_dict = convert_symbol_to_json(concat_var)
        concat_var2 = convert_symbol_from_json(json_dict)

        assert isinstance(concat_var2, pybamm.ConcatenationVariable)
        assert concat_var2.name == "a"
        assert len(concat_var2.children) == 3
        domains = [child.domains["primary"] for child in concat_var2.children]
        assert domains == [
            ["negative electrode"],
            ["separator"],
            ["positive electrode"],
        ]

    def test_full_broadcast_serialisation(self):
        child = pybamm.Scalar(5)
        fb = pybamm.FullBroadcast(
            child,
            "negative electrode",
            {"primary": ["negative electrode"], "secondary": ["current collector"]},
        )
        json_dict = convert_symbol_to_json(fb)
        fb2 = convert_symbol_from_json(json_dict)

        assert isinstance(fb2, pybamm.FullBroadcast)
        assert fb2.broadcast_domain == ["negative electrode"]
        assert fb2.domains["primary"] == ["negative electrode"]
        assert fb2.domains["secondary"] == ["current collector"]
        assert isinstance(fb2.child, pybamm.Scalar)
        assert fb2.child.value == 5

    def test_secondary_broadcast_serialisation(self):
        child = pybamm.Variable("c", domain="negative electrode")
        sb = pybamm.SecondaryBroadcast(child, "current collector")

        json_dict = convert_symbol_to_json(sb)
        sb2 = convert_symbol_from_json(json_dict)

        assert isinstance(sb2, pybamm.SecondaryBroadcast)
        assert sb2.broadcast_domain == ["current collector"]
        assert sb2.child.name == "c"
        assert sb2.child.domain == ["negative electrode"]

    def test_spatial_variable_serialisation(self):
        sv = pybamm.SpatialVariable(
            "x", domain="negative electrode", coord_sys="cartesian"
        )
        json_dict = convert_symbol_to_json(sv)
        sv2 = convert_symbol_from_json(json_dict)

        assert isinstance(sv2, pybamm.SpatialVariable)
        assert sv2.name == "x"
        assert sv2.domains["primary"] == ["negative electrode"]
        assert sv2.coord_sys == "cartesian"

    def test_boundary_value_serialisation(self):
        var = pybamm.SpatialVariable("x", domain="electrode")
        bv = pybamm.BoundaryValue(var, "left")
        json_dict = convert_symbol_to_json(bv)
        bv2 = convert_symbol_from_json(json_dict)

        assert isinstance(bv2, pybamm.BoundaryValue)
        assert bv2.side == "left"
        assert isinstance(bv2.orphans[0], pybamm.SpatialVariable)
        assert bv2.orphans[0].name == "x"

    def test_specific_function_not_supported(self):
        def dummy_func(x):
            return x

        symbol = pybamm.SpecificFunction(dummy_func, pybamm.Scalar(1))
        with pytest.raises(
            NotImplementedError, match=r"SpecificFunction is not supported directly"
        ):
            convert_symbol_to_json(symbol)

    def test_unary_operator_serialisation(self):
        expr = pybamm.Negate(pybamm.Scalar(5))
        json_dict = convert_symbol_to_json(expr)
        expr2 = convert_symbol_from_json(json_dict)

        assert isinstance(expr2, pybamm.Negate)
        assert isinstance(expr2.child, pybamm.Scalar)
        assert expr2.child.value == 5

    def test_binary_operator_serialisation(self):
        expr = pybamm.Addition(pybamm.Scalar(2), pybamm.Scalar(3))
        json_dict = convert_symbol_to_json(expr)
        expr2 = convert_symbol_from_json(json_dict)

        assert isinstance(expr2, pybamm.Addition)
        values = [c.value for c in expr2.children]
        assert values == [2, 3]

    def test_symbol_deserialization_with_domains(self):
        json_data = {
            "type": "Symbol",
            "name": "test symbol",
            "domains": {
                "primary": ["negative electrode", "separator", "positive electrode"],
                "secondary": ["current collector"],
            },
        }

        symbol = convert_symbol_from_json(json_data)

        assert isinstance(symbol, pybamm.Symbol)
        assert symbol.name == "test symbol"
        assert symbol.domains == {
            "primary": ["negative electrode", "separator", "positive electrode"],
            "secondary": ["current collector"],
            "tertiary": [],
            "quaternary": [],
        }

    def test_import_base_class_non_builtin_object(self, tmp_path):
        # Minimal model JSON with a non-existent base class
        model_json = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            "model": {
                "base_class": "nonexistent_module.DummyModel",
                "name": "DummyModel",
                "rhs": [],
                "algebraic": [],
                "initial_conditions": [],
                "boundary_conditions": [],
                "events": [],
                "variables": {},
            },
        }

        file_path = tmp_path / "model.json"

        with open(file_path, "w") as f:
            json.dump(model_json, f)

        with pytest.raises(
            ImportError,
            match=r"(?i)Could not import base class 'nonexistent_module\.DummyModel'",
        ):
            Serialise.load_custom_model(str(file_path))

    def test_function_parameter_with_diff_variable_serialisation(self):
        x = pybamm.Variable("x")
        diff_var = pybamm.Variable("r")
        func_param = pybamm.FunctionParameter("my_func", {"x": x}, diff_var)

        json_dict = convert_symbol_to_json(func_param)
        assert "diff_variable" in json_dict
        assert json_dict["diff_variable"]["type"] == "Variable"
        assert json_dict["diff_variable"]["name"] == "r"

        expr2 = convert_symbol_from_json(json_dict)
        assert isinstance(expr2, pybamm.FunctionParameter)
        assert expr2.diff_variable.name == "r"
        assert expr2.name == "my_func"
        assert list(expr2.input_names) == ["x"]

    def test_indefinite_integral_serialisation(self):
        x = pybamm.SpatialVariable("x", domain="negative electrode")
        ind_int = pybamm.IndefiniteIntegral(x, x)

        json_dict = convert_symbol_to_json(ind_int)
        assert json_dict["type"] == "IndefiniteIntegral"

        assert (
            isinstance(json_dict["children"], list) and len(json_dict["children"]) == 1
        )
        child_json = json_dict["children"][0]
        assert child_json["type"] == "SpatialVariable"
        assert child_json["name"] == "x"

        int_var_json = json_dict["integration_variable"]
        assert int_var_json["type"] == "SpatialVariable"
        assert int_var_json["name"] == "x"

        expr2 = convert_symbol_from_json(json_dict)
        assert isinstance(expr2, pybamm.IndefiniteIntegral)
        assert isinstance(expr2.child, pybamm.SpatialVariable)
        assert expr2.child.name == "x"
        assert isinstance(expr2.integration_variable, list)
        assert len(expr2.integration_variable) == 1
        assert isinstance(expr2.integration_variable[0], pybamm.SpatialVariable)
        assert expr2.integration_variable[0].name == "x"

        bad_json_dict = json_dict.copy()
        bad_json_dict["integration_variable"] = {
            "type": "Symbol",  # Something not a SpatialVariable
            "name": "not spatial",
            "domains": {},
        }

        with pytest.raises(TypeError, match=r"Expected SpatialVariable"):
            convert_symbol_from_json(bad_json_dict)

    def test_invalid_filename(self):
        model = pybamm.lithium_ion.DFN()
        with pytest.raises(
            ValueError, match=r"Filename 'dfn' must end with '.json' extension."
        ):
            Serialise.save_custom_model(model, filename="dfn")

    def test_symbol_fallback_serialisation(self):
        var = pybamm.Variable("v", domain="electrode")
        diff = pybamm.Gradient(var)
        json_dict = convert_symbol_to_json(diff)
        diff2 = convert_symbol_from_json(json_dict)

        assert isinstance(diff2, pybamm.Gradient)
        assert isinstance(diff2.children[0], pybamm.Variable)
        assert diff2.children[0].name == "v"
        assert diff2.children[0].domains["primary"] == ["electrode"]

    def test_unhandled_symbol_type_error(self):
        class NotSymbol:
            def __init__(self):
                self.name = "not_a_symbol"

        dummy = NotSymbol()
        with pytest.raises(ValueError) as e:
            convert_symbol_to_json(dummy)

        assert "Error processing 'not_a_symbol'. Unknown symbol type:" in str(e.value)

    def test_deserialising_unhandled_type(self):
        unhandled_json = {"type": "NotARealSymbol", "foo": "bar"}
        with pytest.raises(
            ValueError,
            match=r"Unknown symbol type: NotARealSymbol",
        ):
            convert_symbol_from_json(unhandled_json)

        unhandled_json2 = {"a": 1, "b": 2}
        with pytest.raises(
            ValueError, match=r"Missing 'type' key in JSON data: {'a': 1, 'b': 2}"
        ):
            convert_symbol_from_json(unhandled_json2)

    def test_file_write_raises_ioerror(self):
        # testing behaviour when file system is read-only to raise exception
        model = pybamm.lithium_ion.SPM()

        with patch("builtins.open", mock_open()) as file:
            file.side_effect = OSError("file system is read-only")

            with pytest.raises(
                ValueError,
                match=r"Failed to save custom model: Failed to write model JSON to file",
            ):
                Serialise.save_custom_model(model, "readonly_test.json")

    def test_symbol_conversion_failure_raises_value_error(self):
        model = pybamm.BaseModel()
        model.name = "TestModel"
        model.rhs = {pybamm.Variable("c"): pybamm.Variable("c")}

        with patch(
            "pybamm.expression_tree.operations.serialise.convert_symbol_to_json",
            side_effect=Exception("conversion failed"),
        ):
            with pytest.raises(
                ValueError, match=r"Failed to save custom model: conversion failed"
            ):
                Serialise.save_custom_model(model, "conversion_fail")

    def test_unsupported_schema_version(self, tmp_path):
        unhandled_schema_json = {
            "schema_version": "9.9",  # Unsupported
            "pybamm_version": pybamm.__version__,
            "name": "BadModel",
            "rhs": [],
            "algebraic": [],
            "initial_conditions": [],
            "boundary_conditions": [],
            "events": [],
            "variables": {},
        }

        file_path = tmp_path / "model.json"

        with open(file_path, "w") as f:
            json.dump(unhandled_schema_json, f)

        with pytest.raises(ValueError, match=r"Unsupported schema version: 9\.9"):
            Serialise.load_custom_model(file_path)

    @pytest.mark.parametrize(
        "compress", [False, True], ids=["uncompressed", "compressed"]
    )
    def test_model_has_correct_schema_version(self, tmp_path, compress):
        model = BasicDFN()
        file_path = tmp_path / "test_schema_version.json"

        Serialise.save_custom_model(model, filename=str(file_path), compress=compress)

        loaded_model = Serialise.load_custom_model(str(file_path))

        assert hasattr(loaded_model, "schema_version")
        assert loaded_model.schema_version == SUPPORTED_SCHEMA_VERSION

    def test_load_invalid_json(self):
        invalid_json = "{ invalid json"

        with patch("builtins.open", mock_open(read_data=invalid_json)):
            with pytest.raises(pybamm.InvalidModelJSONError) as e:
                Serialise.load_custom_model("invalid_json.json")

            assert "contains invalid JSON" in str(e.value)

    def test_load_custom_model_file_not_found(self):
        with pytest.raises(FileNotFoundError) as e:
            Serialise.load_custom_model("non_existent_file.json")
        assert "Could not find file" in str(e.value)

    def test_invalid_symbol_key_raises_value_error(self, tmp_path):
        # Malformed LHS (invalid symbol type)
        bad_lhs = {"not_a_valid_symbol": 123}
        rhs_expr = {"type": "Scalar", "value": 1.0}

        model_json = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            "model": {
                "base_class": "",
                "name": "BadSymbolKeyModel",
                "rhs": [[bad_lhs, rhs_expr]],
                "algebraic": [],
                "initial_conditions": [],
                "boundary_conditions": [],
                "events": [],
                "variables": {},
            },
        }

        file_path = tmp_path / "model.json"
        with open(file_path, "w") as f:
            json.dump(model_json, f)

        with pytest.raises(
            ValueError,
            match=r"Failed to process symbol key for variable {'not_a_valid_symbol': 123}",
        ):
            Serialise.load_custom_model(str(file_path))

    def test_save_raises_for_missing_sections(self):
        class DummyModelMissing:
            # e.g. only has rhs and algebraic
            def __init__(self):
                self.rhs = {}
                self.algebraic = {}
                self.is_processed = False

        m = DummyModelMissing()
        with pytest.raises(AttributeError) as e:
            Serialise.save_custom_model(m, filename="irrelevant")
        msg = str(e.value)
        assert "missing required sections" in msg.lower()
        assert any(
            section in msg for section in ["initial_conditions", "events", "variables"]
        )

    def test_save_raises_for_being_processed(self):
        class DummyModelMissing:
            # e.g. only has rhs and algebraic
            def __init__(self):
                self.is_processed = True

        m = DummyModelMissing()
        with pytest.raises(ValueError, match=r"Cannot serialise a built model."):
            Serialise.save_custom_model(m, filename="irrelevant")

    def test_model_with_missing_json_sections(self, tmp_path):
        model_json = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            "model": {
                "name": "BadModel",
                "base_class": "",
                "algebraic": [],
                "initial_conditions": [],
            },
        }

        file_path = tmp_path / "model1.json"

        with open(file_path, "w") as f:
            json.dump(model_json, f)

        with pytest.raises(
            KeyError, match=r"(?i)rhs.*boundary_conditions.*events.*variables"
        ):
            Serialise.load_custom_model(str(file_path))

    def test_invalid_rhs_entry_raises_value_error(self, tmp_path):
        good_lhs = {
            "type": "Variable",
            "name": "x",
            "domains": {},
        }
        bad_rhs = {"this_will_fail": True}

        model_json = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            "model": {
                "base_class": "",
                "name": "BadModel",
                "rhs": [[good_lhs, bad_rhs]],
                "algebraic": [],
                "initial_conditions": [],
                "boundary_conditions": [],
                "events": [],
                "variables": {},
            },
        }

        file_path = tmp_path / "model2.json"

        with open(file_path, "w") as f:
            json.dump(model_json, f)

        with pytest.raises(
            ValueError,
            match=r"Failed to convert rhs",
        ):
            Serialise.load_custom_model(str(file_path))

    def test_invalid_algebraic_entry_raises_value_error(self, tmp_path):
        # Build JSON with all required keys, but rhs has a bad entry
        good_lhs = {
            "type": "Variable",
            "name": "x",
            "domains": {},
        }
        bad_rhs = {"this_will_fail": True}

        # 2) Build JSON with all required keys
        model_json = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            "model": {
                "base_class": "",
                "name": "BadModel",
                # One valid pair in RHS
                "rhs": [],
                "algebraic": [[good_lhs, bad_rhs]],
                "initial_conditions": [],
                "boundary_conditions": [],
                "events": [],
                "variables": {},
            },
        }
        file_path = tmp_path / "model3.json"

        with open(file_path, "w") as f:
            json.dump(model_json, f)

        with pytest.raises(
            ValueError,
            match=r"Failed to convert algebraic",
        ):
            Serialise.load_custom_model(str(file_path))

    def test_invalid_initial_conditions_entry_raises_value_error(self, tmp_path):
        # Build JSON with all required keys, but rhs has a bad entry
        good_lhs = {
            "type": "Variable",
            "name": "x",
            "domains": {},
        }
        bad_rhs = {"this_will_fail": True}

        # 2) Build JSON with all required keys
        model_json = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            "model": {
                "base_class": "",
                "name": "BadModel",
                # One valid pair in RHS
                "rhs": [],
                "algebraic": [],
                "initial_conditions": [[good_lhs, bad_rhs]],
                "boundary_conditions": [],
                "events": [],
                "variables": {},
            },
        }
        file_path = tmp_path / "model4.json"

        with open(file_path, "w") as f:
            json.dump(model_json, f)

        with pytest.raises(
            ValueError,
            match=r"Failed to convert initial condition",
        ):
            Serialise.load_custom_model(str(file_path))

    def test_invalid_boundary_conditions_raise_value_error(self, tmp_path):
        good_variable = {
            "type": "Variable",
            "name": "x",
            "domains": {},
        }

        # Malformed RHS: missing tuple structure
        bad_condition_dict = {
            "left": {
                "this_is_not_valid": True
            },  # Should be (expression_json, boundary_type)
        }

        model_json = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            "model": {
                "base_class": "",
                "name": "BadBoundaryModel",
                "rhs": [],
                "algebraic": [],
                "initial_conditions": [],
                "boundary_conditions": [[good_variable, bad_condition_dict]],
                "events": [],
                "variables": {},
                "all_variable_keys": [good_variable],
            },
        }

        file_path = tmp_path / "model5.json"

        with open(file_path, "w") as f:
            json.dump(model_json, f)

        with pytest.raises(
            ValueError,
            match=r"(?i)failed to convert boundary.*not enough values to unpack",
        ):
            Serialise.load_custom_model(str(file_path))

        # Valid variable
        variable_json = {
            "type": "Variable",
            "name": "c",
            "domains": {},
        }

        invalid_expression_json = "not_a_valid_expression"

        condition_dict = {"left": (invalid_expression_json, "Dirichlet")}

        model_data = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            "model": {
                "base_class": "",
                "name": "BadBoundaryExpressionModel",
                "rhs": [],
                "algebraic": [],
                "initial_conditions": [],
                "boundary_conditions": [[variable_json, condition_dict]],
                "events": [],
                "variables": {},
                "all_variable_keys": [variable_json],
            },
        }

        model_file = tmp_path / "bad_boundary_expr.json"
        with open(model_file, "w") as f:
            json.dump(model_data, f)

        with pytest.raises(
            ValueError,
            match=r"(?i)failed to convert boundary expression.*left.*not_a_valid_expression",
        ):
            Serialise.load_custom_model(str(model_file))

    def test_event_conversion_failure(self, tmp_path):
        model_data = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            "model": {
                "base_class": "",
                "name": "BadEventModel",
                "rhs": [],
                "algebraic": [],
                "initial_conditions": [],
                "boundary_conditions": [],
                "variables": {},
                "events": [
                    {
                        "name": "Bad Event",
                        "expression": {"bad": "structure"},  # malformed
                        "event_type": "termination",
                    }
                ],
            },
        }

        file = tmp_path / "bad_event_model.json"
        with open(file, "w") as f:
            json.dump(model_data, f)

        with pytest.raises(
            ValueError,
            match=r"Failed to convert event 'Bad Event'",
        ):
            Serialise.load_custom_model(str(file))

    def test_variable_conversion_failure(self, tmp_path):
        model_data = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            "model": {
                "base_class": "",
                "name": "BadVariableModel",
                "rhs": [],
                "algebraic": [],
                "initial_conditions": [],
                "boundary_conditions": [],
                "events": [],
                "variables": {"Bad Variable": {"bad": "structure"}},
                "fixed_input_parameters": {},
            },
        }

        file = tmp_path / "bad_variable_model.json"
        with open(file, "w") as f:
            json.dump(model_data, f)

        with pytest.raises(
            ValueError,
            match=r"Failed to convert variable 'Bad Variable'",
        ):
            Serialise.load_custom_model(str(file))

    @pytest.mark.parametrize(
        "compress", [False, True], ids=["uncompressed", "compressed"]
    )
    def test_save_and_load_custom_model(self, tmp_path, monkeypatch, compress):
        model = pybamm.BaseModel(name="test_model")
        a = pybamm.Variable("a", domain="electrode")
        b = pybamm.Variable("b", domain="electrode")
        model.rhs = {a: b}
        model.initial_conditions = {a: pybamm.Scalar(1)}
        model.algebraic = {}
        model.boundary_conditions = {a: {"left": (pybamm.Scalar(0), "Dirichlet")}}
        model.events = [pybamm.Event("terminal", pybamm.Scalar(1) - b, "TERMINATION")]
        model.variables = {"a": a, "b": b}

        # Save model to specified filename
        file_path = tmp_path / "test_model.json"
        Serialise.save_custom_model(model, filename=str(file_path), compress=compress)
        assert file_path.exists()

        # Save using default filename logic
        with monkeypatch.context() as m:
            m.chdir(tmp_path)
            Serialise().save_custom_model(model, compress=compress)
            pattern = r"test_model_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}\.json"
            matched = [f for f in os.listdir(tmp_path) if re.fullmatch(pattern, f)]
            assert matched

        # Load and test model
        loaded_model = Serialise.load_custom_model(str(file_path))
        assert loaded_model.name == "test_model"
        assert isinstance(loaded_model.rhs, dict)
        assert next(iter(loaded_model.rhs.keys())).name == "a"
        assert next(iter(loaded_model.rhs.values())).name == "b"

    @pytest.mark.parametrize(
        "compress", [False, True], ids=["uncompressed", "compressed"]
    )
    @pytest.mark.parametrize(
        "model, filename",
        [
            (BasicSPM(), "basic_spm.json"),
            (BasicDFN(), "basic_dfn.json"),
            (pybamm.lithium_ion.SPM(), "spm.json"),
            (pybamm.lithium_ion.DFN(), "dfn.json"),
        ],
        ids=["basic_spm", "basic_dfn", "spm", "dfn"],
    )
    def test_plotting_serialised_models(self, model, filename, tmp_path, compress):
        path = tmp_path / filename
        Serialise.save_custom_model(model, filename=str(path), compress=compress)
        loaded_model = Serialise.load_custom_model(str(path))
        sim = pybamm.Simulation(loaded_model)
        sim.solve([0, 3600])
        sim.plot(show_plot=False)

    def test_parameter_serialisation(self, tmp_path):
        file_path = tmp_path / "params.json"

        # Load Marquis parameters
        params = pybamm.ParameterValues("Marquis2019")

        # Save to JSON
        Serialise.save_parameters(params, file_path)

        # Load back
        param2 = Serialise.load_parameters(file_path)

        # Build and run
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, parameter_values=param2)
        sim.solve([0, 3600])
        sim.plot(show_plot=False)


class TestExpressionFunctionParameter:
    def test_basic_serialisation(self):
        x = pybamm.SpatialVariable("x", domain="negative electrode")
        expr = pybamm.FunctionParameter("b", {"x": x}) + pybamm.Parameter("c")

        efp = ExpressionFunctionParameter("f", expr, "f", ["x"])
        src = efp.to_source()

        # Check flexible matching (order may differ)
        assert "def f(x):" in src
        assert 'Parameter("c")' in src
        assert "b" in src or "FunctionParameter" in src

    def test_multiple_args(self):
        x = pybamm.Variable("x")
        y = pybamm.Variable("y")
        expr = x * y + pybamm.Parameter("d")

        efp = ExpressionFunctionParameter("f", expr, "f", ["x", "y"])
        src = efp.to_source()

        assert "def f(x, y):" in src
        assert "x*y" in src or "x * y" in src
        assert 'Parameter("d")' in src

    def test_nested_expression(self):
        z = pybamm.Variable("z")
        expr = pybamm.Parameter("a") * (z + 2)

        efp = ExpressionFunctionParameter("f", expr, "f", ["z"])
        src = efp.to_source()

        assert "def f(z):" in src
        assert 'Parameter("a")' in src
        assert "(z + 2" in src  # allows 2 or 2.0


class TestGeometrySerialization:
    def test_serialise_and_load_geometry(self):
        """Test saving and loading geometry to/from file."""
        # Create a custom geometry
        geometry = pybamm.battery_geometry()

        # Use temporary directory for test files
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_geometry.json"

            # Save geometry
            Serialise.save_custom_geometry(geometry, filename=str(filepath))
            assert filepath.exists()

            # Load geometry
            loaded_geometry = Serialise.load_custom_geometry(str(filepath))

            # Verify domains match
            assert set(loaded_geometry.keys()) == set(geometry.keys())

            # Verify spatial variables and their bounds
            for domain in geometry.keys():
                assert domain in loaded_geometry
                # Compare variable names
                orig_vars = {
                    (var.name if hasattr(var, "name") else var)
                    for var in geometry[domain].keys()
                    if var != "tabs"
                }
                loaded_vars = {
                    (var.name if hasattr(var, "name") else var)
                    for var in loaded_geometry[domain].keys()
                    if var != "tabs"
                }
                assert orig_vars == loaded_vars

    def test_serialise_and_load_geometry_dict(self):
        """Test serializing to dict and loading from dict."""
        # Create a custom geometry
        geometry = pybamm.battery_geometry()

        # Serialize to dict
        geometry_dict = Serialise.serialise_custom_geometry(geometry)

        # Verify structure
        assert "schema_version" in geometry_dict
        assert "pybamm_version" in geometry_dict
        assert "geometry" in geometry_dict

        # Load from dict
        loaded_geometry = Serialise.load_custom_geometry(geometry_dict)

        # Verify domains match
        assert set(loaded_geometry.keys()) == set(geometry.keys())

    def test_geometry_with_default_filename(self, monkeypatch):
        """Test geometry saving with auto-generated filename."""
        geometry = pybamm.battery_geometry()

        with tempfile.TemporaryDirectory() as tmpdir:
            with monkeypatch.context() as m:
                m.chdir(tmpdir)

                # Save with no filename (auto-generate)
                Serialise.save_custom_geometry(geometry)

                # Check a file was created
                json_files = list(Path(tmpdir).glob("geometry_*.json"))
                assert len(json_files) == 1

    def test_geometry_invalid_extension(self):
        """Test that non-.json extension raises error."""
        geometry = pybamm.battery_geometry()

        with pytest.raises(ValueError, match=r"must end with '\.json' extension"):
            Serialise.save_custom_geometry(geometry, filename="test.txt")


class TestSpatialMethodsSerialization:
    def test_serialise_and_load_spatial_methods(self):
        """Test saving and loading spatial methods to/from file."""
        # Create spatial methods dict
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "current collector": pybamm.ZeroDimensionalSpatialMethod(),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_spatial_methods.json"

            # Save spatial methods
            Serialise.save_spatial_methods(spatial_methods, filename=str(filepath))
            assert filepath.exists()

            # Load spatial methods
            loaded_methods = Serialise.load_spatial_methods(str(filepath))

            # Verify domains match
            assert set(loaded_methods.keys()) == set(spatial_methods.keys())

            # Verify class types match
            for domain in spatial_methods.keys():
                assert isinstance(loaded_methods[domain], type(spatial_methods[domain]))

            # Verify options are preserved
            for domain in spatial_methods.keys():
                if hasattr(spatial_methods[domain], "options"):
                    assert (
                        loaded_methods[domain].options
                        == spatial_methods[domain].options
                    )

    def test_serialise_and_load_spatial_methods_dict(self):
        """Test serializing to dict and loading from dict."""
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
        }

        # Serialize to dict
        methods_dict = Serialise.serialise_spatial_methods(spatial_methods)

        # Verify structure
        assert "schema_version" in methods_dict
        assert "pybamm_version" in methods_dict
        assert "spatial_methods" in methods_dict

        # Load from dict
        loaded_methods = Serialise.load_spatial_methods(methods_dict)

        # Verify domains match
        assert set(loaded_methods.keys()) == set(spatial_methods.keys())

    def test_spatial_methods_with_options(self):
        """Test that custom options are preserved."""
        # Create spatial method with custom options
        custom_options = {
            "extrapolation": {
                "order": {"gradient": "linear", "value": "quadratic"},
                "use bcs": True,
            }
        }
        spatial_methods = {"macroscale": pybamm.FiniteVolume(options=custom_options)}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_methods.json"

            # Save and load
            Serialise.save_spatial_methods(spatial_methods, filename=str(filepath))
            loaded_methods = Serialise.load_spatial_methods(str(filepath))

            # Verify options are preserved
            assert loaded_methods["macroscale"].options == custom_options

    def test_spatial_methods_invalid_class(self):
        """Test error handling for invalid spatial method class."""
        # Create invalid spatial methods data
        invalid_data = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            "spatial_methods": {
                "macroscale": {
                    "class": "NonExistentMethod",
                    "module": "pybamm.spatial_methods.finite_volume",
                    "options": {},
                }
            },
        }

        with pytest.raises(ImportError):
            Serialise.load_spatial_methods(invalid_data)


class TestVarPtsSerialization:
    def test_serialise_and_load_var_pts(self):
        """Test saving and loading var_pts to/from file."""
        # Create var_pts with string keys
        var_pts = {
            "x_n": 20,
            "x_s": 25,
            "x_p": 30,
            "r_n": 15,
            "r_p": 15,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_var_pts.json"

            # Save var_pts
            Serialise.save_var_pts(var_pts, filename=str(filepath))
            assert filepath.exists()

            # Load var_pts
            loaded_var_pts = Serialise.load_var_pts(str(filepath))

            # Verify all keys and values match
            assert loaded_var_pts == var_pts

    def test_serialise_and_load_var_pts_dict(self):
        """Test serializing to dict and loading from dict."""
        var_pts = {"x_n": 20, "x_s": 25, "x_p": 30}

        # Serialize to dict
        var_pts_dict = Serialise.serialise_var_pts(var_pts)

        # Verify structure
        assert "schema_version" in var_pts_dict
        assert "pybamm_version" in var_pts_dict
        assert "var_pts" in var_pts_dict

        # Load from dict
        loaded_var_pts = Serialise.load_var_pts(var_pts_dict)

        # Verify match
        assert loaded_var_pts == var_pts

    def test_var_pts_with_spatial_variables(self):
        """Test var_pts with SpatialVariable keys."""
        # Create var_pts with SpatialVariable keys
        x_n = pybamm.SpatialVariable("x_n", domain="negative electrode")
        x_s = pybamm.SpatialVariable("x_s", domain="separator")

        var_pts = {
            x_n: 20,
            x_s: 25,
            "r_p": 15,  # Mix with string key
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_var_pts.json"

            # Save var_pts
            Serialise.save_var_pts(var_pts, filename=str(filepath))

            # Load var_pts (will have all string keys)
            loaded_var_pts = Serialise.load_var_pts(str(filepath))

            # Verify all keys are converted to strings
            expected = {"x_n": 20, "x_s": 25, "r_p": 15}
            assert loaded_var_pts == expected

    def test_var_pts_mixed_keys(self):
        """Test var_pts with both string and SpatialVariable keys."""
        x_n = pybamm.SpatialVariable("x_n", domain="negative electrode")
        var_pts = {
            x_n: 20,
            "x_s": 25,
            "x_p": 30,
        }

        # Serialize to dict
        var_pts_dict = Serialise.serialise_var_pts(var_pts)

        # All keys should be strings
        assert set(var_pts_dict["var_pts"].keys()) == {"x_n", "x_s", "x_p"}

    def test_var_pts_with_default_filename(self, monkeypatch):
        """Test var_pts saving with auto-generated filename."""
        var_pts = {"x_n": 20}

        with tempfile.TemporaryDirectory() as tmpdir:
            with monkeypatch.context() as m:
                m.chdir(tmpdir)

                # Save with no filename (auto-generate)
                Serialise.save_var_pts(var_pts)

                # Check a file was created
                json_files = list(Path(tmpdir).glob("var_pts_*.json"))
                assert len(json_files) == 1


class TestSubmeshTypesSerialization:
    def test_serialise_and_load_submesh_types(self):
        """Test saving and loading submesh types to/from file."""
        # Create submesh types dict
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "current collector": pybamm.MeshGenerator(pybamm.SubMesh0D),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_submesh_types.json"

            # Save submesh types
            Serialise.save_submesh_types(submesh_types, filename=str(filepath))
            assert filepath.exists()

            # Load submesh types
            loaded_submesh_types = Serialise.load_submesh_types(str(filepath))

            # Verify domains match
            assert set(loaded_submesh_types.keys()) == set(submesh_types.keys())

            # Verify class types match
            for domain in submesh_types.keys():
                assert isinstance(
                    loaded_submesh_types[domain], type(submesh_types[domain])
                )

    def test_serialise_and_load_submesh_types_dict(self):
        """Test serializing to dict and loading from dict."""
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        }

        # Serialize to dict
        submesh_dict = Serialise.serialise_submesh_types(submesh_types)

        # Verify structure
        assert "schema_version" in submesh_dict
        assert "pybamm_version" in submesh_dict
        assert "submesh_types" in submesh_dict

        # Load from dict
        loaded_submesh_types = Serialise.load_submesh_types(submesh_dict)

        # Verify domains match
        assert set(loaded_submesh_types.keys()) == set(submesh_types.keys())

    def test_submesh_types_with_default_filename(self, monkeypatch):
        """Test submesh types saving with auto-generated filename."""
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with monkeypatch.context() as m:
                m.chdir(tmpdir)

                # Save with no filename (auto-generate)
                Serialise.save_submesh_types(submesh_types)

                # Check a file was created
                json_files = list(Path(tmpdir).glob("submesh_types_*.json"))
                assert len(json_files) == 1

    def test_submesh_types_invalid_class(self):
        """Test error handling for invalid submesh type class."""
        # Create invalid submesh types data
        invalid_data = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            "submesh_types": {
                "negative electrode": {
                    "class": "NonExistentMesh",
                    "module": "pybamm.meshes.zero_dimensional_submesh",
                }
            },
        }

        with pytest.raises(ImportError):
            Serialise.load_submesh_types(invalid_data)

    def test_submesh_types_invalid_extension(self):
        """Test that non-.json extension raises error."""
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        }

        with pytest.raises(ValueError, match=r"must end with '\.json' extension"):
            Serialise.save_submesh_types(submesh_types, filename="test.txt")


class TestSerializationErrorHandling:
    def test_invalid_schema_version_geometry(self):
        """Test that invalid schema version raises error."""
        invalid_data = {
            "schema_version": "99.0",
            "pybamm_version": pybamm.__version__,
            "geometry": {},
        }

        with pytest.raises(ValueError, match=r"Unsupported schema version"):
            Serialise.load_custom_geometry(invalid_data)

    def test_missing_geometry_section(self):
        """Test error when geometry section is missing."""
        invalid_data = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
        }

        with pytest.raises(KeyError, match=r"Missing 'geometry' section"):
            Serialise.load_custom_geometry(invalid_data)

    def test_missing_spatial_methods_section(self):
        """Test error when spatial_methods section is missing."""
        invalid_data = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
        }

        with pytest.raises(KeyError, match=r"Missing 'spatial_methods' section"):
            Serialise.load_spatial_methods(invalid_data)

    def test_missing_var_pts_section(self):
        """Test error when var_pts section is missing."""
        invalid_data = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
        }

        with pytest.raises(KeyError, match=r"Missing 'var_pts' section"):
            Serialise.load_var_pts(invalid_data)

    def test_file_not_found_geometry(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            Serialise.load_custom_geometry("nonexistent_file.json")

    def test_file_not_found_spatial_methods(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            Serialise.load_spatial_methods("nonexistent_file.json")

    def test_file_not_found_var_pts(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            Serialise.load_var_pts("nonexistent_file.json")

    def test_missing_submesh_types_section(self):
        """Test error when submesh_types section is missing."""
        invalid_data = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
        }

        with pytest.raises(KeyError, match=r"Missing 'submesh_types' section"):
            Serialise.load_submesh_types(invalid_data)

    def test_file_not_found_submesh_types(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            Serialise.load_submesh_types("nonexistent_file.json")


class TestSerializationEdgeCases:
    """Tests for edge cases and uncovered code paths in serialization."""

    def test_geometry_with_symbol_keys(self):
        """Test geometry serialization with Symbol keys (not string keys)."""
        # Create a geometry with SpatialVariable as keys
        x_n = pybamm.SpatialVariable("x_n", domain="negative electrode")
        r_n = pybamm.SpatialVariable("r_n", domain="negative particle")

        # Create a custom geometry with Symbol keys
        geometry = pybamm.Geometry(
            {
                "negative electrode": {
                    x_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}
                },
                "negative particle": {
                    r_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}
                },
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "geometry_symbol_keys.json"

            # Save and load
            Serialise.save_custom_geometry(geometry, filename=str(filepath))
            loaded_geometry = Serialise.load_custom_geometry(str(filepath))

            # Verify domains exist
            assert "negative electrode" in loaded_geometry
            assert "negative particle" in loaded_geometry

            # Verify Symbol keys are reconstructed
            for domain in loaded_geometry:
                for key in loaded_geometry[domain].keys():
                    if isinstance(key, pybamm.Symbol):
                        assert hasattr(key, "name")

    def test_geometry_with_non_dict_value(self):
        """Test geometry with non-dict value for string key."""
        # Create a custom geometry with a non-dict value (primitive)
        geometry = pybamm.Geometry(
            {
                "current collector": {
                    "position": 1  # Non-dict value (primitive)
                }
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "geometry_non_dict.json"

            # Save and load
            Serialise.save_custom_geometry(geometry, filename=str(filepath))
            loaded_geometry = Serialise.load_custom_geometry(str(filepath))

            # Verify the geometry was saved and loaded
            assert "current collector" in loaded_geometry
            assert loaded_geometry["current collector"]["position"] == 1

    def test_geometry_file_write_error(self, monkeypatch):
        """Test OSError handling when writing geometry file."""
        geometry = pybamm.battery_geometry()

        # Mock open to raise OSError
        def mock_open_error(*args, **kwargs):
            raise OSError("Permission denied")

        monkeypatch.setattr("builtins.open", mock_open_error)

        with pytest.raises(ValueError, match=r"Failed to save custom geometry"):
            Serialise.save_custom_geometry(geometry, filename="test.json")

    def test_geometry_invalid_json(self):
        """Test JSONDecodeError when loading geometry with invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "invalid.json"

            # Write invalid JSON
            with open(filepath, "w") as f:
                f.write("{invalid json content")

            with pytest.raises(ValueError, match=r"contains invalid JSON"):
                Serialise.load_custom_geometry(str(filepath))

    def test_spatial_methods_invalid_filename_extension(self):
        """Test that non-.json filename raises error for spatial methods."""
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}

        with pytest.raises(ValueError, match=r"must end with '\.json' extension"):
            Serialise.save_spatial_methods(spatial_methods, filename="test.txt")

    def test_spatial_methods_file_write_error(self, monkeypatch):
        """Test OSError handling when writing spatial methods file."""
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}

        # Mock open to raise OSError
        def mock_open_error(*args, **kwargs):
            raise OSError("Disk full")

        monkeypatch.setattr("builtins.open", mock_open_error)

        with pytest.raises(ValueError, match=r"Failed to save spatial methods"):
            Serialise.save_spatial_methods(spatial_methods, filename="test.json")

    def test_spatial_methods_general_error(self, monkeypatch):
        """Test general exception handling in save_spatial_methods."""
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}

        # Mock serialise to raise an exception
        def mock_serialise_error(*args):
            raise RuntimeError("Serialization failed")

        monkeypatch.setattr(
            "pybamm.expression_tree.operations.serialise.Serialise.serialise_spatial_methods",
            mock_serialise_error,
        )

        with pytest.raises(ValueError, match=r"Failed to save spatial methods"):
            Serialise.save_spatial_methods(spatial_methods)

    def test_var_pts_invalid_filename_extension(self):
        """Test that non-.json filename raises error for var_pts."""
        var_pts = {"x_n": 20}

        with pytest.raises(ValueError, match=r"must end with '\.json' extension"):
            Serialise.save_var_pts(var_pts, filename="test.txt")

    def test_var_pts_file_write_error(self, monkeypatch):
        """Test OSError handling when writing var_pts file."""
        var_pts = {"x_n": 20}

        # Mock open to raise OSError
        def mock_open_error(*args, **kwargs):
            raise OSError("Write failed")

        monkeypatch.setattr("builtins.open", mock_open_error)

        with pytest.raises(ValueError, match=r"Failed to save var_pts"):
            Serialise.save_var_pts(var_pts, filename="test.json")

    def test_var_pts_general_error(self, monkeypatch):
        """Test general exception handling in save_var_pts."""
        var_pts = {"x_n": 20}

        # Mock serialise to raise an exception
        def mock_serialise_error(*args):
            raise RuntimeError("Serialization failed")

        monkeypatch.setattr(
            "pybamm.expression_tree.operations.serialise.Serialise.serialise_var_pts",
            mock_serialise_error,
        )

        with pytest.raises(ValueError, match=r"Failed to save var_pts"):
            Serialise.save_var_pts(var_pts)

    def test_submesh_types_file_write_error(self, monkeypatch):
        """Test OSError handling when writing submesh types file."""
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)
        }

        # Mock open to raise OSError
        def mock_open_error(*args, **kwargs):
            raise OSError("Write failed")

        monkeypatch.setattr("builtins.open", mock_open_error)

        with pytest.raises(ValueError, match=r"Failed to save submesh types"):
            Serialise.save_submesh_types(submesh_types, filename="test.json")

    def test_submesh_types_general_error(self, monkeypatch):
        """Test general exception handling in save_submesh_types."""
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)
        }

        # Mock serialise to raise an exception
        def mock_serialise_error(*args):
            raise RuntimeError("Serialization failed")

        monkeypatch.setattr(
            "pybamm.expression_tree.operations.serialise.Serialise.serialise_submesh_types",
            mock_serialise_error,
        )

        with pytest.raises(ValueError, match=r"Failed to save submesh types"):
            Serialise.save_submesh_types(submesh_types)

    def test_geometry_general_error(self, monkeypatch):
        """Test general exception handling in save_custom_geometry."""
        geometry = pybamm.battery_geometry()

        # Mock serialise to raise an exception
        def mock_serialise_error(*args):
            raise RuntimeError("Geometry serialization failed")

        monkeypatch.setattr(
            "pybamm.expression_tree.operations.serialise.Serialise.serialise_custom_geometry",
            mock_serialise_error,
        )

        with pytest.raises(ValueError, match=r"Failed to save custom geometry"):
            Serialise.save_custom_geometry(geometry)

    def test_spatial_methods_default_filename(self, monkeypatch):
        """Test spatial methods with auto-generated filename."""
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}

        with tempfile.TemporaryDirectory() as tmpdir:
            with monkeypatch.context() as m:
                m.chdir(tmpdir)

                # Save with no filename (auto-generate)
                Serialise.save_spatial_methods(spatial_methods)

                # Check a file was created
                json_files = list(Path(tmpdir).glob("spatial_methods_*.json"))
                assert len(json_files) == 1

    def test_geometry_with_non_symbol_values_in_symbol_key(self):
        """Test geometry with non-Symbol values nested in Symbol-keyed dict."""
        x_n = pybamm.SpatialVariable("x_n", domain="negative electrode")

        # Create geometry with Symbol key but non-Symbol value
        geometry = pybamm.Geometry({"negative electrode": {x_n: {"min": 0, "max": 1}}})

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "geometry_mixed.json"

            # Save and load
            Serialise.save_custom_geometry(geometry, filename=str(filepath))
            loaded_geometry = Serialise.load_custom_geometry(str(filepath))

            # Verify the geometry was saved and loaded
            assert "negative electrode" in loaded_geometry

    def test_geometry_reconstruction_non_symbol_value_in_reconstructed(self):
        """Test geometry loading with non-Symbol values that remain non-Symbol."""
        x_n = pybamm.SpatialVariable("x_n", domain="negative electrode")

        # Create a more complex geometry
        geometry = pybamm.Geometry(
            {
                "negative electrode": {
                    x_n: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.Scalar(1),
                        "tabs": {"negative": {"z_centre": 0.5}},
                    }
                }
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "geometry_tabs.json"

            # Save and load
            Serialise.save_custom_geometry(geometry, filename=str(filepath))
            loaded_geometry = Serialise.load_custom_geometry(str(filepath))

            # Verify the geometry was saved and loaded
            assert "negative electrode" in loaded_geometry

    def test_spatial_methods_invalid_json(self):
        """Test JSONDecodeError when loading spatial methods with invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "invalid_spatial.json"

            # Write invalid JSON
            with open(filepath, "w") as f:
                f.write("{invalid json for spatial methods")

            with pytest.raises(ValueError, match=r"contains invalid JSON"):
                Serialise.load_spatial_methods(str(filepath))

    def test_spatial_methods_unsupported_schema(self):
        """Test unsupported schema version for spatial methods."""
        invalid_data = {
            "schema_version": "999.0",
            "pybamm_version": "1.0.0",
            "spatial_methods": {},
        }

        with pytest.raises(ValueError, match=r"Unsupported schema version"):
            Serialise.load_spatial_methods(invalid_data)

    def test_spatial_methods_import_error(self):
        """Test import error handling in spatial methods loading."""
        invalid_data = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            "spatial_methods": {
                "domain": {
                    "class": "NonExistentClass",
                    "module": "pybamm.spatial_methods.nonexistent",
                    "options": {},
                }
            },
        }

        with pytest.raises(ImportError, match=r"Could not import spatial method"):
            Serialise.load_spatial_methods(invalid_data)

    def test_var_pts_invalid_json(self):
        """Test JSONDecodeError when loading var_pts with invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "invalid_var_pts.json"

            # Write invalid JSON
            with open(filepath, "w") as f:
                f.write("{invalid json for var_pts")

            with pytest.raises(ValueError, match=r"contains invalid JSON"):
                Serialise.load_var_pts(str(filepath))

    def test_var_pts_unsupported_schema(self):
        """Test unsupported schema version for var_pts."""
        invalid_data = {
            "schema_version": "999.0",
            "pybamm_version": "1.0.0",
            "var_pts": {},
        }

        with pytest.raises(ValueError, match=r"Unsupported schema version"):
            Serialise.load_var_pts(invalid_data)

    def test_var_pts_unexpected_key_type(self):
        """Test ValueError for unexpected key type in var_pts."""
        # Create var_pts with an unexpected key type
        var_pts = {123: 20}  # integer key instead of string or SpatialVariable

        with pytest.raises(ValueError, match=r"Unexpected key type in var_pts"):
            Serialise.serialise_var_pts(var_pts)

    def test_submesh_types_without_mesh_generator(self):
        """Test submesh types serialization without MeshGenerator wrapper."""
        # Directly use submesh class without MeshGenerator
        submesh_types = {"negative electrode": pybamm.Uniform1DSubMesh}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "submesh_no_wrapper.json"

            # This should work - the code handles both cases
            Serialise.save_submesh_types(submesh_types, filename=str(filepath))
            loaded = Serialise.load_submesh_types(str(filepath))

            # Verify it was wrapped in MeshGenerator on load
            assert "negative electrode" in loaded

    def test_submesh_types_invalid_json(self):
        """Test JSONDecodeError when loading submesh types with invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "invalid_submesh.json"

            # Write invalid JSON
            with open(filepath, "w") as f:
                f.write("{invalid json for submesh types")

            with pytest.raises(ValueError, match=r"contains invalid JSON"):
                Serialise.load_submesh_types(str(filepath))

    def test_submesh_types_unsupported_schema(self):
        """Test unsupported schema version for submesh types."""
        invalid_data = {
            "schema_version": "999.0",
            "pybamm_version": "1.0.0",
            "submesh_types": {},
        }

        with pytest.raises(ValueError, match=r"Unsupported schema version"):
            Serialise.load_submesh_types(invalid_data)

    @pytest.mark.parametrize(
        "compress", [False, True], ids=["uncompressed", "compressed"]
    )
    def test_load_custom_model_from_dict(self, compress):
        """Test loading a custom model directly from a dictionary."""
        # Create and save a custom model
        model = pybamm.BaseModel(name="test_dict_model")
        a = pybamm.Variable("a", domain="electrode")
        b = pybamm.Variable("b", domain="electrode")
        model.rhs = {a: b}
        model.initial_conditions = {a: pybamm.Scalar(1)}
        model.algebraic = {}
        model.boundary_conditions = {a: {"left": (pybamm.Scalar(0), "Dirichlet")}}
        model.events = [pybamm.Event("terminal", pybamm.Scalar(1) - b, "TERMINATION")]
        model.variables = {"a": a, "b": b}

        model_json = Serialise.serialise_custom_model(model, compress=compress)

        # Load from dict directly
        loaded_model = Serialise.load_custom_model(model_json)

        # Verify it loaded correctly
        assert loaded_model.name == "test_dict_model"
        assert isinstance(loaded_model.rhs, dict)

    def test_expression_function_parameter_evaluate(self):
        """Test _unary_evaluate method of ExpressionFunctionParameter."""
        x = pybamm.Variable("x")
        expr = x + pybamm.Parameter("c")
        efp = ExpressionFunctionParameter("f", expr, "f", ["x"])

        # Test _unary_evaluate
        result = efp._unary_evaluate(pybamm.Scalar(5))
        assert isinstance(result, pybamm.Scalar)
        assert result.value == 5

    def test_load_model_from_dict(self):
        """Test loading a discretised model from a dictionary instead of a file."""
        model = pybamm.lithium_ion.SPM(name="test_spm_dict")
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # Serialize to dict
        model_dict = Serialise().serialise_model(model)

        # Load from dict
        loaded_model = Serialise().load_model(model_dict)

        # Check that it solves
        solver = loaded_model.default_solver
        solution = solver.solve(loaded_model, [0, 100])
        assert solution.t[-1] == 100

    def test_load_model_with_battery_model_param(self):
        """Test loading a model with battery_model parameter specified."""
        model = pybamm.lithium_ion.SPM(name="test_spm_param")
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # Serialize to file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = (
                Path(tmpdir) / "test_model"
            )  # Don't include .json, save_model adds it
            Serialise().save_model(model, filename=str(filepath))

            # Load with battery_model parameter
            loaded_model = Serialise().load_model(
                str(filepath) + ".json", battery_model=pybamm.lithium_ion.SPM()
            )

            # Check that it solves
            solver = loaded_model.default_solver
            solution = solver.solve(loaded_model, [0, 100])
            assert solution.t[-1] == 100

    def test_load_spatial_methods_general_exception(self):
        """Test general exception handling in load_spatial_methods."""
        invalid_data = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            "spatial_methods": {
                "domain": {
                    "class": "FiniteVolume",
                    "module": "pybamm.spatial_methods.finite_volume",
                    "options": "invalid_options",  # This will cause an exception during instantiation
                }
            },
        }

        # This should raise ImportError because AttributeError is caught and converted
        with pytest.raises(ImportError, match=r"Could not import spatial method"):
            Serialise.load_spatial_methods(invalid_data)

    def test_load_submesh_types_general_exception(self):
        """Test general exception handling in load_submesh_types."""
        invalid_data = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            "submesh_types": {
                "domain": {
                    # Missing 'class' key will cause KeyError
                    "module": "pybamm.meshes.one_dimensional_submeshes",
                }
            },
        }

        with pytest.raises(ValueError, match=r"Failed to reconstruct submesh type"):
            Serialise.load_submesh_types(invalid_data)

    def test_load_custom_model_missing_model_section(self, tmp_path):
        """Test that missing 'model' section raises KeyError."""
        model_json = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            # Missing 'model' key
        }

        file_path = tmp_path / "no_model.json"
        with open(file_path, "w") as f:
            json.dump(model_json, f)

        with pytest.raises(KeyError, match=r"Missing 'model' section"):
            Serialise.load_custom_model(str(file_path))

    def test_load_custom_model_empty_base_class(self, tmp_path):
        """Test loading custom model with empty base class (should use pybamm.BaseModel)."""
        model_json = {
            "schema_version": "1.1",
            "pybamm_version": pybamm.__version__,
            "model": {
                "name": "TestModel",
                "base_class": "",  # Empty string should trigger pybamm.BaseModel
                "options": {},
                "rhs": [],
                "algebraic": [],
                "initial_conditions": [],
                "boundary_conditions": [],
                "events": [],
                "variables": {},
            },
        }

        file_path = tmp_path / "empty_base.json"
        with open(file_path, "w") as f:
            json.dump(model_json, f)

        # Should load successfully using pybamm.BaseModel
        loaded_model = Serialise.load_custom_model(str(file_path))
        assert isinstance(loaded_model, pybamm.BaseModel)
        assert loaded_model.name == "TestModel"

    def test_load_parameters_with_string_values(self, tmp_path):
        """Test load_parameters with numeric string values (converted to float)."""
        params = {
            "param1": "3.14",  # Numeric string will be converted to float
            "param2": 42,
        }

        file_path = tmp_path / "params_string.json"
        with open(file_path, "w") as f:
            json.dump(params, f)

        loaded = Serialise.load_parameters(str(file_path))
        assert loaded["param1"] == 3.14  # String was converted to float
        assert loaded["param2"] == 42

    def test_load_parameters_with_dict_values(self, tmp_path):
        """Test load_parameters with dict values (not containing 'type')."""
        params = {
            "param1": {"key1": "value1", "key2": "value2"},
            "param2": 42,
        }

        file_path = tmp_path / "params_dict.json"
        with open(file_path, "w") as f:
            json.dump(params, f)

        loaded = Serialise.load_parameters(str(file_path))
        assert loaded["param1"] == {"key1": "value1", "key2": "value2"}
        assert loaded["param2"] == 42

    def test_load_parameters_unsupported_format(self, tmp_path):
        """Test load_parameters with unsupported parameter format."""
        params = {
            "param1": None,  # None is unsupported
        }

        file_path = tmp_path / "params_unsupported.json"
        with open(file_path, "w") as f:
            json.dump(params, f)

        with pytest.raises(ValueError, match=r"Unsupported parameter format"):
            Serialise.load_parameters(str(file_path))

    def test_load_compressed_model_with_corrupted_data(self, tmp_path):
        """Test that corrupted compressed data raises ValueError."""
        # Create a file with corrupted compressed data
        corrupted_data = {
            "compressed": True,
            "data": "this_is_not_valid_base64_zlib_data!!!",
        }

        file_path = tmp_path / "corrupted_compressed.json"
        with open(file_path, "w") as f:
            json.dump(corrupted_data, f)

        with pytest.raises(ValueError, match=r"Failed to decompress model data"):
            Serialise.load_custom_model(str(file_path))

    def test_compression_reduces_size(self):
        """Test that compression actually reduces the serialized size."""
        model = BasicDFN()

        uncompressed = Serialise.serialise_custom_model(model, compress=False)
        compressed = Serialise.serialise_custom_model(model, compress=True)

        # Convert to JSON strings to compare sizes
        uncompressed_str = json.dumps(uncompressed, default=Serialise._json_encoder)
        compressed_str = json.dumps(compressed)

        # Compressed version should be smaller
        assert len(compressed_str) < len(uncompressed_str)

        # Compressed version should have the marker
        assert compressed.get("compressed") is True
        assert "data" in compressed

    def test_compressed_format_structure(self):
        """Test that compressed output has the expected structure."""
        model = pybamm.BaseModel(name="test")
        a = pybamm.Variable("a")
        model.rhs = {a: pybamm.Scalar(1)}
        model.initial_conditions = {a: pybamm.Scalar(0)}
        model.algebraic = {}
        model.boundary_conditions = {}
        model.events = []
        model.variables = {"a": a}

        compressed = Serialise.serialise_custom_model(model, compress=True)

        # Check structure
        assert set(compressed.keys()) == {"compressed", "data"}
        assert compressed["compressed"] is True
        assert isinstance(compressed["data"], str)

        # Verify the data can be decompressed
        loaded = Serialise.load_custom_model(compressed)
        assert loaded.name == "test"
