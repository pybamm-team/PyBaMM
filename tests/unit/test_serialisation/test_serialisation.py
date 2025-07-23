#
# Tests for the serialisation class
#

import json
import os
import re
from datetime import datetime

import numpy as np
import pytest
from numpy import testing

import pybamm
from pybamm.expression_tree.operations.serialise import (
    SUPPORTED_SCHEMA_VERSION,
    Serialise,
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

        model.save_model("heat_equation", variables=model._variables, mesh=mesh)
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
            AttributeError, match="'Mesh' object has no attribute '_geometry'"
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

    def test_save_load_model(self):
        model = pybamm.lithium_ion.SPM(name="test_spm")
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

        # test error if not discretised
        with pytest.raises(
            NotImplementedError,
            match="PyBaMM can only serialise a discretised, ready-to-solve model",
        ):
            Serialise().save_model(model, filename="test_model")

        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # default save
        Serialise().save_model(model, filename="test_model")
        assert os.path.exists("test_model.json")

        # default save where filename isn't provided
        Serialise().save_model(model)
        filename = "test_spm_" + datetime.now().strftime("%Y_%m_%d-%p%I_%M") + ".json"
        assert os.path.exists(filename)
        os.remove(filename)

        # default load
        new_model = Serialise().load_model("test_model.json")

        # check new model solves
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, [0, 3600])

        # check an error is raised when plotting the solution
        with pytest.raises(
            AttributeError,
            match="No variables to plot",
        ):
            new_solution.plot()

        # load when specifying the battery model to use
        newest_model = Serialise().load_model(
            "test_model.json", battery_model=pybamm.lithium_ion.SPM
        )

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
            match="Serialising models coupled to experiments is not yet supported.",
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

        Serialise().save_model(
            model,
            variables=model.variables,
            filename="test_base_model",
        )

        new_model = Serialise().load_model("test_base_model.json")
        os.remove("test_base_model.json")

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
            variables=model.variables,
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
        j = Serialise.convert_symbol_to_json(S)
        S2 = Serialise.convert_symbol_from_json(j)
        assert isinstance(S2, pybamm.Scalar)
        assert S2.value == pytest.approx(2.718)

    def test_serialise_time(self):
        t = pybamm.Time()
        j = Serialise.convert_symbol_to_json(t)
        t2 = Serialise.convert_symbol_from_json(j)
        assert isinstance(t2, pybamm.Time)

    def test_convert_symbol_to_json_with_number_and_list(self):
        for val in (0, 3.14, -7, True):
            out = Serialise.convert_symbol_to_json(val)
            assert out is val or out == val

        sample = [1, 2, 3, "foo", 4.5]
        out = Serialise.convert_symbol_to_json(sample)
        assert out is sample

    def test_convert_symbol_from_json_with_primitives(self):
        assert Serialise.convert_symbol_from_json(3.14) == 3.14
        assert Serialise.convert_symbol_from_json(42) == 42
        assert Serialise.convert_symbol_from_json(True) is True

    def test_convert_symbol_from_json_with_none(self):
        assert Serialise.convert_symbol_from_json(None) is None

    def test_convert_symbol_from_json_unexpected_string(self):
        with pytest.raises(ValueError, match=r"Unexpected raw string in JSON: foo"):
            Serialise.convert_symbol_from_json("foo")

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

        with pytest.raises(TypeError, match="is not JSON serializable"):
            Serialise._json_encoder(Dummy())

    def test_create_symbol_key(self):
        var1 = pybamm.Variable("x", bounds=(0, 1))
        var2 = pybamm.Variable("x", bounds=(0, 2))

        json1 = Serialise.convert_symbol_to_json(var1)
        json2 = Serialise.convert_symbol_to_json(var2)

        key1 = Serialise._create_symbol_key(json1)
        key2 = Serialise._create_symbol_key(json2)

        assert isinstance(key1, str)
        assert isinstance(key2, str)
        assert key1 != key2

    def test_primary_broadcast_serialisation(self):
        child = pybamm.Scalar(42)
        symbol = pybamm.PrimaryBroadcast(child, "negative electrode")
        json_dict = Serialise.convert_symbol_to_json(symbol)
        symbol2 = Serialise.convert_symbol_from_json(json_dict)

        assert isinstance(symbol2, pybamm.PrimaryBroadcast)
        assert symbol2.broadcast_domain == ["negative electrode"]
        assert isinstance(symbol2.orphans[0], pybamm.Scalar)
        assert symbol2.orphans[0].value == 42

    def test_interpolant_serialisation(self):
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

    def test_variable_serialisation(self):
        var = pybamm.Variable("var", domain="separator")
        json_dict = Serialise.convert_symbol_to_json(var)
        var2 = Serialise.convert_symbol_from_json(json_dict)

        assert isinstance(var2, pybamm.Variable)
        assert var2.name == "var"
        assert var2.domains["primary"] == ["separator"]
        assert var2.bounds[0].value == -float("inf")
        assert var2.bounds[1].value == float("inf")

    def test_concatenation_variable_serialisation(self):
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
        json_dict = Serialise.convert_symbol_to_json(fb)
        fb2 = Serialise.convert_symbol_from_json(json_dict)

        assert isinstance(fb2, pybamm.FullBroadcast)
        assert fb2.broadcast_domain == ["negative electrode"]
        assert fb2.domains["primary"] == ["negative electrode"]
        assert fb2.domains["secondary"] == ["current collector"]
        assert isinstance(fb2.child, pybamm.Scalar)
        assert fb2.child.value == 5

    def test_secondary_broadcast_serialisation(self):
        child = pybamm.Variable("c", domain="negative electrode")
        sb = pybamm.SecondaryBroadcast(child, "current collector")

        json_dict = Serialise.convert_symbol_to_json(sb)
        sb2 = Serialise.convert_symbol_from_json(json_dict)

        assert isinstance(sb2, pybamm.SecondaryBroadcast)
        assert sb2.broadcast_domain == ["current collector"]
        assert sb2.child.name == "c"
        assert sb2.child.domain == ["negative electrode"]

    def test_spatial_variable_serialisation(self):
        sv = pybamm.SpatialVariable(
            "x", domain="negative electrode", coord_sys="cartesian"
        )
        json_dict = Serialise.convert_symbol_to_json(sv)
        sv2 = Serialise.convert_symbol_from_json(json_dict)

        assert isinstance(sv2, pybamm.SpatialVariable)
        assert sv2.name == "x"
        assert sv2.domains["primary"] == ["negative electrode"]
        assert sv2.coord_sys == "cartesian"

    def test_boundary_value_serialisation(self):
        var = pybamm.SpatialVariable("x", domain="electrode")
        bv = pybamm.BoundaryValue(var, "left")
        json_dict = Serialise.convert_symbol_to_json(bv)
        bv2 = Serialise.convert_symbol_from_json(json_dict)

        assert isinstance(bv2, pybamm.BoundaryValue)
        assert bv2.side == "left"
        assert isinstance(bv2.orphans[0], pybamm.SpatialVariable)
        assert bv2.orphans[0].name == "x"

    def test_specific_function_not_supported(self):
        def dummy_func(x):
            return x

        symbol = pybamm.SpecificFunction(dummy_func, pybamm.Scalar(1))
        with pytest.raises(
            NotImplementedError, match="SpecificFunction is not supported directly"
        ):
            Serialise.convert_symbol_to_json(symbol)

    def test_unary_operator_serialisation(self):
        expr = pybamm.Negate(pybamm.Scalar(5))
        json_dict = Serialise.convert_symbol_to_json(expr)
        expr2 = Serialise.convert_symbol_from_json(json_dict)

        assert isinstance(expr2, pybamm.Negate)
        assert isinstance(expr2.child, pybamm.Scalar)
        assert expr2.child.value == 5

    def test_binary_operator_serialisation(self):
        expr = pybamm.Addition(pybamm.Scalar(2), pybamm.Scalar(3))
        json_dict = Serialise.convert_symbol_to_json(expr)
        expr2 = Serialise.convert_symbol_from_json(json_dict)

        assert isinstance(expr2, pybamm.Addition)
        values = [c.value for c in expr2.children]
        assert values == [2, 3]

    def test_function_parameter_with_diff_variable_serialisation(self):
        x = pybamm.Variable("x")
        diff_var = pybamm.Variable("r")
        func_param = pybamm.FunctionParameter("my_func", {"x": x}, diff_var)

        json_dict = Serialise.convert_symbol_to_json(func_param)
        assert "diff_variable" in json_dict
        assert json_dict["diff_variable"]["type"] == "Variable"
        assert json_dict["diff_variable"]["name"] == "r"

        expr2 = Serialise.convert_symbol_from_json(json_dict)
        assert isinstance(expr2, pybamm.FunctionParameter)
        assert expr2.diff_variable.name == "r"
        assert expr2.name == "my_func"
        assert list(expr2.input_names) == ["x"]

    def test_indefinite_integral_serialisation(self):
        x = pybamm.SpatialVariable("x", domain="negative electrode")
        ind_int = pybamm.IndefiniteIntegral(x, x)

        json_dict = Serialise.convert_symbol_to_json(ind_int)
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

        expr2 = Serialise.convert_symbol_from_json(json_dict)
        assert isinstance(expr2, pybamm.IndefiniteIntegral)
        assert isinstance(expr2.child, pybamm.SpatialVariable)

        assert expr2.child.name == "x"
        assert isinstance(expr2.integration_variable, list)
        assert len(expr2.integration_variable) == 1
        assert isinstance(expr2.integration_variable[0], pybamm.SpatialVariable)
        assert expr2.integration_variable[0].name == "x"

    def test_symbol_fallback_serialisation(self):
        var = pybamm.Variable("v", domain="electrode")
        diff = pybamm.Gradient(var)
        json_dict = Serialise.convert_symbol_to_json(diff)
        diff2 = Serialise.convert_symbol_from_json(json_dict)

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
            Serialise.convert_symbol_to_json(dummy)

        assert "Error processing 'not_a_symbol'. Unknown symbol type:" in str(e.value)

    def test_deserialising_unhandled_type(self):
        unhandled_json = {"type": "NotARealSymbol", "foo": "bar"}
        with pytest.raises(
            ValueError,
            match=r"Unhandled symbol type or malformed entry: .*NotARealSymbol",
        ):
            Serialise.convert_symbol_from_json(unhandled_json)

        unhandled_json2 = {"a": 1, "b": 2}
        with pytest.raises(
            ValueError, match=r"Unhandled symbol type or malformed entry: .*"
        ):
            Serialise.convert_symbol_from_json(unhandled_json2)

    def test_unsupported_schema_version(self):
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

        file = "model.json"

        with open(file, "w") as f:
            json.dump(unhandled_schema_json, f)

        try:
            with pytest.raises(ValueError, match="Unsupported schema version: 9.9"):
                Serialise.load_custom_model(file, battery_model=pybamm.BaseModel())
        finally:
            os.remove(file)

    def test_model_has_correct_schema_version(self):
        model = BasicDFN()
        filename = "test_scehma_version"

        Serialise.save_custom_model(model, filename=filename)
        loaded_model = Serialise.load_custom_model(
            f"{filename}.json", battery_model=pybamm.lithium_ion.BaseModel()
        )

        try:
            assert hasattr(loaded_model, "schema_version")
            assert loaded_model.schema_version == SUPPORTED_SCHEMA_VERSION
        finally:
            # Clean up
            os.remove(f"{filename}.json")

    def test_save_and_load_custom_model(self):
        model = pybamm.BaseModel(name="test_model")
        a = pybamm.Variable("a", domain="electrode")
        b = pybamm.Variable("b", domain="electrode")
        model.rhs = {a: b}
        model.initial_conditions = {a: pybamm.Scalar(1)}
        model.algebraic = {}
        model.boundary_conditions = {a: {"left": (pybamm.Scalar(0), "Dirichlet")}}
        model.events = [pybamm.Event("terminal", pybamm.Scalar(1) - b, "TERMINATION")]
        model.variables = {"a": a, "b": b}

        # save model
        Serialise.save_custom_model(model, filename="test_model")

        # check json exists
        assert os.path.exists("test_model.json")

        # saving with defualt filename
        Serialise().save_custom_model(model)
        pattern = r"test_model_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}\.json"
        matched = [f for f in os.listdir(".") if re.fullmatch(pattern, f)]
        assert matched

        for f in matched:
            os.remove(f)

        # load model
        loaded_model = Serialise.load_custom_model("test_model.json")
        os.remove("test_model.json")

        assert loaded_model.name == "test_model"
        assert isinstance(loaded_model.rhs, dict)
        assert next(iter(loaded_model.rhs.keys())).name == "a"
        assert next(iter(loaded_model.rhs.values())).name == "b"

    def test_plotting_serialised_models(self):
        models = [
            BasicSPM(),
            BasicDFN(),
            pybamm.lithium_ion.SPM(),
            pybamm.lithium_ion.DFN(),
        ]
        filenames = ["basic_spm", "basic_dfn", "spm", "dfn"]

        for model, name in zip(models, filenames, strict=True):
            # Save the model
            Serialise.save_custom_model(model, filename=name)

            # Load the model
            loaded_model = Serialise.load_custom_model(
                f"{name}.json", battery_model=pybamm.lithium_ion.BaseModel()
            )

            sim = pybamm.Simulation(loaded_model)
            sim.solve([0, 3600])
            sim.plot(show_plot=False)

            os.remove(f"{name}.json")
