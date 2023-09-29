#
# Tests for the serialisation class
#
from tests import TestCase
import tests
import os
import unittest
import unittest.mock as mock
from datetime import datetime
import numpy as np
import pybamm

from numpy import testing
from pybamm.expression_tree.operations.serialise import Serialise


def scalar_var_dict():
    """variable, json pair for a pybamm.Scalar instance"""
    a = pybamm.Scalar(5)
    a_dict = {
        "py/id": mock.ANY,
        "py/object": "pybamm.expression_tree.scalar.Scalar",
        "name": "5.0",
        "id": mock.ANY,
        "value": 5.0,
        "children": [],
    }

    return a, a_dict


def mesh_var_dict():
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
        "py/id": mock.ANY,
        "submesh_pts": {"negative particle": {"r": 20}},
        "base_domains": ["negative particle"],
        "sub_meshes": {
            "negative particle": {
                "py/object": "pybamm.meshes.one_dimensional_submeshes.Uniform1DSubMesh",
                "py/id": mock.ANY,
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


class TestSerialiseModels(TestCase):
    # test lithium models
    def test_spm_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_equal(solution.all_ys[x], new_solution.all_ys[x])

    def test_spme_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.lithium_ion.SPMe()
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_equal(solution.all_ys[x], new_solution.all_ys[x])

    def test_mpm_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.lithium_ion.MPM()
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_almost_equal(
                solution.all_ys[x], new_solution.all_ys[x]
            )

    def test_dfn_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.lithium_ion.DFN()
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_almost_equal(
                solution.all_ys[x], new_solution.all_ys[x]
            )

    def test_newman_tobias_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.lithium_ion.NewmanTobias()
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_almost_equal(
                solution.all_ys[x], new_solution.all_ys[x]
            )

    def test_msmr_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_almost_equal(
                solution.all_ys[x], new_solution.all_ys[x], decimal=3
            )

    # test lead-acid models
    def test_lead_acid_full_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.lead_acid.Full()
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_almost_equal(
                solution.all_ys[x], new_solution.all_ys[x]
            )

    def test_loqs_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.lead_acid.LOQS()
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_almost_equal(
                solution.all_ys[x], new_solution.all_ys[x]
            )

    def test_thevenin_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.equivalent_circuit.Thevenin()
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_almost_equal(
                solution.all_ys[x], new_solution.all_ys[x]
            )


class TestSerialise(TestCase):
    # test the symbol encoder

    def test_symbol_encoder_symbol(self):
        """test basic symbol encoder with & without children"""

        # without children
        a, a_dict = scalar_var_dict()

        a_ser_json = Serialise._SymbolEncoder().default(a)

        self.assertEqual(a_ser_json, a_dict)

        # with children
        add = pybamm.Addition(2, 4)
        add_json = {
            "py/id": mock.ANY,
            "py/object": "pybamm.expression_tree.binary_operators.Addition",
            "name": "+",
            "id": mock.ANY,
            "domains": {
                "primary": [],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
            "children": [
                {
                    "py/id": mock.ANY,
                    "py/object": "pybamm.expression_tree.scalar.Scalar",
                    "name": "2.0",
                    "id": mock.ANY,
                    "value": 2.0,
                    "children": [],
                },
                {
                    "py/id": mock.ANY,
                    "py/object": "pybamm.expression_tree.scalar.Scalar",
                    "name": "4.0",
                    "id": mock.ANY,
                    "value": 4.0,
                    "children": [],
                },
            ],
        }

        add_ser_json = Serialise._SymbolEncoder().default(add)

        self.assertEqual(add_ser_json, add_json)

    def test_symbol_encoder_explicitTimeIntegral(self):
        """test symbol encoder with initial conditions"""
        expr = pybamm.ExplicitTimeIntegral(pybamm.Scalar(5), pybamm.Scalar(1))

        expr_json = {
            "py/object": "pybamm.expression_tree.unary_operators.ExplicitTimeIntegral",
            "py/id": mock.ANY,
            "name": "explicit time integral",
            "id": mock.ANY,
            "children": [
                {
                    "py/object": "pybamm.expression_tree.scalar.Scalar",
                    "py/id": mock.ANY,
                    "name": "5.0",
                    "id": mock.ANY,
                    "value": 5.0,
                    "children": [],
                }
            ],
            "initial_condition": {
                "py/object": "pybamm.expression_tree.scalar.Scalar",
                "py/id": mock.ANY,
                "name": "1.0",
                "id": mock.ANY,
                "value": 1.0,
                "children": [],
            },
        }

        expr_ser_json = Serialise._SymbolEncoder().default(expr)

        self.assertEqual(expr_json, expr_ser_json)

    def test_symbol_encoder_event(self):
        """test symbol encoder with event"""

        expression = pybamm.Scalar(1)
        event = pybamm.Event("my event", expression)

        event_json = {
            "py/object": "pybamm.models.event.Event",
            "py/id": mock.ANY,
            "name": "my event",
            "event_type": ["EventType.TERMINATION", 0],
            "expression": {
                "py/object": "pybamm.expression_tree.scalar.Scalar",
                "py/id": mock.ANY,
                "name": "1.0",
                "id": mock.ANY,
                "value": 1.0,
                "children": [],
            },
        }

        event_ser_json = Serialise._SymbolEncoder().default(event)
        self.assertEqual(event_ser_json, event_json)

    # test the mesh encoder
    def test_mesh_encoder(self):
        mesh, mesh_json = mesh_var_dict()

        # serialise mesh
        mesh_ser_json = Serialise._MeshEncoder().default(mesh)

        self.assertEqual(mesh_ser_json, mesh_json)

    def test_deconstruct_pybamm_dicts(self):
        """tests serialisation of dictionaries with pybamm classes as keys"""

        x = pybamm.SpatialVariable("x", "negative electrode")

        test_dict = {"rod": {x: {"min": 0.0, "max": 2.0}}}

        ser_dict = {
            "rod": {
                "symbol_x": {
                    "py/object": "pybamm.expression_tree.independent_variable.SpatialVariable",
                    "py/id": mock.ANY,
                    "name": "x",
                    "id": mock.ANY,
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

        self.assertEqual(Serialise()._deconstruct_pybamm_dicts(test_dict), ser_dict)

    def test_get_pybamm_class(self):
        # symbol
        _, scalar_dict = scalar_var_dict()

        scalar_class = Serialise()._get_pybamm_class(scalar_dict)

        self.assertIsInstance(scalar_class, pybamm.Scalar)

        # mesh
        _, mesh_dict = mesh_var_dict()

        mesh_class = Serialise()._get_pybamm_class(mesh_dict)

        self.assertIsInstance(mesh_class, pybamm.Mesh)

    def test_reconstruct_symbol(self):
        scalar, scalar_dict = scalar_var_dict()

        new_scalar = Serialise()._reconstruct_symbol(scalar_dict)

        self.assertEqual(new_scalar, scalar)

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

        self.assertEqual(new_equation, equation)

    def test_reconstruct_mesh(self):
        mesh, mesh_dict = mesh_var_dict()

        new_mesh = Serialise()._reconstruct_mesh(mesh_dict)

        testing.assert_array_equal(
            new_mesh["negative particle"].edges, mesh["negative particle"].edges
        )
        testing.assert_array_equal(
            new_mesh["negative particle"].nodes, mesh["negative particle"].nodes
        )

        # reconstructed meshes are only used for plotting, geometry not reconstructed.
        with self.assertRaisesRegex(
            AttributeError, "'Mesh' object has no attribute '_geometry'"
        ):
            self.assertEqual(new_mesh.geometry, mesh.geometry)

    def test_reconstruct_pybamm_dict(self):
        x = pybamm.SpatialVariable("x", "negative electrode")

        test_dict = {"rod": {x: {"min": 0.0, "max": 2.0}}}

        ser_dict = {
            "rod": {
                "symbol_x": {
                    "py/object": "pybamm.expression_tree.independent_variable.SpatialVariable",
                    "py/id": mock.ANY,
                    "name": "x",
                    "id": mock.ANY,
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

        self.assertEqual(new_dict, test_dict)

    def test_save_load_model(self):
        model = pybamm.lithium_ion.SPM(name="test_spm")
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

        # test error if not discretised
        with self.assertRaisesRegex(
            NotImplementedError,
            "PyBaMM can only serialise a discretised, ready-to-solve model",
        ):
            Serialise().save_model(model, filename="test_model")

        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # default save
        Serialise().save_model(model, filename="test_model")
        self.assertTrue(os.path.exists("test_model.json"))

        # default save where filename isn't provided
        Serialise().save_model(model)
        filename = (
            "test_spm_" + datetime.now().strftime("%Y_%m_%d-%p%I_%M_%S") + ".json"
        )
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

        # default load
        new_model = Serialise().load_model("test_model.json")

        # check new model solves
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, [0, 3600])

        # check an error is raised when plotting the solution
        with self.assertRaisesRegex(
            AttributeError,
            "Variables not provided by the serialised model",
        ):
            new_solution.plot()

        # load when specifying the battery model to use
        newest_model = Serialise().load_model(
            "test_model.json", battery_model=pybamm.lithium_ion.SPM
        )
        os.remove("test_model.json")

        # check new model solves
        newest_solver = newest_model.default_solver
        newest_solution = newest_solver.solve(newest_model, [0, 3600])

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
        new_solution.plot(["c", "2c"], testing=True)

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
        new_solution.plot(testing=True)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
