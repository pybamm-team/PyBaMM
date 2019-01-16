#
# Tests for the base model class
#
import pybamm

import unittest


class TestBaseModel(unittest.TestCase):
    def test_rhs_set_get(self):
        model = pybamm.BaseModel()
        rhs = {"c": "alpha", "d": "beta"}
        model.rhs = rhs
        self.assertEqual(rhs, model.rhs)

    def test_initial_conditions_set_get(self):
        model = pybamm.BaseModel()
        initial_conditions = {"c0": "gamma", "d0": "delta"}
        model.initial_conditions = initial_conditions
        self.assertEqual(initial_conditions, model.initial_conditions)

    def test_boundary_conditions_set_get(self):
        model = pybamm.BaseModel()
        boundary_conditions = {"c_left": "epsilon", "c_right": "eta"}
        model.boundary_conditions = boundary_conditions
        self.assertEqual(boundary_conditions, model.boundary_conditions)

    def test_model_dict_behaviour(self):
        model = pybamm.BaseModel()
        rhs = {"c": "alpha", "d": "beta"}
        model.rhs = rhs
        self.assertEqual(model["c"], rhs["c"])
        self.assertEqual(model["c"], model.rhs["c"])
