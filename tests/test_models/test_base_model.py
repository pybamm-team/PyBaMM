#
# Tests for the base model class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestBaseModel(unittest.TestCase):
    def test_rhs_set_get(self):
        model = pybamm.BaseModel()
        rhs = {
            pybamm.Symbol("c"): pybamm.Symbol("alpha"),
            pybamm.Symbol("d"): pybamm.Symbol("beta"),
        }
        model.rhs = rhs
        self.assertEqual(rhs, model.rhs)
        # test domains
        rhs = {
            pybamm.Symbol("c", domain=["negative electrode"]): pybamm.Symbol(
                "alpha", domain=["negative electrode"]
            ),
            pybamm.Symbol("d", domain=["positive electrode"]): pybamm.Symbol(
                "beta", domain=["positive electrode"]
            ),
        }
        model.rhs = rhs
        self.assertEqual(rhs, model.rhs)
        # non-matching domains should fail
        with self.assertRaises(pybamm.DomainError):
            model.rhs = {
                pybamm.Symbol("c", domain=["positive electrode"]): pybamm.Symbol(
                    "alpha", domain=["negative electrode"]
                )
            }

    def test_algebraic_set_get(self):
        model = pybamm.BaseModel()
        algebraic = {pybamm.Symbol("b"): pybamm.Symbol("c") - pybamm.Symbol("a")}
        model.algebraic = algebraic
        self.assertEqual(algebraic, model.algebraic)

    def test_initial_conditions_set_get(self):
        model = pybamm.BaseModel()
        initial_conditions = {
            pybamm.Symbol("c0"): pybamm.Symbol("gamma"),
            pybamm.Symbol("d0"): pybamm.Symbol("delta"),
        }
        model.initial_conditions = initial_conditions
        self.assertEqual(initial_conditions, model.initial_conditions)

        # Test number input
        c0 = pybamm.Symbol("c0")
        model.initial_conditions = {c0: 34}
        self.assertIsInstance(model.initial_conditions[c0], pybamm.Scalar)
        self.assertEqual(model.initial_conditions[c0].value, 34)

        # non-matching domains should fail
        with self.assertRaises(pybamm.DomainError):
            model.initial_conditions = {
                pybamm.Symbol("c", domain=["positive electrode"]): pybamm.Symbol(
                    "alpha", domain=["negative electrode"]
                )
            }

    def test_boundary_conditions_set_get(self):
        model = pybamm.BaseModel()
        boundary_conditions = {"c": {"left": "epsilon", "right": "eta"}}
        model.boundary_conditions = boundary_conditions
        self.assertEqual(boundary_conditions, model.boundary_conditions)

        # Test number input
        c0 = pybamm.Symbol("c0")
        model.boundary_conditions = {c0: {"left": -2, "right": 4}}
        self.assertIsInstance(model.boundary_conditions[c0]["left"], pybamm.Scalar)
        self.assertIsInstance(model.boundary_conditions[c0]["right"], pybamm.Scalar)
        self.assertEqual(model.boundary_conditions[c0]["left"].value, -2)
        self.assertEqual(model.boundary_conditions[c0]["right"].value, 4)

    def test_variables_set_get(self):
        model = pybamm.BaseModel()
        variables = {"c": "alpha", "d": "beta"}
        model.variables = variables
        self.assertEqual(variables, model.variables)

    def test_model_dict_behaviour(self):
        model = pybamm.BaseModel()
        key = pybamm.Symbol("c")
        rhs = {key: pybamm.Symbol("alpha")}
        model.rhs = rhs
        self.assertEqual(model[key], rhs[key])
        self.assertEqual(model[key], model.rhs[key])

    def test_update(self):
        # model
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        model = pybamm.BaseModel()
        c = pybamm.Variable("c", domain=whole_cell)
        rhs = {c: 5 * pybamm.div(pybamm.grad(c)) - 1}
        initial_conditions = {c: 1}
        boundary_conditions = {c: {"left": 0, "right": 0}}
        variables = {"c": c}
        model.rhs = rhs
        model.initial_conditions = initial_conditions
        model.boundary_conditions = boundary_conditions
        model.variables = variables

        # update with submodel
        submodel = pybamm.BaseModel()
        d = pybamm.Variable("d", domain=whole_cell)
        submodel.rhs = {
            d: 5 * pybamm.div(pybamm.grad(c)) + pybamm.div(pybamm.grad(d)) - 1
        }
        submodel.initial_conditions = {d: 3}
        submodel.boundary_conditions = {d: {"left": 4, "right": 7}}
        submodel.variables = {"d": d}
        model.update(submodel)

        # check
        self.assertEqual(model.rhs[d], submodel.rhs[d])
        self.assertEqual(model.initial_conditions[d], submodel.initial_conditions[d])
        self.assertEqual(model.boundary_conditions[d], submodel.boundary_conditions[d])
        self.assertEqual(model.variables["d"], submodel.variables["d"])
        self.assertEqual(model.rhs[c], rhs[c])
        self.assertEqual(model.initial_conditions[c], initial_conditions[c])
        self.assertEqual(model.boundary_conditions[c], boundary_conditions[c])
        self.assertEqual(model.variables["c"], variables["c"])

        # update with conflicting submodel
        submodel2 = pybamm.BaseModel()
        submodel2.rhs = {d: pybamm.div(pybamm.grad(d)) - 1}
        with self.assertRaises(AssertionError) as error:
            model.update(submodel2)
        self.assertIsInstance(error.exception.args[0], pybamm.ModelError)

        # update with multiple submodels
        submodel1 = submodel  # copy submodel from previous test
        submodel2 = pybamm.BaseModel()
        e = pybamm.Variable("e", domain=whole_cell)
        submodel2.rhs = {
            e: 5 * pybamm.div(pybamm.grad(d)) + pybamm.div(pybamm.grad(e)) - 1
        }
        submodel2.initial_conditions = {e: 3}
        submodel2.boundary_conditions = {e: {"left": 4, "right": 7}}

        model = pybamm.BaseModel()
        model.update(submodel1, submodel2)

        self.assertEqual(model.rhs[d], submodel1.rhs[d])
        self.assertEqual(model.initial_conditions[d], submodel1.initial_conditions[d])
        self.assertEqual(model.boundary_conditions[d], submodel1.boundary_conditions[d])
        self.assertEqual(model.rhs[e], submodel2.rhs[e])
        self.assertEqual(model.initial_conditions[e], submodel2.initial_conditions[e])
        self.assertEqual(model.boundary_conditions[e], submodel2.boundary_conditions[e])

    def test_check_well_posedness_variables(self):
        # Well-posed ODE model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        d = pybamm.Variable("d", domain=whole_cell)
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1, d: -c}
        model.initial_conditions = {c: 1, d: 2}
        model.boundary_conditions = {
            c: {"left": 0, "right": 0},
            d: {"left": 0, "right": 0},
        }
        model.check_well_posedness()

        # Well-posed DAE model
        e = pybamm.Variable("e", domain=whole_cell)
        model.algebraic = {e: e - c - d}
        model.check_well_posedness()

        # Underdetermined model - not enough differential equations
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1}
        model.algebraic = {e: e - c - d}
        with self.assertRaisesRegex(pybamm.ModelError, "underdetermined"):
            model.check_well_posedness()

        # Underdetermined model - not enough algebraic equations
        model.algebraic = {}
        with self.assertRaisesRegex(pybamm.ModelError, "underdetermined"):
            model.check_well_posedness()

        # Overdetermined model - repeated keys
        model.algebraic = {c: c - d, d: e + d}
        with self.assertRaisesRegex(pybamm.ModelError, "overdetermined"):
            model.check_well_posedness()
        # Overdetermined model - extra keys in algebraic
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1, d: -d}
        model.algebraic = {e: c - d}
        with self.assertRaisesRegex(pybamm.ModelError, "overdetermined"):
            model.check_well_posedness()
        model.rhs = {c: 1, d: -1}
        model.algebraic = {e: c - d}
        with self.assertRaisesRegex(pybamm.ModelError, "overdetermined"):
            model.check_well_posedness()

    def test_check_well_posedness_initial_boundary_conditions(self):
        # Well-posed model - Dirichlet
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        model = pybamm.BaseModel()
        c = pybamm.Variable("c", domain=whole_cell)
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(c)) - 1}
        model.initial_conditions = {c: 1}
        model.boundary_conditions = {2 * c: {"left": 0, "right": 0}}
        model.check_well_posedness()

        # Well-posed model - Neumann
        model.boundary_conditions = {3 * pybamm.grad(c) + 2: {"left": 0, "right": 0}}
        model.check_well_posedness()

        # Model with bad initial conditions (expect assertion error)
        d = pybamm.Variable("d", domain=whole_cell)
        model.initial_conditions = {d: 3}
        with self.assertRaisesRegex(pybamm.ModelError, "initial condition"):
            model.check_well_posedness()

        # Model with bad boundary conditions - Dirichlet (expect assertion error)
        d = pybamm.Variable("d", domain=whole_cell)
        model.initial_conditions = {c: 3}
        model.boundary_conditions = {d: {"left": 0, "right": 0}}
        with self.assertRaisesRegex(pybamm.ModelError, "boundary condition"):
            model.check_well_posedness()

        # Model with bad boundary conditions - Neumann (expect assertion error)
        d = pybamm.Variable("d", domain=whole_cell)
        model.initial_conditions = {c: 3}
        model.boundary_conditions = {4 * pybamm.grad(d): {"left": 0, "right": 0}}
        with self.assertRaisesRegex(pybamm.ModelError, "boundary condition"):
            model.check_well_posedness()

        # Algebraic well-posed model
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        model = pybamm.BaseModel()
        model.algebraic = {c: 5 * pybamm.div(pybamm.grad(c)) - 1}
        model.boundary_conditions = {2 * c: {"left": 0, "right": 0}}
        model.check_well_posedness()
        model.boundary_conditions = {pybamm.grad(c): {"left": 0, "right": 0}}
        model.check_well_posedness()

        # Algebraic model with bad boundary conditions
        model.boundary_conditions = {d: {"left": 0, "right": 0}}
        with self.assertRaisesRegex(pybamm.ModelError, "boundary condition"):
            model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
