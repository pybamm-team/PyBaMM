#
# Tests for the base model class
#
import os
import platform
import subprocess  # nosec
import unittest

import casadi
import numpy as np

import pybamm


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
        model.initial_conditions[c0] = 34
        self.assertIsInstance(model.initial_conditions[c0], pybamm.Scalar)
        self.assertEqual(model.initial_conditions[c0].value, 34)

        # Variable in initial conditions should fail
        with self.assertRaisesRegex(
            TypeError, "Initial conditions cannot contain 'Variable' objects"
        ):
            model.initial_conditions = {c0: pybamm.Variable("v")}

        # non-matching domains should fail
        with self.assertRaises(pybamm.DomainError):
            model.initial_conditions = {
                pybamm.Symbol("c", domain=["positive electrode"]): pybamm.Symbol(
                    "alpha", domain=["negative electrode"]
                )
            }

    def test_boundary_conditions_set_get(self):
        model = pybamm.BaseModel()
        boundary_conditions = {
            "c": {"left": ("epsilon", "Dirichlet"), "right": ("eta", "Dirichlet")}
        }
        model.boundary_conditions = boundary_conditions
        self.assertEqual(boundary_conditions, model.boundary_conditions)

        # Test number input
        c0 = pybamm.Symbol("c0")
        model.boundary_conditions[c0] = {
            "left": (-2, "Dirichlet"),
            "right": (4, "Dirichlet"),
        }
        self.assertIsInstance(model.boundary_conditions[c0]["left"][0], pybamm.Scalar)
        self.assertIsInstance(model.boundary_conditions[c0]["right"][0], pybamm.Scalar)
        self.assertEqual(model.boundary_conditions[c0]["left"][0].value, -2)
        self.assertEqual(model.boundary_conditions[c0]["right"][0].value, 4)
        self.assertEqual(model.boundary_conditions[c0]["left"][1], "Dirichlet")
        self.assertEqual(model.boundary_conditions[c0]["right"][1], "Dirichlet")

        # Check bad bc type
        bad_bcs = {c0: {"left": (-2, "bad type"), "right": (4, "bad type")}}
        with self.assertRaisesRegex(pybamm.ModelError, "boundary condition"):
            model.boundary_conditions = bad_bcs

    def test_variables_set_get(self):
        model = pybamm.BaseModel()
        variables = {"c": "alpha", "d": "beta"}
        model.variables = variables
        self.assertEqual(variables, model.variables)
        self.assertEqual(model.variable_names(), list(variables.keys()))

    def test_jac_set_get(self):
        model = pybamm.BaseModel()
        model.jacobian = "test"
        self.assertEqual(model.jacobian, "test")

    def test_read_parameters(self):
        # Read parameters from different parts of the model
        model = pybamm.BaseModel()
        a = pybamm.Parameter("a")
        b = pybamm.InputParameter("b", "test")
        c = pybamm.Parameter("c")
        d = pybamm.Parameter("d")
        e = pybamm.Parameter("e")
        f = pybamm.InputParameter("f")
        g = pybamm.Parameter("g")
        h = pybamm.Parameter("h")
        i = pybamm.InputParameter("i")

        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: -u * a}
        model.algebraic = {v: v - b}
        model.initial_conditions = {u: c, v: d}
        model.events = [pybamm.Event("u=e", u - e)]
        model.variables = {"v+f+i": v + f + i}
        model.boundary_conditions = {
            u: {"left": (g, "Dirichlet"), "right": (0, "Neumann")},
            v: {"left": (0, "Dirichlet"), "right": (h, "Neumann")},
        }

        # Test variables_and_events
        self.assertIn("v+f+i", model.variables_and_events)
        self.assertIn("Event: u=e", model.variables_and_events)

        self.assertEqual(
            set([x.name for x in model.parameters]),
            set([x.name for x in [a, b, c, d, e, f, g, h, i]]),
        )
        self.assertTrue(
            all(
                isinstance(x, (pybamm.Parameter, pybamm.InputParameter))
                for x in model.parameters
            )
        )

        model.variables = {
            "v+f+i": v + pybamm.FunctionParameter("f", {"Time [s]": pybamm.t}) + i
        }
        model.print_parameter_info()

    def test_read_input_parameters(self):
        # Read input parameters from different parts of the model
        model = pybamm.BaseModel()
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        c = pybamm.InputParameter("c")
        d = pybamm.InputParameter("d")
        e = pybamm.InputParameter("e")
        f = pybamm.InputParameter("f")

        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: -u * a}
        model.algebraic = {v: v - b}
        model.initial_conditions = {u: c, v: d}
        model.events = [pybamm.Event("u=e", u - e)]
        model.variables = {"v+f": v + f}

        self.assertEqual(
            set([x.name for x in model.input_parameters]),
            set([x.name for x in [a, b, c, d, e, f]]),
        )
        self.assertTrue(
            all(isinstance(x, pybamm.InputParameter) for x in model.input_parameters)
        )

    def test_update(self):
        # model
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        model = pybamm.BaseModel()
        c = pybamm.Variable("c", domain=whole_cell)
        rhs = {c: 5 * pybamm.div(pybamm.grad(c)) - 1}
        initial_conditions = {c: 1}
        boundary_conditions = {c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}}
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
        submodel.boundary_conditions = {
            d: {"left": (4, "Dirichlet"), "right": (7, "Dirichlet")}
        }
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
        with self.assertRaises(pybamm.ModelError):
            model.update(submodel2)

        # update with multiple submodels
        submodel1 = submodel  # copy submodel from previous test
        submodel2 = pybamm.BaseModel()
        e = pybamm.Variable("e", domain=whole_cell)
        submodel2.rhs = {
            e: 5 * pybamm.div(pybamm.grad(d)) + pybamm.div(pybamm.grad(e)) - 1
        }
        submodel2.initial_conditions = {e: 3}
        submodel2.boundary_conditions = {
            e: {"left": (4, "Dirichlet"), "right": (7, "Dirichlet")}
        }

        model = pybamm.BaseModel()
        model.update(submodel1, submodel2)

        self.assertEqual(model.rhs[d], submodel1.rhs[d])
        self.assertEqual(model.initial_conditions[d], submodel1.initial_conditions[d])
        self.assertEqual(model.boundary_conditions[d], submodel1.boundary_conditions[d])
        self.assertEqual(model.rhs[e], submodel2.rhs[e])
        self.assertEqual(model.initial_conditions[e], submodel2.initial_conditions[e])
        self.assertEqual(model.boundary_conditions[e], submodel2.boundary_conditions[e])

    def test_new_copy(self):
        model = pybamm.BaseModel(name="a model")
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        d = pybamm.Variable("d", domain=whole_cell)
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1, d: -c}
        model.initial_conditions = {c: 1, d: 2}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
            d: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
        }
        model.use_jacobian = False
        model.convert_to_format = "python"

        new_model = model.new_copy()
        self.assertEqual(new_model.name, model.name)
        self.assertEqual(new_model.use_jacobian, model.use_jacobian)
        self.assertEqual(new_model.convert_to_format, model.convert_to_format)

    def test_check_no_repeated_keys(self):
        model = pybamm.BaseModel()

        var = pybamm.Variable("var")
        model.rhs = {var: -1}
        var = pybamm.Variable("var")
        model.algebraic = {var: var}
        with self.assertRaisesRegex(pybamm.ModelError, "Multiple equations specified"):
            model.check_no_repeated_keys()

    def test_check_well_posedness_variables(self):
        # Well-posed ODE model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        d = pybamm.Variable("d", domain=whole_cell)
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1, d: -c}
        model.initial_conditions = {c: 1, d: 2}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
            d: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
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

        # After discretisation, don't check for overdetermined from extra algebraic keys
        model = pybamm.BaseModel()
        model.algebraic = {c: 5 * pybamm.StateVector(slice(0, 15)) - 1}
        # passes with post_discretisation=True
        model.check_well_posedness(post_discretisation=True)
        # fails with post_discretisation=False (default)
        with self.assertRaisesRegex(pybamm.ModelError, "extra algebraic keys"):
            model.check_well_posedness()

        # after discretisation, algebraic equation without a StateVector fails
        model = pybamm.BaseModel()
        model.algebraic = {
            c: 1,
            d: pybamm.StateVector(slice(0, 15)) - pybamm.StateVector(slice(15, 30)),
        }
        with self.assertRaisesRegex(
            pybamm.ModelError,
            "each algebraic equation must contain at least one StateVector",
        ):
            model.check_well_posedness(post_discretisation=True)

        # model must be in semi-explicit form
        model = pybamm.BaseModel()
        model.rhs = {c: d.diff(pybamm.t), d: -1}
        model.initial_conditions = {c: 1, d: 1}
        with self.assertRaisesRegex(
            pybamm.ModelError, "time derivative of variable found"
        ):
            model.check_well_posedness()

        # model must be in semi-explicit form
        model = pybamm.BaseModel()
        model.algebraic = {c: 2 * d - c, d: c * d.diff(pybamm.t) - d}
        model.initial_conditions = {c: 1, d: 1}
        with self.assertRaisesRegex(
            pybamm.ModelError, "time derivative of variable found"
        ):
            model.check_well_posedness()

        # model must be in semi-explicit form
        model = pybamm.BaseModel()
        model.rhs = {c: d.diff(pybamm.t), d: -1}
        model.initial_conditions = {c: 1, d: 1}
        with self.assertRaisesRegex(
            pybamm.ModelError, "time derivative of variable found"
        ):
            model.check_well_posedness()

        # model must be in semi-explicit form
        model = pybamm.BaseModel()
        model.algebraic = {
            d: 5 * pybamm.StateVector(slice(0, 15)) - 1,
            c: 5 * pybamm.StateVectorDot(slice(0, 15)) - 1,
        }
        with self.assertRaisesRegex(
            pybamm.ModelError, "time derivative of state vector found"
        ):
            model.check_well_posedness(post_discretisation=True)

        # model must be in semi-explicit form
        model = pybamm.BaseModel()
        model.rhs = {c: 5 * pybamm.StateVectorDot(slice(0, 15)) - 1}
        model.initial_conditions = {c: 1}
        with self.assertRaisesRegex(
            pybamm.ModelError, "time derivative of state vector found"
        ):
            model.check_well_posedness(post_discretisation=True)

    def test_check_well_posedness_initial_boundary_conditions(self):
        # Well-posed model - Dirichlet
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        model = pybamm.BaseModel()
        c = pybamm.Variable("c", domain=whole_cell)
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(c)) - 1}
        model.initial_conditions = {c: 1}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        model.check_well_posedness()

        # Well-posed model - Neumann
        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        model.check_well_posedness()

        # Model with bad initial conditions (expect assertion error)
        d = pybamm.Variable("d", domain=whole_cell)
        model.initial_conditions = {d: 3}
        with self.assertRaisesRegex(pybamm.ModelError, "initial condition"):
            model.check_well_posedness()

        # Algebraic well-posed model
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        model = pybamm.BaseModel()
        model.algebraic = {c: 5 * pybamm.div(pybamm.grad(c)) - 1}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        model.check_well_posedness()
        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        model.check_well_posedness()

    def test_check_well_posedness_output_variables(self):
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        d = pybamm.Variable("d", domain=whole_cell)
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1, d: -c}
        model.initial_conditions = {c: 1, d: 2}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
            d: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
        }
        model._variables = {"something else": c}

        # check error raised if undefined variable in list of Variables
        pybamm.settings.debug_mode = True
        model = pybamm.BaseModel()
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables = {"d": d}
        with self.assertRaisesRegex(pybamm.ModelError, "No key set for variable"):
            model.check_well_posedness()

        # check error is raised even if some modified form of d is in model.rhs
        two_d = 2 * d
        model.rhs[two_d] = -d
        model.initial_conditions[two_d] = 1
        with self.assertRaisesRegex(pybamm.ModelError, "No key set for variable"):
            model.check_well_posedness()

        # add d to rhs, fine
        model.rhs[d] = -d
        model.initial_conditions[d] = 1
        model.check_well_posedness()

    def test_export_casadi(self):
        model = pybamm.BaseModel()
        t = pybamm.t
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        model.rhs = {a: -a * p}
        model.algebraic = {b: a - b}
        model.initial_conditions = {a: q, b: 1}
        model.variables = {"a+b": a + b - t}

        out = model.export_casadi_objects(["a+b"], input_parameter_order=["p", "q"])

        # Try making a function from the outputs
        t, x, z, p = out["t"], out["x"], out["z"], out["inputs"]
        x0, z0 = out["x0"], out["z0"]
        rhs, alg = out["rhs"], out["algebraic"]
        var = out["variables"]["a+b"]
        jac_rhs, jac_alg = out["jac_rhs"], out["jac_algebraic"]
        x0_fn = casadi.Function("x0", [p], [x0])
        z0_fn = casadi.Function("x0", [p], [z0])
        rhs_fn = casadi.Function("rhs", [t, x, z, p], [rhs])
        alg_fn = casadi.Function("alg", [t, x, z, p], [alg])
        jac_rhs_fn = casadi.Function("jac_rhs", [t, x, z, p], [jac_rhs])
        jac_alg_fn = casadi.Function("jac_alg", [t, x, z, p], [jac_alg])
        var_fn = casadi.Function("var", [t, x, z, p], [var])

        # Test that function values are as expected
        self.assertEqual(x0_fn([0, 5]), 5)
        self.assertEqual(z0_fn([0, 0]), 1)
        self.assertEqual(rhs_fn(0, 3, 2, [7, 2]), -21)
        self.assertEqual(alg_fn(0, 3, 2, [7, 2]), 1)
        np.testing.assert_array_equal(np.array(jac_rhs_fn(5, 6, 7, [8, 9])), [[-8, 0]])
        np.testing.assert_array_equal(np.array(jac_alg_fn(5, 6, 7, [8, 9])), [[1, -1]])
        self.assertEqual(var_fn(6, 3, 2, [7, 2]), -1)

        # Now change the order of input parameters
        out = model.export_casadi_objects(["a+b"], input_parameter_order=["q", "p"])

        # Try making a function from the outputs
        t, x, z, p = out["t"], out["x"], out["z"], out["inputs"]
        x0, z0 = out["x0"], out["z0"]
        rhs, alg = out["rhs"], out["algebraic"]
        var = out["variables"]["a+b"]
        jac_rhs, jac_alg = out["jac_rhs"], out["jac_algebraic"]
        x0_fn = casadi.Function("x0", [p], [x0])
        z0_fn = casadi.Function("x0", [p], [z0])
        rhs_fn = casadi.Function("rhs", [t, x, z, p], [rhs])
        alg_fn = casadi.Function("alg", [t, x, z, p], [alg])
        jac_rhs_fn = casadi.Function("jac_rhs", [t, x, z, p], [jac_rhs])
        jac_alg_fn = casadi.Function("jac_alg", [t, x, z, p], [jac_alg])
        var_fn = casadi.Function("var", [t, x, z, p], [var])

        # Test that function values are as expected
        self.assertEqual(x0_fn([5, 0]), 5)
        self.assertEqual(z0_fn([0, 0]), 1)
        self.assertEqual(rhs_fn(0, 3, 2, [2, 7]), -21)
        self.assertEqual(alg_fn(0, 3, 2, [2, 7]), 1)
        np.testing.assert_array_equal(np.array(jac_rhs_fn(5, 6, 7, [9, 8])), [[-8, 0]])
        np.testing.assert_array_equal(np.array(jac_alg_fn(5, 6, 7, [9, 8])), [[1, -1]])
        self.assertEqual(var_fn(6, 3, 2, [2, 7]), -1)

        # Test fails if order not specified
        with self.assertRaisesRegex(
            ValueError, "input_parameter_order must be specified"
        ):
            model.export_casadi_objects(["a+b"])

        # Fine if order is not specified if there is only one input parameter
        model = pybamm.BaseModel()
        p = pybamm.InputParameter("p")
        model.rhs = {a: -a}
        model.algebraic = {b: a - b}
        model.initial_conditions = {a: 1, b: 1}
        model.variables = {"a+b": a + b - p}

        out = model.export_casadi_objects(["a+b"])

        # Try making a function from the outputs
        t, x, z, p = out["t"], out["x"], out["z"], out["inputs"]
        var = out["variables"]["a+b"]
        var_fn = casadi.Function("var", [t, x, z, p], [var])

        # Test that function values are as expected
        # a + b - p = 3 + 2 - 7 = -2
        self.assertEqual(var_fn(6, 3, 2, [7]), -2)

        # Test fails if not discretised
        model = pybamm.lithium_ion.SPMe()
        with self.assertRaisesRegex(
            pybamm.DiscretisationError, "Cannot automatically discretise model"
        ):
            model.export_casadi_objects(["Electrolyte concentration [mol.m-3]"])

    @unittest.skipIf(platform.system() == "Windows", "Skipped for Windows")
    def test_generate_casadi(self):
        model = pybamm.BaseModel()
        t = pybamm.t
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        model.rhs = {a: -a * p}
        model.algebraic = {b: a - b}
        model.initial_conditions = {a: q, b: 1}
        model.variables = {"a+b": a + b - t}

        # Generate C code
        model.generate("test.c", ["a+b"], input_parameter_order=["p", "q"])

        # Compile
        subprocess.run(["gcc", "-fPIC", "-shared", "-o", "test.so", "test.c"])  # nosec

        # Read the generated functions
        x0_fn = casadi.external("x0", "./test.so")
        z0_fn = casadi.external("z0", "./test.so")
        rhs_fn = casadi.external("rhs_", "./test.so")
        alg_fn = casadi.external("alg_", "./test.so")
        jac_rhs_fn = casadi.external("jac_rhs", "./test.so")
        jac_alg_fn = casadi.external("jac_alg", "./test.so")
        var_fn = casadi.external("variables", "./test.so")

        # Test that function values are as expected
        self.assertEqual(x0_fn([2, 5]), 5)
        self.assertEqual(z0_fn([0, 0]), 1)
        self.assertEqual(rhs_fn(0, 3, 2, [7, 2]), -21)
        self.assertEqual(alg_fn(0, 3, 2, [7, 2]), 1)
        np.testing.assert_array_equal(np.array(jac_rhs_fn(5, 6, 7, [8, 9])), [[-8, 0]])
        np.testing.assert_array_equal(np.array(jac_alg_fn(5, 6, 7, [8, 9])), [[1, -1]])
        self.assertEqual(var_fn(6, 3, 2, [7, 2]), -1)

        # Remove generated files.
        os.remove("test.c")
        os.remove("test.so")

    def test_set_initial_conditions(self):
        # Set up model
        model = pybamm.BaseModel()
        var_scalar = pybamm.Variable("var_scalar")
        var_1D = pybamm.Variable("var_1D", domain="negative electrode")
        var_2D = pybamm.Variable(
            "var_2D",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        var_concat_neg = pybamm.Variable("var_concat_neg", domain="negative electrode")
        var_concat_sep = pybamm.Variable("var_concat_sep", domain="separator")
        var_concat = pybamm.concatenation(var_concat_neg, var_concat_sep)
        model.rhs = {var_scalar: -var_scalar, var_1D: -var_1D}
        model.algebraic = {var_2D: -var_2D, var_concat: -var_concat}
        model.initial_conditions = {var_scalar: 1, var_1D: 1, var_2D: 1, var_concat: 1}
        model.variables = {
            "var_scalar": var_scalar,
            "var_1D": var_1D,
            "var_2D": var_2D,
            "var_concat_neg": var_concat_neg,
            "var_concat_sep": var_concat_sep,
            "var_concat": var_concat,
        }

        # Test original initial conditions
        self.assertEqual(model.initial_conditions[var_scalar].value, 1)
        self.assertEqual(model.initial_conditions[var_1D].value, 1)
        self.assertEqual(model.initial_conditions[var_2D].value, 1)
        self.assertEqual(model.initial_conditions[var_concat].value, 1)

        # Discretise
        geometry = {
            "negative electrode": {"x_n": {"min": 0, "max": 1}},
            "separator": {"x_s": {"min": 1, "max": 2}},
            "negative particle": {"r_n": {"min": 0, "max": 1}},
        }
        submeshes = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
        }
        var_pts = {"x_n": 10, "x_s": 10, "r_n": 5}
        mesh = pybamm.Mesh(geometry, submeshes, var_pts)
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume(),
            "separator": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        model_disc = disc.process_model(model, inplace=False)
        t = np.linspace(0, 1)
        y = np.tile(3 * t, (1 + 30 + 50, 1))
        sol = pybamm.Solution(t, y, model_disc, {})

        # Update out-of-place first, since otherwise we'll have already modified the
        # model
        new_model = model.set_initial_conditions_from(sol, inplace=False)
        # Make sure original model is unchanged
        np.testing.assert_array_equal(
            model.initial_conditions[var_scalar].evaluate(), 1
        )
        np.testing.assert_array_equal(model.initial_conditions[var_1D].evaluate(), 1)
        np.testing.assert_array_equal(model.initial_conditions[var_2D].evaluate(), 1)
        np.testing.assert_array_equal(
            model.initial_conditions[var_concat].evaluate(), 1
        )

        # Now update inplace
        model.set_initial_conditions_from(sol)

        # Test new initial conditions (both in place and not)
        for mdl in [model, new_model]:
            var_scalar = mdl.variables["var_scalar"]
            self.assertIsInstance(mdl.initial_conditions[var_scalar], pybamm.Vector)
            self.assertEqual(mdl.initial_conditions[var_scalar].entries, 3)

            var_1D = mdl.variables["var_1D"]
            self.assertIsInstance(mdl.initial_conditions[var_1D], pybamm.Vector)
            self.assertEqual(mdl.initial_conditions[var_1D].shape, (10, 1))
            np.testing.assert_array_equal(mdl.initial_conditions[var_1D].entries, 3)

            var_2D = mdl.variables["var_2D"]
            self.assertIsInstance(mdl.initial_conditions[var_2D], pybamm.Vector)
            self.assertEqual(mdl.initial_conditions[var_2D].shape, (50, 1))
            np.testing.assert_array_equal(mdl.initial_conditions[var_2D].entries, 3)

            var_concat = mdl.variables["var_concat"]
            self.assertIsInstance(mdl.initial_conditions[var_concat], pybamm.Vector)
            self.assertEqual(mdl.initial_conditions[var_concat].shape, (20, 1))
            np.testing.assert_array_equal(mdl.initial_conditions[var_concat].entries, 3)

        # Test updating a discretised model (out-of-place)
        new_model_disc = model_disc.set_initial_conditions_from(sol, inplace=False)

        # Test new initial conditions
        var_scalar = list(new_model_disc.initial_conditions.keys())[0]
        self.assertIsInstance(
            new_model_disc.initial_conditions[var_scalar], pybamm.Vector
        )
        self.assertEqual(new_model_disc.initial_conditions[var_scalar].entries, 3)

        var_1D = list(new_model_disc.initial_conditions.keys())[1]
        self.assertIsInstance(new_model_disc.initial_conditions[var_1D], pybamm.Vector)
        self.assertEqual(new_model_disc.initial_conditions[var_1D].shape, (10, 1))
        np.testing.assert_array_equal(
            new_model_disc.initial_conditions[var_1D].entries, 3
        )

        var_2D = list(new_model_disc.initial_conditions.keys())[2]
        self.assertIsInstance(new_model_disc.initial_conditions[var_2D], pybamm.Vector)
        self.assertEqual(new_model_disc.initial_conditions[var_2D].shape, (50, 1))
        np.testing.assert_array_equal(
            new_model_disc.initial_conditions[var_2D].entries, 3
        )

        var_concat = list(new_model_disc.initial_conditions.keys())[3]
        self.assertIsInstance(
            new_model_disc.initial_conditions[var_concat], pybamm.Vector
        )
        self.assertEqual(new_model_disc.initial_conditions[var_concat].shape, (20, 1))
        np.testing.assert_array_equal(
            new_model_disc.initial_conditions[var_concat].entries, 3
        )

        np.testing.assert_array_equal(
            new_model_disc.concatenated_initial_conditions.evaluate(), 3
        )

        # Test updating a new model with a different model
        new_model = pybamm.BaseModel()
        new_var_scalar = pybamm.Variable("var_scalar")
        new_var_1D = pybamm.Variable("var_1D", domain="negative electrode")
        new_var_2D = pybamm.Variable(
            "var_2D",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        new_var_concat_neg = pybamm.Variable(
            "var_concat_neg", domain="negative electrode"
        )
        new_var_concat_sep = pybamm.Variable("var_concat_sep", domain="separator")
        new_var_concat = pybamm.concatenation(new_var_concat_neg, new_var_concat_sep)
        new_model.rhs = {
            new_var_scalar: -2 * new_var_scalar,
            new_var_1D: -2 * new_var_1D,
        }
        new_model.algebraic = {
            new_var_2D: -2 * new_var_2D,
            new_var_concat: -2 * new_var_concat,
        }
        new_model.initial_conditions = {
            new_var_scalar: 1,
            new_var_1D: 1,
            new_var_2D: 1,
            new_var_concat: 1,
        }
        new_model.variables = {
            "var_scalar": new_var_scalar,
            "var_1D": new_var_1D,
            "var_2D": new_var_2D,
            "var_concat_neg": new_var_concat_neg,
            "var_concat_sep": new_var_concat_sep,
            "var_concat": new_var_concat,
        }

        # Update the new model with the solution from another model
        new_model.set_initial_conditions_from(sol)

        # Test new initial conditions (both in place and not)
        var_scalar = new_model.variables["var_scalar"]
        self.assertIsInstance(new_model.initial_conditions[var_scalar], pybamm.Vector)
        self.assertEqual(new_model.initial_conditions[var_scalar].entries, 3)

        var_1D = new_model.variables["var_1D"]
        self.assertIsInstance(new_model.initial_conditions[var_1D], pybamm.Vector)
        self.assertEqual(new_model.initial_conditions[var_1D].shape, (10, 1))
        np.testing.assert_array_equal(new_model.initial_conditions[var_1D].entries, 3)

        var_2D = new_model.variables["var_2D"]
        self.assertIsInstance(new_model.initial_conditions[var_2D], pybamm.Vector)
        self.assertEqual(new_model.initial_conditions[var_2D].shape, (50, 1))
        np.testing.assert_array_equal(new_model.initial_conditions[var_2D].entries, 3)

        var_concat = new_model.variables["var_concat"]
        self.assertIsInstance(new_model.initial_conditions[var_concat], pybamm.Vector)
        self.assertEqual(new_model.initial_conditions[var_concat].shape, (20, 1))
        np.testing.assert_array_equal(
            new_model.initial_conditions[var_concat].entries, 3
        )

        # Update the new model with a dictionary
        sol_dict = {
            "var_scalar": 5 * t,
            "var_1D": np.tile(5 * t, (10, 1)),
            "var_concat_neg": np.tile(5 * t, (10, 1)),
            "var_concat_sep": np.tile(5 * t, (10, 1)),
            "var_2D": np.tile(5 * t, (10, 5, 1)),
        }
        new_model.set_initial_conditions_from(sol_dict)

        # Test new initial conditions (both in place and not)
        var_scalar = new_model.variables["var_scalar"]
        self.assertIsInstance(new_model.initial_conditions[var_scalar], pybamm.Vector)
        self.assertEqual(new_model.initial_conditions[var_scalar].entries, 5)

        var_1D = new_model.variables["var_1D"]
        self.assertIsInstance(new_model.initial_conditions[var_1D], pybamm.Vector)
        self.assertEqual(new_model.initial_conditions[var_1D].shape, (10, 1))
        np.testing.assert_array_equal(new_model.initial_conditions[var_1D].entries, 5)

        var_2D = new_model.variables["var_2D"]
        self.assertIsInstance(new_model.initial_conditions[var_2D], pybamm.Vector)
        self.assertEqual(new_model.initial_conditions[var_2D].shape, (50, 1))
        np.testing.assert_array_equal(new_model.initial_conditions[var_2D].entries, 5)

        var_concat = new_model.variables["var_concat"]
        self.assertIsInstance(new_model.initial_conditions[var_concat], pybamm.Vector)
        self.assertEqual(new_model.initial_conditions[var_concat].shape, (20, 1))
        np.testing.assert_array_equal(
            new_model.initial_conditions[var_concat].entries, 5
        )

        # Test updating a discretised model (out-of-place)
        model_disc = disc.process_model(model, inplace=False)
        new_model_disc = model_disc.set_initial_conditions_from(sol_dict, inplace=False)

        # Test new initial conditions
        var_scalar = list(new_model_disc.initial_conditions.keys())[0]
        self.assertIsInstance(
            new_model_disc.initial_conditions[var_scalar], pybamm.Vector
        )
        self.assertEqual(new_model_disc.initial_conditions[var_scalar].entries, 5)

        var_1D = list(new_model_disc.initial_conditions.keys())[1]
        self.assertIsInstance(new_model_disc.initial_conditions[var_1D], pybamm.Vector)
        self.assertEqual(new_model_disc.initial_conditions[var_1D].shape, (10, 1))
        np.testing.assert_array_equal(
            new_model_disc.initial_conditions[var_1D].entries, 5
        )

        var_2D = list(new_model_disc.initial_conditions.keys())[2]
        self.assertIsInstance(new_model_disc.initial_conditions[var_2D], pybamm.Vector)
        self.assertEqual(new_model_disc.initial_conditions[var_2D].shape, (50, 1))
        np.testing.assert_array_equal(
            new_model_disc.initial_conditions[var_2D].entries, 5
        )

        var_concat = list(new_model_disc.initial_conditions.keys())[3]
        self.assertIsInstance(
            new_model_disc.initial_conditions[var_concat], pybamm.Vector
        )
        self.assertEqual(new_model_disc.initial_conditions[var_concat].shape, (20, 1))
        np.testing.assert_array_equal(
            new_model_disc.initial_conditions[var_concat].entries, 5
        )

        np.testing.assert_array_equal(
            new_model_disc.concatenated_initial_conditions.evaluate(), 5
        )

    def test_set_initial_condition_errors(self):
        model = pybamm.BaseModel()
        var = pybamm.Scalar(1)
        model.rhs = {var: -var}
        model.initial_conditions = {var: 1}
        with self.assertRaisesRegex(NotImplementedError, "Variable must have type"):
            model.set_initial_conditions_from({})

        var = pybamm.Variable(
            "var",
            domain="negative particle",
            auxiliary_domains={
                "secondary": "negative electrode",
                "tertiary": "current collector",
            },
        )
        model.rhs = {var: -var}
        model.initial_conditions = {var: 1}
        with self.assertRaisesRegex(
            NotImplementedError, "Variable must be 0D, 1D, or 2D"
        ):
            model.set_initial_conditions_from({"var": np.ones((5, 6, 7, 8))})

        var_concat_neg = pybamm.Variable("var concat neg", domain="negative electrode")
        var_concat_sep = pybamm.Variable("var concat sep", domain="separator")
        var_concat = pybamm.concatenation(var_concat_neg, var_concat_sep)
        model.algebraic = {var_concat: -var_concat}
        model.initial_conditions = {var_concat: 1}
        with self.assertRaisesRegex(
            NotImplementedError, "Variable in concatenation must be 1D"
        ):
            model.set_initial_conditions_from({"var concat neg": np.ones((5, 6, 7))})

        # Inconsistent model and variable names
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: -var}
        model.initial_conditions = {var: pybamm.Scalar(1)}
        with self.assertRaisesRegex(pybamm.ModelError, "must appear in the solution"):
            model.set_initial_conditions_from({"wrong var": 2})
        var = pybamm.concatenation(
            pybamm.Variable("var", "test"), pybamm.Variable("var2", "test2")
        )
        model.rhs = {var: -var}
        model.initial_conditions = {var: pybamm.Scalar(1)}
        with self.assertRaisesRegex(pybamm.ModelError, "must appear in the solution"):
            model.set_initial_conditions_from({"wrong var": 2})

    def test_set_variables_error(self):
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        with self.assertRaisesRegex(ValueError, "not var"):
            model.variables = {"not var": var}

    def test_build_submodels(self):
        class Submodel1(pybamm.BaseSubModel):
            def __init__(self, param, domain, options=None):
                super().__init__(param, domain, options=options)

            def get_fundamental_variables(self):
                u = pybamm.Variable("u")
                v = pybamm.Variable("v")
                return {"u": u, "v": v}

            def get_coupled_variables(self, variables):
                return variables

            def set_rhs(self, variables):
                u = variables["u"]
                self.rhs = {u: 2}

            def set_algebraic(self, variables):
                v = variables["v"]
                self.algebraic = {v: v - 1}

            def set_initial_conditions(self, variables):
                u = variables["u"]
                v = variables["v"]
                self.initial_conditions = {u: 0, v: 0}

            def set_events(self, variables):
                u = variables["u"]
                self.events.append(
                    pybamm.Event(
                        "Large u",
                        u - 200,
                        pybamm.EventType.TERMINATION,
                    )
                )

        class Submodel2(pybamm.BaseSubModel):
            def __init__(self, param, domain, options=None):
                super().__init__(param, domain, options=options)

            def get_coupled_variables(self, variables):
                u = variables["u"]
                variables.update({"w": 2 * u})
                return variables

        model = pybamm.BaseModel()
        model.submodels = {
            "submodel 1": Submodel1(None, "negative"),
            "submodel 2": Submodel2(None, "negative"),
        }
        self.assertFalse(model._built)
        model.build_model()
        self.assertTrue(model._built)
        u = model.variables["u"]
        v = model.variables["v"]
        self.assertEqual(model.rhs[u].value, 2)
        self.assertEqual(model.algebraic[v], -1.0 + v)

    def test_timescale_lengthscale_get_set_not_implemented(self):
        model = pybamm.BaseModel()
        with self.assertRaises(NotImplementedError):
            model.timescale
        with self.assertRaises(NotImplementedError):
            model.length_scales
        with self.assertRaises(NotImplementedError):
            model.timescale = 1
        with self.assertRaises(NotImplementedError):
            model.length_scales = 1


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
