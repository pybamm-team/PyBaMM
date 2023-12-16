#
# Tests for the base model class
#
from tests import TestCase
import pybamm

import numpy as np
import unittest
from tests import (
    get_mesh_for_testing,
    get_discretisation_for_testing,
    get_1p1d_discretisation_for_testing,
    get_2p1d_mesh_for_testing,
)
from tests.shared import SpatialMethodForTesting

from scipy.sparse import block_diag, csc_matrix
from scipy.sparse.linalg import inv


class TestDiscretise(TestCase):
    def test_concatenate_in_order(self):
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        c = pybamm.Variable("c")

        initial_conditions = {
            c: pybamm.Scalar(1),
            a: pybamm.Scalar(2),
            b: pybamm.Scalar(3),
        }

        # create discretisation
        disc = get_discretisation_for_testing()

        disc.y_slices = {c: [slice(0, 1)], a: [slice(2, 3)], b: [slice(3, 4)]}
        result = disc._concatenate_in_order(initial_conditions)

        self.assertIsInstance(result, pybamm.Vector)
        np.testing.assert_array_equal(result.evaluate(), [[1], [2], [3]])

        initial_conditions = {a: pybamm.Scalar(2), b: pybamm.Scalar(3)}
        with self.assertRaises(pybamm.ModelError):
            result = disc._concatenate_in_order(initial_conditions, check_complete=True)

    def test_no_mesh(self):
        disc = pybamm.Discretisation(None, None)
        self.assertEqual(disc._spatial_methods, {})

    def test_add_internal_boundary_conditions(self):
        model = pybamm.BaseModel()
        c_e_n = pybamm.Variable("c_e_n", ["negative electrode"])
        c_e_s = pybamm.Variable("c_e_s", ["separator"])
        c_e_p = pybamm.Variable("c_e_p", ["positive electrode"])
        c_e = pybamm.concatenation(c_e_n, c_e_s, c_e_p)
        lbc = (pybamm.Scalar(0), "Neumann")
        rbc = (pybamm.Scalar(0), "Neumann")
        model.boundary_conditions = {c_e: {"left": lbc, "right": rbc}}

        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": SpatialMethodForTesting()}

        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.set_variable_slices([c_e_n, c_e_s, c_e_p])
        disc.bcs = disc.process_boundary_conditions(model)
        disc.set_internal_boundary_conditions(model)

        for child in c_e.children:
            self.assertTrue(child in disc.bcs.keys())

    def test_discretise_slicing(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        variables = [c]
        disc.set_variable_slices(variables)

        self.assertEqual(disc.y_slices, {c: [slice(0, 100)]})

        submesh = mesh[whole_cell]

        c_true = submesh.nodes**2
        y = c_true
        np.testing.assert_array_equal(y[disc.y_slices[c][0]], c_true)

        # Several variables
        d = pybamm.Variable("d", domain=whole_cell, bounds=(0, 1))
        jn = pybamm.Variable("jn", domain=["negative electrode"])
        variables = [c, d, jn]
        disc.set_variable_slices(variables)

        self.assertEqual(
            disc.y_slices,
            {c: [slice(0, 100)], d: [slice(100, 200)], jn: [slice(200, 240)]},
        )
        np.testing.assert_array_equal(
            disc.bounds[0], [-np.inf] * 100 + [0] * 100 + [-np.inf] * 40
        )
        np.testing.assert_array_equal(
            disc.bounds[1], [np.inf] * 100 + [1] * 100 + [np.inf] * 40
        )
        d_true = 4 * submesh.nodes
        jn_true = mesh["negative electrode"].nodes ** 3
        y = np.concatenate([c_true, d_true, jn_true])
        np.testing.assert_array_equal(y[disc.y_slices[c][0]], c_true)
        np.testing.assert_array_equal(y[disc.y_slices[d][0]], d_true)
        np.testing.assert_array_equal(y[disc.y_slices[jn][0]], jn_true)

        # Variables with a concatenation
        js = pybamm.Variable("js", domain=["separator"])
        jp = pybamm.Variable("jp", domain=["positive electrode"])
        j = pybamm.concatenation(jn, js, jp)
        variables = [c, d, j]
        disc.set_variable_slices(variables)
        self.assertEqual(
            disc.y_slices,
            {
                c: [slice(0, 100)],
                d: [slice(100, 200)],
                jn: [slice(200, 240)],
                js: [slice(240, 265)],
                jp: [slice(265, 300)],
                j: [slice(200, 300)],
            },
        )
        np.testing.assert_array_equal(
            disc.bounds[0], [-np.inf] * 100 + [0] * 100 + [-np.inf] * 100
        )
        np.testing.assert_array_equal(
            disc.bounds[1], [np.inf] * 100 + [1] * 100 + [np.inf] * 100
        )
        d_true = 4 * submesh.nodes
        jn_true = mesh["negative electrode"].nodes ** 3
        y = np.concatenate([c_true, d_true, jn_true])
        np.testing.assert_array_equal(y[disc.y_slices[c][0]], c_true)
        np.testing.assert_array_equal(y[disc.y_slices[d][0]], d_true)
        np.testing.assert_array_equal(y[disc.y_slices[jn][0]], jn_true)

        with self.assertRaisesRegex(TypeError, "y_slices should be"):
            disc.y_slices = 1

        # bounds with an InputParameter
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        v = pybamm.Variable("v", domain=whole_cell, bounds=(a, b))
        disc.set_variable_slices([v])
        np.testing.assert_array_equal(disc.bounds[0], [-np.inf] * 100)
        np.testing.assert_array_equal(disc.bounds[1], [np.inf] * 100)

    def test_process_symbol_base(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.SpatialMethod(),
            "negative particle": pybamm.SpatialMethod(),
            "positive particle": pybamm.SpatialMethod(),
            "current collector": pybamm.SpatialMethod(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # variable
        var = pybamm.Variable("var")
        var_vec = pybamm.Variable("var vec", domain=["negative electrode"])
        disc.y_slices = {var: [slice(53)], var_vec: [slice(53, 93)]}
        var_disc = disc.process_symbol(var)
        self.assertIsInstance(var_disc, pybamm.StateVector)
        self.assertEqual(var_disc.y_slices[0], disc.y_slices[var][0])

        # variable dot
        var_dot = pybamm.VariableDot("var'")
        var_dot_disc = disc.process_symbol(var_dot)
        self.assertIsInstance(var_dot_disc, pybamm.StateVectorDot)
        self.assertEqual(var_dot_disc.y_slices[0], disc.y_slices[var][0])

        # scalar
        scal = pybamm.Scalar(5)
        scal_disc = disc.process_symbol(scal)
        self.assertIsInstance(scal_disc, pybamm.Scalar)
        self.assertEqual(scal_disc.value, scal.value)
        # vector
        vec = pybamm.Vector([1, 2, 3, 4])
        vec_disc = disc.process_symbol(vec)
        self.assertIsInstance(vec_disc, pybamm.Vector)
        np.testing.assert_array_equal(vec_disc.entries, vec.entries)
        # matrix
        mat = pybamm.Matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
        mat_disc = disc.process_symbol(mat)
        self.assertIsInstance(mat_disc, pybamm.Matrix)
        np.testing.assert_array_equal(mat_disc.entries, mat.entries)

        # binary operator
        binary = var + scal
        binary_disc = disc.process_symbol(binary)
        self.assertEqual(binary_disc, 5 + pybamm.StateVector(slice(0, 53)))

        # non-spatial unary operator
        un1 = -var
        un1_disc = disc.process_symbol(un1)
        self.assertIsInstance(un1_disc, pybamm.Negate)
        self.assertIsInstance(un1_disc.children[0], pybamm.StateVector)

        un2 = abs(var)
        un2_disc = disc.process_symbol(un2)
        self.assertIsInstance(un2_disc, pybamm.AbsoluteValue)
        self.assertIsInstance(un2_disc.children[0], pybamm.StateVector)

        # function of one variable
        def myfun(x):
            return np.exp(x)

        func = pybamm.Function(myfun, var)
        func_disc = disc.process_symbol(func)
        self.assertIsInstance(func_disc, pybamm.Function)
        self.assertIsInstance(func_disc.children[0], pybamm.StateVector)

        # function of a scalar gets simplified
        func = pybamm.Function(myfun, scal)
        func_disc = disc.process_symbol(func)
        self.assertIsInstance(func_disc, pybamm.Scalar)

        # function of multiple variables
        def myfun(x, y):
            return np.exp(x) * y

        func = pybamm.Function(myfun, var, scal)
        func_disc = disc.process_symbol(func)
        self.assertIsInstance(func_disc, pybamm.Function)
        self.assertIsInstance(func_disc.children[0], pybamm.StateVector)
        self.assertIsInstance(func_disc.children[1], pybamm.Scalar)

        # boundary value
        bv_left = pybamm.BoundaryValue(var_vec, "left")
        bv_left_disc = disc.process_symbol(bv_left)
        self.assertIsInstance(bv_left_disc, pybamm.MatrixMultiplication)
        self.assertIsInstance(bv_left_disc.left, pybamm.Matrix)
        self.assertIsInstance(bv_left_disc.right, pybamm.StateVector)
        bv_right = pybamm.BoundaryValue(var_vec, "left")
        bv_right_disc = disc.process_symbol(bv_right)
        self.assertIsInstance(bv_right_disc, pybamm.MatrixMultiplication)
        self.assertIsInstance(bv_right_disc.left, pybamm.Matrix)
        self.assertIsInstance(bv_right_disc.right, pybamm.StateVector)

        # not implemented
        sym = pybamm.Symbol("sym")
        with self.assertRaises(NotImplementedError):
            disc.process_symbol(sym)

    def test_process_complex_expression(self):
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        expression = (5 * (3**var2)) / ((var1 - 4) + var2)

        # create discretisation
        disc = get_discretisation_for_testing()

        disc.y_slices = {var1: [slice(53)], var2: [slice(53, 106)]}
        exp_disc = disc.process_symbol(expression)
        self.assertEqual(
            exp_disc,
            (5.0 * (3.0 ** pybamm.StateVector(slice(53, 106))))
            / (
                (-4.0 + pybamm.StateVector(slice(0, 53)))
                + pybamm.StateVector(slice(53, 106))
            ),
        )

    def test_discretise_spatial_operator(self):
        # create discretisation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        variables = [var]
        disc.set_variable_slices(variables)

        # Simple expressions
        for eqn in [pybamm.grad(var), pybamm.div(pybamm.grad(var))]:
            eqn_disc = disc.process_symbol(eqn)

            self.assertIsInstance(eqn_disc, pybamm.MatrixMultiplication)
            self.assertIsInstance(eqn_disc.children[0], pybamm.Matrix)

            submesh = mesh[whole_cell]
            y = submesh.nodes**2
            var_disc = disc.process_symbol(var)
            # grad and var are identity operators here (for testing purposes)
            np.testing.assert_array_equal(
                eqn_disc.evaluate(None, y), var_disc.evaluate(None, y)
            )

        # More complex expressions
        for eqn in [var * pybamm.grad(var), var * pybamm.div(pybamm.grad(var))]:
            eqn_disc = disc.process_symbol(eqn)

            self.assertIsInstance(eqn_disc, pybamm.Multiplication)
            self.assertIsInstance(eqn_disc.children[0], pybamm.StateVector)
            self.assertIsInstance(eqn_disc.children[1], pybamm.MatrixMultiplication)
            self.assertIsInstance(eqn_disc.children[1].children[0], pybamm.Matrix)

            y = submesh.nodes**2
            var_disc = disc.process_symbol(var)
            # grad and var are identity operators here (for testing purposes)
            np.testing.assert_array_equal(
                eqn_disc.evaluate(None, y), var_disc.evaluate(None, y) ** 2
            )

    def test_discretise_average(self):
        var = pybamm.Variable("var", domain="negative particle size")
        R = pybamm.SpatialVariable("R", "negative particle size")
        f_a_dist = (R * 2) ** 2 * 2
        var_av = pybamm.SizeAverage(var, f_a_dist)

        geometry = {"negative particle size": {"R_n": {"min": 0.5, "max": 1.5}}}
        var_pts = {"R_n": 10}
        submesh_types = {"negative particle size": pybamm.Uniform1DSubMesh}
        spatial_methods = {"negative particle size": pybamm.FiniteVolume()}

        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        disc = pybamm.Discretisation(mesh, spatial_methods)

        disc.set_variable_slices([var])
        var_av_proc = disc.process_symbol(var_av)

        self.assertIsInstance(var_av_proc, pybamm.MatrixMultiplication)
        self.assertIsInstance(var_av_proc.right.right, pybamm.StateVector)

    def test_process_dict(self):
        # one equation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        N = pybamm.grad(c)
        rhs = {c: pybamm.div(N)}
        initial_conditions = {c: pybamm.Scalar(3)}
        variables = {"c_squared": c**2}
        boundary_conditions = {c: {"left": (0, "Neumann"), "right": (0, "Neumann")}}

        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        submesh = mesh[whole_cell]

        y = submesh.nodes[:, np.newaxis] ** 2
        disc.bcs = boundary_conditions

        disc.set_variable_slices(list(rhs.keys()))
        # rhs - grad and div are identity operators here
        processed_rhs = disc.process_dict(rhs)
        np.testing.assert_array_equal(y, processed_rhs[c].evaluate(None, y))
        # initial conditions
        y0 = disc.process_dict(initial_conditions)
        np.testing.assert_array_equal(
            y0[c].evaluate(0, None),
            3 * np.ones_like(submesh.nodes[:, np.newaxis]),
        )
        # vars
        processed_vars = disc.process_dict(variables)
        np.testing.assert_array_equal(
            y**2, processed_vars["c_squared"].evaluate(None, y)
        )

        # two equations
        T = pybamm.Variable("T", domain=["negative electrode"])
        q = pybamm.grad(T)
        rhs = {c: pybamm.div(N), T: pybamm.div(q)}
        initial_conditions = {c: pybamm.Scalar(3), T: pybamm.Scalar(5)}
        boundary_conditions = {}
        y = np.concatenate([submesh.nodes**2, mesh["negative electrode"].nodes ** 4])[
            :, np.newaxis
        ]

        variables = list(rhs.keys())
        disc.set_variable_slices(variables)
        # rhs
        processed_rhs = disc.process_dict(rhs)
        np.testing.assert_array_equal(
            y[disc.y_slices[c][0]], processed_rhs[c].evaluate(None, y)
        )
        np.testing.assert_array_equal(
            y[disc.y_slices[T][0]], processed_rhs[T].evaluate(None, y)
        )
        # initial conditions
        y0 = disc.process_dict(initial_conditions)
        np.testing.assert_array_equal(
            y0[c].evaluate(0, None),
            3 * np.ones_like(submesh.nodes[:, np.newaxis]),
        )
        np.testing.assert_array_equal(
            y0[T].evaluate(0, None),
            5 * np.ones_like(mesh["negative electrode"].nodes[:, np.newaxis]),
        )

    def test_process_variables_dict(self):
        # want to check the case where the keys are strings and
        # and the equation evals to a number

        variables = {"c": pybamm.Scalar(0)}

        disc = get_discretisation_for_testing()
        disc.process_dict(variables)

    def test_process_model_ode(self):
        # one equation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        N = pybamm.grad(c)
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N)}
        model.initial_conditions = {c: pybamm.Scalar(3)}
        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        model.variables = {"c": c, "N": N}

        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        submesh = mesh[whole_cell]
        disc.process_model(model)

        y0 = model.concatenated_initial_conditions.evaluate()
        np.testing.assert_array_equal(
            y0, 3 * np.ones_like(submesh.nodes[:, np.newaxis])
        )
        np.testing.assert_array_equal(y0, model.concatenated_rhs.evaluate(None, y0))

        # grad and div are identity operators here
        np.testing.assert_array_equal(y0, model.variables["c"].evaluate(None, y0))
        np.testing.assert_array_equal(y0, model.variables["N"].evaluate(None, y0))

        # mass matrix is identity
        np.testing.assert_array_equal(
            np.eye(submesh.nodes.shape[0]), model.mass_matrix.entries.toarray()
        )

        # Create StateVector to differentiate model with respect to
        y = pybamm.StateVector(slice(0, submesh.npts))

        # jacobian is identity
        jacobian = model.concatenated_rhs.jac(y).evaluate(0, y0)
        np.testing.assert_array_equal(np.eye(submesh.npts), jacobian.toarray())

        # several equations
        T = pybamm.Variable("T", domain=["negative electrode"])
        q = pybamm.grad(T)
        S = pybamm.Variable("S", domain=["negative electrode"])
        p = pybamm.grad(S)
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N), T: pybamm.div(q), S: pybamm.div(p)}
        model.initial_conditions = {
            c: pybamm.Scalar(2),
            T: pybamm.Scalar(5),
            S: pybamm.Scalar(8),
        }
        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Neumann")},
            T: {"left": (0, "Neumann"), "right": (0, "Neumann")},
            S: {"left": (0, "Neumann"), "right": (0, "Neumann")},
        }
        model.variables = {"ST": S * T}

        disc.process_model(model)
        y0 = model.concatenated_initial_conditions.evaluate()
        y0_expect = np.empty((0, 1))
        for var_id, _ in sorted(disc.y_slices.items(), key=lambda kv: kv[1]):
            if var_id == c:
                vect = 2 * np.ones_like(submesh.nodes[:, np.newaxis])
            elif var_id == T:
                vect = 5 * np.ones_like(mesh["negative electrode"].nodes[:, np.newaxis])
            else:
                vect = 8 * np.ones_like(mesh["negative electrode"].nodes[:, np.newaxis])

            y0_expect = np.concatenate([y0_expect, vect])

        np.testing.assert_array_equal(y0, y0_expect)

        # grad and div are identity operators here
        np.testing.assert_array_equal(y0, model.concatenated_rhs.evaluate(None, y0))

        S0 = model.initial_conditions[S].evaluate() * np.ones_like(
            mesh[S.domain[0]].nodes[:, np.newaxis]
        )
        T0 = model.initial_conditions[T].evaluate() * np.ones_like(
            mesh[T.domain[0]].nodes[:, np.newaxis]
        )
        np.testing.assert_array_equal(S0 * T0, model.variables["ST"].evaluate(None, y0))

        # mass matrix is identity
        np.testing.assert_array_equal(
            np.eye(np.size(y0)), model.mass_matrix.entries.toarray()
        )

        # Create StateVector to differentiate model with respect to
        y = pybamm.StateVector(slice(0, np.size(y0)))

        # jacobian is identity
        jacobian = model.concatenated_rhs.jac(y).evaluate(0, y0)
        np.testing.assert_array_equal(np.eye(np.size(y0)), jacobian.toarray())

        # test that discretising again gives an error
        with self.assertRaisesRegex(pybamm.ModelError, "Cannot re-discretise a model"):
            disc.process_model(model)

        # test that not enough initial conditions raises an error
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N), T: pybamm.div(q), S: pybamm.div(p)}
        model.initial_conditions = {T: pybamm.Scalar(5), S: pybamm.Scalar(8)}
        model.boundary_conditions = {}
        model.variables = {"ST": S * T}
        with self.assertRaises(pybamm.ModelError):
            disc.process_model(model)

        # test that any time derivatives of variables in rhs raises an
        # error
        model = pybamm.BaseModel()
        model.rhs = {
            c: pybamm.div(N) + c.diff(pybamm.t),
            T: pybamm.div(q),
            S: pybamm.div(p),
        }
        model.initial_conditions = {
            c: pybamm.Scalar(2),
            T: pybamm.Scalar(5),
            S: pybamm.Scalar(8),
        }
        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Neumann")},
            T: {"left": (0, "Neumann"), "right": (0, "Neumann")},
            S: {"left": (0, "Neumann"), "right": (0, "Neumann")},
        }
        model.variables = {"ST": S * T}
        with self.assertRaises(pybamm.ModelError):
            disc.process_model(model)

    def test_process_model_fail(self):
        # one equation
        c = pybamm.Variable("c")
        d = pybamm.Variable("d")
        model = pybamm.BaseModel()
        model.rhs = {c: -c}
        model.initial_conditions = {c: pybamm.Scalar(3)}
        model.variables = {"c": c, "d": d}

        disc = pybamm.Discretisation()
        # turn debug mode off to not check well posedness
        debug_mode = pybamm.settings.debug_mode
        pybamm.settings.debug_mode = False
        with self.assertRaisesRegex(pybamm.ModelError, "No key set for variable"):
            disc.process_model(model)
        pybamm.settings.debug_mode = debug_mode

    def test_process_model_dae(self):
        # one rhs equation and one algebraic
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        d = pybamm.Variable("d", domain=whole_cell)
        N = pybamm.grad(c)
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N)}
        model.algebraic = {d: d - 2 * c}
        model.initial_conditions = {d: pybamm.Scalar(6), c: pybamm.Scalar(3)}

        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        model.variables = {"c": c, "N": N, "d": d}

        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        disc.process_model(model)
        submesh = mesh[whole_cell]

        y0 = model.concatenated_initial_conditions.evaluate()
        np.testing.assert_array_equal(
            y0,
            np.concatenate(
                [
                    3 * np.ones_like(submesh.nodes),
                    6 * np.ones_like(submesh.nodes),
                ]
            )[:, np.newaxis],
        )

        # grad and div are identity operators here
        np.testing.assert_array_equal(
            y0[: submesh.npts], model.concatenated_rhs.evaluate(None, y0)
        )

        np.testing.assert_array_equal(
            model.concatenated_algebraic.evaluate(None, y0),
            np.zeros_like(submesh.nodes[:, np.newaxis]),
        )

        # mass matrix is identity upper left, zeros elsewhere
        mass = block_diag(
            (
                np.eye(np.size(submesh.nodes)),
                np.zeros((np.size(submesh.nodes), np.size(submesh.nodes))),
            )
        )
        np.testing.assert_array_equal(
            mass.toarray(), model.mass_matrix.entries.toarray()
        )

        # jacobian
        y = pybamm.StateVector(slice(0, np.size(y0)))
        jac_rhs = model.concatenated_rhs.jac(y)
        jac_algebraic = model.concatenated_algebraic.jac(y)
        jacobian = pybamm.SparseStack(jac_rhs, jac_algebraic).evaluate(0, y0)

        jacobian_actual = np.block(
            [
                [
                    np.eye(np.size(submesh.nodes)),
                    np.zeros(
                        (
                            np.size(submesh.nodes),
                            np.size(submesh.nodes),
                        )
                    ),
                ],
                [
                    -2 * np.eye(np.size(submesh.nodes)),
                    np.eye(np.size(submesh.nodes)),
                ],
            ]
        )
        np.testing.assert_array_equal(jacobian_actual, jacobian.toarray())

        # check that any time derivatives of variables in algebraic raises an
        # error
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N)}
        model.algebraic = {d: d - 2 * c.diff(pybamm.t)}
        model.initial_conditions = {d: pybamm.Scalar(6), c: pybamm.Scalar(3)}
        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        model.variables = {"c": c, "N": N, "d": d}

        with self.assertRaises(pybamm.ModelError):
            disc.process_model(model)

    def test_process_model_algebraic(self):
        # TODO: implement this based on test_process_model_dae
        # one rhs equation and one algebraic
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        N = pybamm.grad(c)
        Q = pybamm.Scalar(1)
        model = pybamm.BaseModel()
        model.algebraic = {c: pybamm.div(N) - Q}
        model.initial_conditions = {c: pybamm.Scalar(0)}

        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        model.variables = {"c": c, "N": N}

        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        disc.process_model(model)
        submesh = mesh[whole_cell]

        y0 = model.concatenated_initial_conditions.evaluate()
        np.testing.assert_array_equal(
            y0,
            np.zeros_like(submesh.nodes)[:, np.newaxis],
        )

        # grad and div are identity operators here
        np.testing.assert_array_equal(
            model.concatenated_rhs.evaluate(None, y0), np.ones([0, 1])
        )

        np.testing.assert_array_equal(
            model.concatenated_algebraic.evaluate(None, y0),
            -np.ones_like(submesh.nodes[:, np.newaxis]),
        )

        # mass matrix is identity upper left, zeros elsewhere
        mass = np.zeros((np.size(submesh.nodes), np.size(submesh.nodes)))
        np.testing.assert_array_equal(mass, model.mass_matrix.entries.toarray())

        # jacobian
        y = pybamm.StateVector(slice(0, np.size(y0)))
        jacobian = model.concatenated_algebraic.jac(y).evaluate(0, y0)
        np.testing.assert_array_equal(np.eye(submesh.npts), jacobian.toarray())

    def test_process_model_concatenation(self):
        # concatenation of variables as the key
        cn = pybamm.Variable("c", domain=["negative electrode"])
        cs = pybamm.Variable("c", domain=["separator"])
        cp = pybamm.Variable("c", domain=["positive electrode"])
        c = pybamm.concatenation(cn, cs, cp)
        N = pybamm.grad(c)
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N)}
        model.initial_conditions = {c: pybamm.Scalar(3)}

        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        model.check_well_posedness()

        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        submesh = mesh[("negative electrode", "separator", "positive electrode")]

        disc.process_model(model)
        y0 = model.concatenated_initial_conditions.evaluate()
        np.testing.assert_array_equal(
            y0, 3 * np.ones_like(submesh.nodes[:, np.newaxis])
        )

        # grad and div are identity operators here
        np.testing.assert_array_equal(y0, model.concatenated_rhs.evaluate(None, y0))
        model.check_well_posedness()

    def test_process_model_not_inplace(self):
        # concatenation of variables as the key
        c = pybamm.Variable("c", domain=["negative electrode"])
        N = pybamm.grad(c)
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N)}
        model.initial_conditions = {c: pybamm.Scalar(3)}
        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        model.check_well_posedness()

        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        submesh = mesh["negative electrode"]

        discretised_model = disc.process_model(model, inplace=False)
        y0 = discretised_model.concatenated_initial_conditions.evaluate()
        np.testing.assert_array_equal(
            y0, 3 * np.ones_like(submesh.nodes[:, np.newaxis])
        )

        # grad and div are identity operators here
        np.testing.assert_array_equal(
            y0, discretised_model.concatenated_rhs.evaluate(None, y0)
        )
        discretised_model.check_well_posedness()

    def test_initial_condition_bounds(self):
        # concatenation of variables as the key
        c = pybamm.Variable("c", bounds=(0, 1))
        model = pybamm.BaseModel()
        model.rhs = {c: 1}
        model.initial_conditions = {c: pybamm.Scalar(3)}

        disc = pybamm.Discretisation()
        with self.assertRaisesRegex(
            pybamm.ModelError, "initial condition is outside of variable bounds"
        ):
            disc.process_model(model)

    def test_process_empty_model(self):
        model = pybamm.BaseModel()
        disc = pybamm.Discretisation()
        with self.assertRaisesRegex(pybamm.ModelError, "Cannot discretise empty model"):
            disc.process_model(model)

    def test_broadcast(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]

        a = pybamm.InputParameter("a")
        var = pybamm.Variable("var")

        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        submesh = mesh[whole_cell]

        # scalar
        broad = disc.process_symbol(pybamm.FullBroadcast(a, whole_cell, {}))
        np.testing.assert_array_equal(
            broad.evaluate(inputs={"a": 7}),
            7 * np.ones_like(submesh.nodes[:, np.newaxis]),
        )
        self.assertEqual(broad.domain, whole_cell)

        broad_disc = disc.process_symbol(broad)
        self.assertIsInstance(broad_disc, pybamm.Multiplication)
        self.assertIsInstance(broad_disc.children[0], pybamm.Vector)
        self.assertIsInstance(broad_disc.children[1], pybamm.InputParameter)

        # process Broadcast variable
        disc.y_slices = {var: [slice(1)]}
        broad1 = pybamm.FullBroadcast(var, ["negative electrode"], None)
        broad1_disc = disc.process_symbol(broad1)
        self.assertIsInstance(broad1_disc, pybamm.Multiplication)
        self.assertIsInstance(broad1_disc.children[0], pybamm.Vector)
        self.assertIsInstance(broad1_disc.children[1], pybamm.StateVector)

        # broadcast to edges
        broad_to_edges = pybamm.FullBroadcastToEdges(a, ["negative electrode"], None)
        broad_to_edges_disc = disc.process_symbol(broad_to_edges)
        np.testing.assert_array_equal(
            broad_to_edges_disc.evaluate(inputs={"a": 7}),
            7 * np.ones_like(mesh["negative electrode"].edges[:, np.newaxis]),
        )

    def test_broadcast_2D(self):
        # broadcast in 2D --> MatrixMultiplication
        var = pybamm.Variable("var", ["current collector"])
        disc = get_1p1d_discretisation_for_testing()
        mesh = disc.mesh
        broad = pybamm.PrimaryBroadcast(var, "separator")

        disc.set_variable_slices([var])
        broad_disc = disc.process_symbol(broad)
        self.assertIsInstance(broad_disc, pybamm.MatrixMultiplication)
        self.assertIsInstance(broad_disc.children[0], pybamm.Matrix)
        self.assertIsInstance(broad_disc.children[1], pybamm.StateVector)
        self.assertEqual(
            broad_disc.shape,
            (mesh["separator"].npts * mesh["current collector"].npts, 1),
        )
        y_test = np.linspace(0, 1, mesh["current collector"].npts)
        np.testing.assert_array_equal(
            broad_disc.evaluate(y=y_test),
            np.outer(y_test, np.ones(mesh["separator"].npts)).reshape(-1, 1),
        )

        # test broadcast to edges
        broad_to_edges = pybamm.PrimaryBroadcastToEdges(var, "separator")
        broad_to_edges_disc = disc.process_symbol(broad_to_edges)
        self.assertIsInstance(broad_to_edges_disc, pybamm.MatrixMultiplication)
        self.assertIsInstance(broad_to_edges_disc.children[0], pybamm.Matrix)
        self.assertIsInstance(broad_to_edges_disc.children[1], pybamm.StateVector)
        self.assertEqual(
            broad_to_edges_disc.shape,
            ((mesh["separator"].npts + 1) * mesh["current collector"].npts, 1),
        )
        y_test = np.linspace(0, 1, mesh["current collector"].npts)
        np.testing.assert_array_equal(
            broad_to_edges_disc.evaluate(y=y_test),
            np.outer(y_test, np.ones(mesh["separator"].npts + 1)).reshape(-1, 1),
        )

    def test_secondary_broadcast_2D(self):
        # secondary broadcast in 2D --> Matrix multiplication
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        var = pybamm.Variable("var", domain=["negative particle"])
        broad = pybamm.SecondaryBroadcast(var, "negative electrode")

        disc.set_variable_slices([var])
        broad_disc = disc.process_symbol(broad)
        self.assertIsInstance(broad_disc, pybamm.MatrixMultiplication)
        self.assertIsInstance(broad_disc.children[0], pybamm.Matrix)
        self.assertIsInstance(broad_disc.children[1], pybamm.StateVector)
        self.assertEqual(
            broad_disc.shape,
            (mesh["negative particle"].npts * mesh["negative electrode"].npts, 1),
        )
        broad = pybamm.SecondaryBroadcast(var, "negative electrode")

        # test broadcast to edges
        broad_to_edges = pybamm.SecondaryBroadcastToEdges(var, "negative electrode")
        disc.set_variable_slices([var])
        broad_to_edges_disc = disc.process_symbol(broad_to_edges)
        self.assertIsInstance(broad_to_edges_disc, pybamm.MatrixMultiplication)
        self.assertIsInstance(broad_to_edges_disc.children[0], pybamm.Matrix)
        self.assertIsInstance(broad_to_edges_disc.children[1], pybamm.StateVector)
        self.assertEqual(
            broad_to_edges_disc.shape,
            (mesh["negative particle"].npts * (mesh["negative electrode"].npts + 1), 1),
        )

    def test_tertiary_broadcast_3D(self):
        disc = get_1p1d_discretisation_for_testing()
        mesh = disc.mesh
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )
        broad = pybamm.TertiaryBroadcast(var, "current collector")

        disc.set_variable_slices([var])
        broad_disc = disc.process_symbol(broad)
        self.assertIsInstance(broad_disc, pybamm.MatrixMultiplication)
        self.assertIsInstance(broad_disc.children[0], pybamm.Matrix)
        self.assertIsInstance(broad_disc.children[1], pybamm.StateVector)
        self.assertEqual(
            broad_disc.shape,
            (
                mesh["negative particle"].npts
                * mesh["negative electrode"].npts
                * mesh["current collector"].npts,
                1,
            ),
        )

        # test broadcast to edges
        broad_to_edges = pybamm.TertiaryBroadcastToEdges(var, "current collector")
        disc.set_variable_slices([var])
        broad_to_edges_disc = disc.process_symbol(broad_to_edges)
        self.assertIsInstance(broad_to_edges_disc, pybamm.MatrixMultiplication)
        self.assertIsInstance(broad_to_edges_disc.children[0], pybamm.Matrix)
        self.assertIsInstance(broad_to_edges_disc.children[1], pybamm.StateVector)
        self.assertEqual(
            broad_to_edges_disc.shape,
            (
                mesh["negative particle"].npts
                * mesh["negative electrode"].npts
                * (mesh["current collector"].npts + 1),
                1,
            ),
        )

    def test_concatenation(self):
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")

        # create discretisation
        disc = get_discretisation_for_testing()

        conc = disc.concatenate(a, b, c)
        self.assertIsInstance(conc, pybamm.NumpyConcatenation)

    def test_concatenation_of_scalars(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        a = pybamm.PrimaryBroadcast(5, ["negative electrode"])
        b = pybamm.PrimaryBroadcast(4, ["separator"])

        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        variables = [pybamm.Variable("var", domain=whole_cell)]
        disc.set_variable_slices(variables)

        eqn = pybamm.concatenation(a, b)
        eqn_disc = disc.process_symbol(eqn)
        expected_vector = np.concatenate(
            [
                5 * np.ones_like(mesh["negative electrode"].nodes),
                4 * np.ones_like(mesh["separator"].nodes),
            ]
        )[:, np.newaxis]
        np.testing.assert_allclose(eqn_disc.evaluate(), expected_vector)

    def test_concatenation_2D(self):
        disc = get_1p1d_discretisation_for_testing(zpts=3)

        a = pybamm.Variable(
            "a",
            domain=["negative electrode"],
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, 1),
        )
        b = pybamm.Variable(
            "b",
            domain=["separator"],
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, 1),
        )
        c = pybamm.Variable(
            "c",
            domain=["positive electrode"],
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, 1),
        )

        # With simplification
        conc = pybamm.concatenation(a, b, c)
        disc.set_variable_slices([conc])
        self.assertEqual(
            disc.y_slices[a], [slice(0, 40), slice(100, 140), slice(200, 240)]
        )
        self.assertEqual(
            disc.y_slices[b], [slice(40, 65), slice(140, 165), slice(240, 265)]
        )
        self.assertEqual(
            disc.y_slices[c], [slice(65, 100), slice(165, 200), slice(265, 300)]
        )
        expr = disc.process_symbol(conc)
        self.assertIsInstance(expr, pybamm.StateVector)

        # Evaulate
        y = np.linspace(0, 1, 300)
        self.assertEqual(expr.evaluate(0, y).shape, (120 + 75 + 105, 1))
        np.testing.assert_equal(expr.evaluate(0, y), y[:, np.newaxis])

        # Without simplification
        conc = pybamm.concatenation(2 * a, 3 * b, 4 * c)
        conc.bounds = (-np.inf, np.inf)
        disc.set_variable_slices([a, b, c])
        expr = disc.process_symbol(conc)
        self.assertIsInstance(expr, pybamm.DomainConcatenation)

        # Evaulate
        y = np.linspace(0, 1, 300)
        self.assertEqual(expr.children[0].evaluate(0, y).shape, (120, 1))
        self.assertEqual(expr.children[1].evaluate(0, y).shape, (75, 1))
        self.assertEqual(expr.children[2].evaluate(0, y).shape, (105, 1))

    def test_exceptions(self):
        c_n = pybamm.Variable("c", domain=["negative electrode"])
        N_n = pybamm.grad(c_n)
        c_s = pybamm.Variable("c", domain=["separator"])
        N_s = pybamm.grad(c_s)
        model = pybamm.BaseModel()
        model.rhs = {c_n: pybamm.div(N_n), c_s: pybamm.div(N_s)}
        model.initial_conditions = {c_n: pybamm.Scalar(3), c_s: pybamm.Scalar(1)}
        model.boundary_conditions = {
            c_n: {"left": (0, "Neumann"), "right": (0, "Neumann")},
            c_s: {"left": (0, "Neumann"), "right": (0, "Neumann")},
        }

        disc = get_discretisation_for_testing()

        # check raises error if different sized key and output var
        model.variables = {c_n.name: c_s}
        with self.assertRaisesRegex(pybamm.ModelError, "variable and its eqn"):
            disc.process_model(model)

        # check doesn't raise if concatenation
        model.variables = {c_n.name: pybamm.concatenation(2 * c_n, 3 * c_s)}
        disc.process_model(model, inplace=False)

        # check doesn't raise if broadcast
        model.variables = {
            c_n.name: pybamm.PrimaryBroadcast(
                pybamm.InputParameter("a"), ["negative electrode"]
            )
        }
        disc.process_model(model)

        # Check setting up a 0D spatial method with 1D mesh raises error
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.ZeroDimensionalSpatialMethod(),
        }
        with self.assertRaisesRegex(
            pybamm.DiscretisationError, "Zero-dimensional spatial method for the "
        ):
            pybamm.Discretisation(mesh, spatial_methods)

    def test_check_tab_bcs_error(self):
        a = pybamm.Variable("a", domain=["current collector"])
        b = pybamm.Variable("b", domain=["negative electrode"])
        bcs = {"negative tab": (0, "Dirichlet"), "positive tab": (0, "Neumann")}

        disc = get_discretisation_for_testing()

        # for 0D bcs keys should be unchanged
        new_bcs = disc.check_tab_conditions(a, bcs)
        self.assertListEqual(list(bcs.keys()), list(new_bcs.keys()))

        # error if domain not "current collector"
        with self.assertRaisesRegex(pybamm.ModelError, "Boundary conditions"):
            disc.check_tab_conditions(b, bcs)

    def test_process_with_no_check(self):
        # create model
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        N = pybamm.grad(c)
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N)}
        model.initial_conditions = {c: pybamm.Scalar(3)}
        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        model.variables = {"c": c, "N": N}

        # create discretisation
        disc = get_discretisation_for_testing()
        disc.process_model(model, check_model=False)

    def test_mass_matrix_inverse(self):
        # get mesh
        mesh = get_2p1d_mesh_for_testing(ypts=5, zpts=5)
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "current collector": pybamm.ScikitFiniteElement(),
        }
        # create model
        a = pybamm.Variable(
            "a",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        b = pybamm.Variable("b", domain="current collector")
        model = pybamm.BaseModel()
        model.rhs = {a: pybamm.Laplacian(a), b: 4 * pybamm.Laplacian(b)}
        model.initial_conditions = {a: pybamm.Scalar(3), b: pybamm.Scalar(10)}
        model.boundary_conditions = {
            a: {"left": (0, "Neumann"), "right": (0, "Neumann")},
            b: {"negative tab": (0, "Neumann"), "positive tab": (0, "Neumann")},
        }
        model.variables = {"a": a, "b": b}

        # create discretisation
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        # test that computing mass matrix block-by-block (as is done during
        # discretisation) gives the correct result
        # Note: inverse is more efficient in csc format
        mass_inv = inv(csc_matrix(model.mass_matrix.entries))
        np.testing.assert_equal(
            model.mass_matrix_inv.entries.toarray(), mass_inv.toarray()
        )

    def test_process_input_variable(self):
        disc = get_discretisation_for_testing()

        a = pybamm.InputParameter("a")
        a_disc = disc.process_symbol(a)
        self.assertEqual(a_disc._expected_size, 1)

        a = pybamm.InputParameter("a", ["negative electrode", "separator"])
        a_disc = disc.process_symbol(a)
        n = disc.mesh[a.domain].npts
        self.assertEqual(a_disc._expected_size, n)

    def test_process_not_constant(self):
        disc = pybamm.Discretisation()

        a = pybamm.NotConstant(pybamm.Scalar(1))
        self.assertEqual(disc.process_symbol(a), pybamm.Scalar(1))
        self.assertEqual(disc.process_symbol(2 * a), pybamm.Scalar(2))

    def test_bc_symmetry(self):
        # define model
        model = pybamm.BaseModel()
        c = pybamm.Variable("Concentration", domain="negative particle")
        N = -pybamm.grad(c)
        dcdt = -pybamm.div(N)
        model.rhs = {c: dcdt}

        # initial conditions
        model.initial_conditions = {c: pybamm.Scalar(1)}

        # define geometry
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )
        geometry = {
            "negative particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }

        # mesh
        submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
        var_pts = {r: 20}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        spatial_methods = {"negative particle": pybamm.FiniteVolume()}

        # boundary conditions (Dirichlet)
        lbc = pybamm.Scalar(0)
        rbc = pybamm.Scalar(2)
        model.boundary_conditions = {
            c: {"left": (lbc, "Dirichlet"), "right": (rbc, "Neumann")}
        }

        # discretise
        disc = pybamm.Discretisation(mesh, spatial_methods)
        with self.assertRaisesRegex(pybamm.ModelError, "Boundary condition at r = 0"):
            disc.process_model(model)

        # boundary conditions (non-homog Neumann)
        lbc = pybamm.Scalar(0)
        rbc = pybamm.Scalar(2)
        model.boundary_conditions = {
            c: {"left": (rbc, "Neumann"), "right": (rbc, "Neumann")}
        }

        # discretise
        disc = pybamm.Discretisation(mesh, spatial_methods)
        with self.assertRaisesRegex(pybamm.ModelError, "Boundary condition at r = 0"):
            disc.process_model(model)

    def test_check_model_errors(self):
        disc = pybamm.Discretisation()
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: pybamm.Vector([1, 1])}
        model.initial_conditions = {var: 1}
        with self.assertRaisesRegex(
            pybamm.ModelError, "initial conditions must be numpy array"
        ):
            disc.check_model(model)
        model.initial_conditions = {var: pybamm.Vector([1, 1, 1])}
        with self.assertRaisesRegex(
            pybamm.ModelError, "rhs and initial conditions must have the same shape"
        ):
            disc.check_model(model)
        model.rhs = {}
        model.algebraic = {var: pybamm.Vector([1, 1])}
        with self.assertRaisesRegex(
            pybamm.ModelError,
            "algebraic and initial conditions must have the same shape",
        ):
            disc.check_model(model)

    def test_length_scale_errors(self):
        disc = pybamm.Discretisation()
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: pybamm.Scalar(1)}
        model.initial_conditions = {var: pybamm.Scalar(1)}
        disc.process_model(model)

    def test_independent_rhs(self):
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        c = pybamm.Variable("c")
        model = pybamm.BaseModel()
        model.rhs = {a: b, b: c, c: -c}
        model.initial_conditions = {
            a: pybamm.Scalar(0),
            b: pybamm.Scalar(1),
            c: pybamm.Scalar(1),
        }
        disc = pybamm.Discretisation()
        disc.process_model(model)
        self.assertEqual(len(model.rhs), 2)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
