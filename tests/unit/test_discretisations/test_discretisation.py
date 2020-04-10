#
# Tests for the base model class
#
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


class TestDiscretise(unittest.TestCase):
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

        disc.y_slices = {c.id: [slice(0, 1)], a.id: [slice(2, 3)], b.id: [slice(3, 4)]}
        result = disc._concatenate_in_order(initial_conditions)

        self.assertIsInstance(result, pybamm.NumpyConcatenation)
        self.assertEqual(result.children[0].evaluate(), 1)
        self.assertEqual(result.children[1].evaluate(), 2)
        self.assertEqual(result.children[2].evaluate(), 3)

        initial_conditions = {a: pybamm.Scalar(2), b: pybamm.Scalar(3)}
        with self.assertRaises(pybamm.ModelError):
            result = disc._concatenate_in_order(initial_conditions, check_complete=True)

    def test_no_mesh(self):
        disc = pybamm.Discretisation(None, None)
        self.assertEqual(disc._spatial_methods, {})

    def test_add_internal_boundary_conditions(self):
        model = pybamm.BaseModel()
        c_e_n = pybamm.PrimaryBroadcast(0, ["negative electrode"])
        c_e_s = pybamm.PrimaryBroadcast(0, ["separator"])
        c_e_p = pybamm.PrimaryBroadcast(0, ["positive electrode"])
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)
        lbc = (pybamm.Scalar(0), "Neumann")
        rbc = (pybamm.Scalar(0), "Neumann")
        model.boundary_conditions = {c_e: {"left": lbc, "right": rbc}}

        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": SpatialMethodForTesting()}

        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.bcs = disc.process_boundary_conditions(model)
        disc.set_internal_boundary_conditions(model)

        for child in c_e.children:
            self.assertTrue(child.id in disc.bcs.keys())

    def test_adding_0D_external_variable(self):
        model = pybamm.BaseModel()
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")

        model.rhs = {a: a * b}
        model.initial_conditions = {a: 0}
        model.external_variables = [b]
        model.variables = {"a": a, "b": b, "c": a * b}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        self.assertIsInstance(model.variables["b"], pybamm.ExternalVariable)
        self.assertEqual(model.variables["b"].evaluate(inputs={"b": np.array([1])}), 1)

    def test_adding_0D_external_variable_fail(self):
        model = pybamm.BaseModel()
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")

        model.rhs = {a: a * b}
        model.initial_conditions = {a: 0}
        model.external_variables = [b]

        disc = pybamm.Discretisation()
        with self.assertRaisesRegex(ValueError, "Variable b must be in the model"):
            disc.process_model(model)

    def test_adding_1D_external_variable(self):
        model = pybamm.BaseModel()

        a = pybamm.Variable("a", domain=["test"])
        b = pybamm.Variable("b", domain=["test"])

        model.rhs = {a: a * b}
        model.boundary_conditions = {
            a: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        model.initial_conditions = {a: 0}
        model.external_variables = [b]
        model.variables = {
            "a": a,
            "b": b,
            "c": a * b,
            "grad b": pybamm.grad(b),
            "div grad b": pybamm.div(pybamm.grad(b)),
        }

        x = pybamm.SpatialVariable("x", domain="test", coord_sys="cartesian")
        geometry = {
            "test": {"primary": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}
        }

        submesh_types = {"test": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)}
        var_pts = {x: 10}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        spatial_methods = {"test": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        self.assertEqual(disc.y_slices[a.id][0], slice(0, 10, None))

        self.assertEqual(model.y_slices[a][0], slice(0, 10, None))

        b_test = np.ones((10, 1))
        np.testing.assert_array_equal(
            model.variables["b"].evaluate(inputs={"b": b_test}), b_test
        )

        # check that b is added to the boundary conditions
        model.bcs[b.id]["left"]
        model.bcs[b.id]["right"]

        # check that grad and div(grad ) produce the correct shapes
        self.assertEqual(model.variables["b"].shape_for_testing, (10, 1))
        self.assertEqual(model.variables["grad b"].shape_for_testing, (11, 1))
        self.assertEqual(model.variables["div grad b"].shape_for_testing, (10, 1))

    def test_concatenation_external_variables(self):
        model = pybamm.BaseModel()

        a = pybamm.Variable("a", domain=["test", "test1"])
        b1 = pybamm.Variable("b", domain=["test"])
        b2 = pybamm.Variable("c", domain=["test1"])
        b = pybamm.Concatenation(b1, b2)

        model.rhs = {a: a * b}
        model.boundary_conditions = {
            a: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        model.initial_conditions = {a: 0}
        model.external_variables = [b]
        model.variables = {
            "a": a,
            "b": b,
            "b1": b1,
            "b2": b2,
            "c": a * b,
            "grad b": pybamm.grad(b),
            "div grad b": pybamm.div(pybamm.grad(b)),
        }

        x = pybamm.SpatialVariable("x", domain="test", coord_sys="cartesian")
        y = pybamm.SpatialVariable("y", domain="test1", coord_sys="cartesian")
        geometry = {
            "test": {
                "primary": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
            },
            "test1": {
                "primary": {y: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}}
            },
        }

        submesh_types = {
            "test": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "test1": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        }
        var_pts = {x: 10, y: 5}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        spatial_methods = {
            "test": pybamm.FiniteVolume(),
            "test1": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        self.assertEqual(disc.y_slices[a.id][0], slice(0, 15, None))

        b_test = np.linspace(0, 1, 15)[:, np.newaxis]
        np.testing.assert_array_equal(
            model.variables["b"].evaluate(inputs={"b": b_test}), b_test
        )
        np.testing.assert_array_equal(
            model.variables["b1"].evaluate(inputs={"b": b_test}), b_test[:10]
        )
        np.testing.assert_array_equal(
            model.variables["b2"].evaluate(inputs={"b": b_test}), b_test[10:]
        )

        # check that b is added to the boundary conditions
        model.bcs[b.id]["left"]
        model.bcs[b.id]["right"]

        # check that grad and div(grad ) produce the correct shapes
        self.assertEqual(model.variables["b"].shape_for_testing, (15, 1))
        self.assertEqual(model.variables["grad b"].shape_for_testing, (16, 1))
        self.assertEqual(model.variables["div grad b"].shape_for_testing, (15, 1))
        self.assertEqual(model.variables["b1"].shape_for_testing, (10, 1))
        self.assertEqual(model.variables["b2"].shape_for_testing, (5, 1))

    def test_adding_2D_external_variable_fail(self):
        model = pybamm.BaseModel()
        a = pybamm.Variable(
            "a",
            domain=["negative electrode", "separator"],
            auxiliary_domains={"secondary": "current collector"},
        )
        b1 = pybamm.Variable(
            "b",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        b2 = pybamm.Variable(
            "b",
            domain="separator",
            auxiliary_domains={"secondary": "current collector"},
        )
        b = pybamm.Concatenation(b1, b2)

        model.rhs = {a: a * b}
        model.initial_conditions = {a: 0}
        model.external_variables = [b]
        model.variables = {"b": b}

        disc = get_1p1d_discretisation_for_testing()
        with self.assertRaisesRegex(
            NotImplementedError, "Cannot create 2D external variable"
        ):
            disc.process_model(model)

    def test_discretise_slicing(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        variables = [c]
        disc.set_variable_slices(variables)

        self.assertEqual(disc.y_slices, {c.id: [slice(0, 100)]})

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        c_true = combined_submesh[0].nodes ** 2
        y = c_true
        np.testing.assert_array_equal(y[disc.y_slices[c.id][0]], c_true)

        # Several variables
        d = pybamm.Variable("d", domain=whole_cell)
        jn = pybamm.Variable("jn", domain=["negative electrode"])
        variables = [c, d, jn]
        disc.set_variable_slices(variables)

        self.assertEqual(
            disc.y_slices,
            {c.id: [slice(0, 100)], d.id: [slice(100, 200)], jn.id: [slice(200, 240)]},
        )
        d_true = 4 * combined_submesh[0].nodes
        jn_true = mesh["negative electrode"][0].nodes ** 3
        y = np.concatenate([c_true, d_true, jn_true])
        np.testing.assert_array_equal(y[disc.y_slices[c.id][0]], c_true)
        np.testing.assert_array_equal(y[disc.y_slices[d.id][0]], d_true)
        np.testing.assert_array_equal(y[disc.y_slices[jn.id][0]], jn_true)

        # Variables with a concatenation
        js = pybamm.Variable("js", domain=["separator"])
        jp = pybamm.Variable("jp", domain=["positive electrode"])
        j = pybamm.Concatenation(jn, js, jp)
        variables = [c, d, j]
        disc.set_variable_slices(variables)
        self.assertEqual(
            disc.y_slices,
            {
                c.id: [slice(0, 100)],
                d.id: [slice(100, 200)],
                jn.id: [slice(200, 240)],
                js.id: [slice(240, 265)],
                jp.id: [slice(265, 300)],
            },
        )
        d_true = 4 * combined_submesh[0].nodes
        jn_true = mesh["negative electrode"][0].nodes ** 3
        y = np.concatenate([c_true, d_true, jn_true])
        np.testing.assert_array_equal(y[disc.y_slices[c.id][0]], c_true)
        np.testing.assert_array_equal(y[disc.y_slices[d.id][0]], d_true)
        np.testing.assert_array_equal(y[disc.y_slices[jn.id][0]], jn_true)

        with self.assertRaisesRegex(TypeError, "y_slices should be"):
            disc.y_slices = 1

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
        disc.y_slices = {var.id: [slice(53)], var_vec.id: [slice(53, 93)]}
        var_disc = disc.process_symbol(var)
        self.assertIsInstance(var_disc, pybamm.StateVector)
        self.assertEqual(var_disc.y_slices[0], disc.y_slices[var.id][0])

        # variable dot
        var_dot = pybamm.VariableDot("var'")
        var_dot_disc = disc.process_symbol(var_dot)
        self.assertIsInstance(var_dot_disc, pybamm.StateVectorDot)
        self.assertEqual(var_dot_disc.y_slices[0], disc.y_slices[var.id][0])

        # scalar
        scal = pybamm.Scalar(5)
        scal_disc = disc.process_symbol(scal)
        self.assertIsInstance(scal_disc, pybamm.Scalar)
        self.assertEqual(scal_disc.value, scal.value)
        # vector
        vec = pybamm.Vector(np.array([1, 2, 3, 4]))
        vec_disc = disc.process_symbol(vec)
        self.assertIsInstance(vec_disc, pybamm.Vector)
        np.testing.assert_array_equal(vec_disc.entries, vec.entries)
        # matrix
        mat = pybamm.Matrix(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
        mat_disc = disc.process_symbol(mat)
        self.assertIsInstance(mat_disc, pybamm.Matrix)
        np.testing.assert_array_equal(mat_disc.entries, mat.entries)

        # binary operator
        bin = var + scal
        bin_disc = disc.process_symbol(bin)
        self.assertIsInstance(bin_disc, pybamm.Addition)
        self.assertIsInstance(bin_disc.children[0], pybamm.StateVector)
        self.assertIsInstance(bin_disc.children[1], pybamm.Scalar)

        bin2 = scal + var
        bin2_disc = disc.process_symbol(bin2)
        self.assertIsInstance(bin2_disc, pybamm.Addition)
        self.assertIsInstance(bin2_disc.children[0], pybamm.Scalar)
        self.assertIsInstance(bin2_disc.children[1], pybamm.StateVector)

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

        func = pybamm.Function(myfun, scal)
        func_disc = disc.process_symbol(func)
        self.assertIsInstance(func_disc, pybamm.Function)
        self.assertIsInstance(func_disc.children[0], pybamm.Scalar)

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
        scal1 = pybamm.Scalar(1)
        scal2 = pybamm.Scalar(2)
        scal3 = pybamm.Scalar(3)
        scal4 = pybamm.Scalar(4)
        expression = (scal1 * (scal3 + var2)) / ((var1 - scal4) + scal2)

        # create discretisation
        disc = get_discretisation_for_testing()

        disc.y_slices = {var1.id: [slice(53)], var2.id: [slice(53, 106)]}
        exp_disc = disc.process_symbol(expression)
        self.assertIsInstance(exp_disc, pybamm.Division)
        # left side
        self.assertIsInstance(exp_disc.children[0], pybamm.Multiplication)
        self.assertIsInstance(exp_disc.children[0].children[0], pybamm.Scalar)
        self.assertIsInstance(exp_disc.children[0].children[1], pybamm.Addition)
        self.assertTrue(
            isinstance(exp_disc.children[0].children[1].children[0], pybamm.Scalar)
        )
        self.assertTrue(
            isinstance(exp_disc.children[0].children[1].children[1], pybamm.StateVector)
        )
        self.assertEqual(
            exp_disc.children[0].children[1].children[1].y_slices[0],
            disc.y_slices[var2.id][0],
        )
        # right side
        self.assertIsInstance(exp_disc.children[1], pybamm.Addition)
        self.assertTrue(
            isinstance(exp_disc.children[1].children[0], pybamm.Subtraction)
        )
        self.assertTrue(
            isinstance(exp_disc.children[1].children[0].children[0], pybamm.StateVector)
        )
        self.assertEqual(
            exp_disc.children[1].children[0].children[0].y_slices[0],
            disc.y_slices[var1.id][0],
        )
        self.assertTrue(
            isinstance(exp_disc.children[1].children[0].children[1], pybamm.Scalar)
        )
        self.assertIsInstance(exp_disc.children[1].children[1], pybamm.Scalar)

    def test_discretise_spatial_operator(self):
        # create discretisation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        variables = [var]
        disc.set_variable_slices(variables)

        # Simple expressions
        for eqn in [pybamm.grad(var), pybamm.div(var)]:
            eqn_disc = disc.process_symbol(eqn)

            self.assertIsInstance(eqn_disc, pybamm.MatrixMultiplication)
            self.assertIsInstance(eqn_disc.children[0], pybamm.Matrix)
            self.assertIsInstance(eqn_disc.children[1], pybamm.StateVector)

            combined_submesh = mesh.combine_submeshes(*whole_cell)
            y = combined_submesh[0].nodes ** 2
            var_disc = disc.process_symbol(var)
            # grad and var are identity operators here (for testing purposes)
            np.testing.assert_array_equal(
                eqn_disc.evaluate(None, y), var_disc.evaluate(None, y)
            )

        # More complex expressions
        for eqn in [var * pybamm.grad(var), var * pybamm.div(var)]:
            eqn_disc = disc.process_symbol(eqn)

            self.assertIsInstance(eqn_disc, pybamm.Multiplication)
            self.assertIsInstance(eqn_disc.children[0], pybamm.StateVector)
            self.assertIsInstance(eqn_disc.children[1], pybamm.MatrixMultiplication)
            self.assertIsInstance(eqn_disc.children[1].children[0], pybamm.Matrix)
            self.assertIsInstance(eqn_disc.children[1].children[1], pybamm.StateVector)

            y = combined_submesh[0].nodes ** 2
            var_disc = disc.process_symbol(var)
            # grad and var are identity operators here (for testing purposes)
            np.testing.assert_array_equal(
                eqn_disc.evaluate(None, y), var_disc.evaluate(None, y) ** 2
            )

    def test_process_dict(self):
        # one equation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        N = pybamm.grad(c)
        rhs = {c: pybamm.div(N)}
        initial_conditions = {c: pybamm.Scalar(3)}
        variables = {"c_squared": c ** 2}
        boundary_conditions = {c.id: {"left": (0, "Neumann"), "right": (0, "Neumann")}}

        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        y = combined_submesh[0].nodes[:, np.newaxis] ** 2
        disc.bcs = boundary_conditions

        disc.set_variable_slices(list(rhs.keys()))
        # rhs - grad and div are identity operators here
        processed_rhs = disc.process_dict(rhs)
        np.testing.assert_array_equal(y, processed_rhs[c].evaluate(None, y))
        # initial conditions
        y0 = disc.process_dict(initial_conditions)
        np.testing.assert_array_equal(
            y0[c].evaluate(0, None),
            3 * np.ones_like(combined_submesh[0].nodes[:, np.newaxis]),
        )
        # vars
        processed_vars = disc.process_dict(variables)
        np.testing.assert_array_equal(
            y ** 2, processed_vars["c_squared"].evaluate(None, y)
        )

        # two equations
        T = pybamm.Variable("T", domain=["negative electrode"])
        q = pybamm.grad(T)
        rhs = {c: pybamm.div(N), T: pybamm.div(q)}
        initial_conditions = {c: pybamm.Scalar(3), T: pybamm.Scalar(5)}
        boundary_conditions = {}
        y = np.concatenate(
            [combined_submesh[0].nodes ** 2, mesh["negative electrode"][0].nodes ** 4]
        )[:, np.newaxis]

        variables = list(rhs.keys())
        disc.set_variable_slices(variables)
        # rhs
        processed_rhs = disc.process_dict(rhs)
        np.testing.assert_array_equal(
            y[disc.y_slices[c.id][0]], processed_rhs[c].evaluate(None, y)
        )
        np.testing.assert_array_equal(
            y[disc.y_slices[T.id][0]], processed_rhs[T].evaluate(None, y)
        )
        # initial conditions
        y0 = disc.process_dict(initial_conditions)
        np.testing.assert_array_equal(
            y0[c].evaluate(0, None),
            3 * np.ones_like(combined_submesh[0].nodes[:, np.newaxis]),
        )
        np.testing.assert_array_equal(
            y0[T].evaluate(0, None),
            5 * np.ones_like(mesh["negative electrode"][0].nodes[:, np.newaxis]),
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

        combined_submesh = mesh.combine_submeshes(*whole_cell)
        disc.process_model(model)

        y0 = model.concatenated_initial_conditions.evaluate()
        np.testing.assert_array_equal(
            y0, 3 * np.ones_like(combined_submesh[0].nodes[:, np.newaxis])
        )
        np.testing.assert_array_equal(y0, model.concatenated_rhs.evaluate(None, y0))

        # grad and div are identity operators here
        np.testing.assert_array_equal(y0, model.variables["c"].evaluate(None, y0))
        np.testing.assert_array_equal(y0, model.variables["N"].evaluate(None, y0))

        # mass matrix is identity
        np.testing.assert_array_equal(
            np.eye(combined_submesh[0].nodes.shape[0]),
            model.mass_matrix.entries.toarray(),
        )

        # Create StateVector to differentiate model with respect to
        y = pybamm.StateVector(slice(0, combined_submesh[0].npts))

        # jacobian is identity
        jacobian = model.concatenated_rhs.jac(y).evaluate(0, y0)
        np.testing.assert_array_equal(
            np.eye(combined_submesh[0].npts), jacobian.toarray()
        )

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
            if var_id == c.id:
                vect = 2 * np.ones_like(combined_submesh[0].nodes[:, np.newaxis])
            elif var_id == T.id:
                vect = 5 * np.ones_like(
                    mesh["negative electrode"][0].nodes[:, np.newaxis]
                )
            else:
                vect = 8 * np.ones_like(
                    mesh["negative electrode"][0].nodes[:, np.newaxis]
                )

            y0_expect = np.concatenate([y0_expect, vect])

        np.testing.assert_array_equal(y0, y0_expect)

        # grad and div are identity operators here
        np.testing.assert_array_equal(y0, model.concatenated_rhs.evaluate(None, y0))

        S0 = model.initial_conditions[S].evaluate() * np.ones_like(
            mesh[S.domain[0]][0].nodes[:, np.newaxis]
        )
        T0 = model.initial_conditions[T].evaluate() * np.ones_like(
            mesh[T.domain[0]][0].nodes[:, np.newaxis]
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

        # test jacobian by eqn gives same as jacobian of concatenated rhs
        model.jacobian, _, _ = disc.create_jacobian(model)
        model_jacobian = model.jacobian.evaluate(0, y0)
        np.testing.assert_array_equal(model_jacobian.toarray(), jacobian.toarray())

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
        combined_submesh = mesh.combine_submeshes(*whole_cell)

        y0 = model.concatenated_initial_conditions.evaluate()
        np.testing.assert_array_equal(
            y0,
            np.concatenate(
                [
                    3 * np.ones_like(combined_submesh[0].nodes),
                    6 * np.ones_like(combined_submesh[0].nodes),
                ]
            )[:, np.newaxis],
        )

        # grad and div are identity operators here
        np.testing.assert_array_equal(
            y0[: combined_submesh[0].npts], model.concatenated_rhs.evaluate(None, y0)
        )

        np.testing.assert_array_equal(
            model.concatenated_algebraic.evaluate(None, y0),
            np.zeros_like(combined_submesh[0].nodes[:, np.newaxis]),
        )

        # mass matrix is identity upper left, zeros elsewhere
        mass = block_diag(
            (
                np.eye(np.size(combined_submesh[0].nodes)),
                np.zeros(
                    (
                        np.size(combined_submesh[0].nodes),
                        np.size(combined_submesh[0].nodes),
                    )
                ),
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
                    np.eye(np.size(combined_submesh[0].nodes)),
                    np.zeros(
                        (
                            np.size(combined_submesh[0].nodes),
                            np.size(combined_submesh[0].nodes),
                        )
                    ),
                ],
                [
                    -2 * np.eye(np.size(combined_submesh[0].nodes)),
                    np.eye(np.size(combined_submesh[0].nodes)),
                ],
            ]
        )
        np.testing.assert_array_equal(jacobian_actual, jacobian.toarray())

        # test jacobian by eqn gives same as jacobian of concatenated rhs & algebraic
        model.jacobian, _, _ = disc.create_jacobian(model)
        model_jacobian = model.jacobian.evaluate(0, y0)
        np.testing.assert_array_equal(model_jacobian.toarray(), jacobian.toarray())

        # test known_evals
        expr = pybamm.SparseStack(jac_rhs, jac_algebraic)
        jacobian, known_evals = expr.evaluate(0, y0, known_evals={})
        np.testing.assert_array_equal(jacobian_actual, jacobian.toarray())
        jacobian = expr.evaluate(0, y0, known_evals=known_evals)[0]
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

    def test_process_model_concatenation(self):
        # concatenation of variables as the key
        cn = pybamm.Variable("c", domain=["negative electrode"])
        cs = pybamm.Variable("c", domain=["separator"])
        cp = pybamm.Variable("c", domain=["positive electrode"])
        c = pybamm.Concatenation(cn, cs, cp)
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

        combined_submesh = mesh.combine_submeshes(
            "negative electrode", "separator", "positive electrode"
        )

        disc.process_model(model)
        y0 = model.concatenated_initial_conditions.evaluate()
        np.testing.assert_array_equal(
            y0, 3 * np.ones_like(combined_submesh[0].nodes[:, np.newaxis])
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
            y0, 3 * np.ones_like(submesh[0].nodes[:, np.newaxis])
        )

        # grad and div are identity operators here
        np.testing.assert_array_equal(
            y0, discretised_model.concatenated_rhs.evaluate(None, y0)
        )
        discretised_model.check_well_posedness()

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

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        # scalar
        broad = disc.process_symbol(pybamm.FullBroadcast(a, whole_cell, {}))
        np.testing.assert_array_equal(
            broad.evaluate(inputs={"a": 7}),
            7 * np.ones_like(combined_submesh[0].nodes[:, np.newaxis]),
        )
        self.assertEqual(broad.domain, whole_cell)

        broad_disc = disc.process_symbol(broad)
        self.assertIsInstance(broad_disc, pybamm.Multiplication)
        self.assertIsInstance(broad_disc.children[0], pybamm.InputParameter)
        self.assertIsInstance(broad_disc.children[1], pybamm.Vector)

        # process Broadcast variable
        disc.y_slices = {var.id: [slice(1)]}
        broad1 = pybamm.FullBroadcast(var, ["negative electrode"], None)
        broad1_disc = disc.process_symbol(broad1)
        self.assertIsInstance(broad1_disc, pybamm.Multiplication)
        self.assertIsInstance(broad1_disc.children[0], pybamm.StateVector)
        self.assertIsInstance(broad1_disc.children[1], pybamm.Vector)

        # broadcast to edges
        broad_to_edges = pybamm.FullBroadcastToEdges(a, ["negative electrode"], None)
        broad_to_edges_disc = disc.process_symbol(broad_to_edges)
        np.testing.assert_array_equal(
            broad_to_edges_disc.evaluate(inputs={"a": 7}),
            7 * np.ones_like(mesh["negative electrode"][0].edges[:, np.newaxis]),
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
            (mesh["separator"][0].npts * mesh["current collector"][0].npts, 1),
        )
        y_test = np.linspace(0, 1, mesh["current collector"][0].npts)
        np.testing.assert_array_equal(
            broad_disc.evaluate(y=y_test),
            np.outer(y_test, np.ones(mesh["separator"][0].npts)).reshape(-1, 1),
        )

        # test broadcast to edges
        broad_to_edges = pybamm.PrimaryBroadcastToEdges(var, "separator")
        broad_to_edges_disc = disc.process_symbol(broad_to_edges)
        self.assertIsInstance(broad_to_edges_disc, pybamm.MatrixMultiplication)
        self.assertIsInstance(broad_to_edges_disc.children[0], pybamm.Matrix)
        self.assertIsInstance(broad_to_edges_disc.children[1], pybamm.StateVector)
        self.assertEqual(
            broad_to_edges_disc.shape,
            ((mesh["separator"][0].npts + 1) * mesh["current collector"][0].npts, 1),
        )
        y_test = np.linspace(0, 1, mesh["current collector"][0].npts)
        np.testing.assert_array_equal(
            broad_to_edges_disc.evaluate(y=y_test),
            np.outer(y_test, np.ones(mesh["separator"][0].npts + 1)).reshape(-1, 1),
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
            (mesh["negative particle"][0].npts * mesh["negative electrode"][0].npts, 1),
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
            (
                mesh["negative particle"][0].npts
                * (mesh["negative electrode"][0].npts + 1),
                1,
            ),
        )

    def test_concatenation(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        c = pybamm.Symbol("c")

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

        eqn = pybamm.Concatenation(a, b)
        eqn_disc = disc.process_symbol(eqn)
        expected_vector = np.concatenate(
            [
                5 * np.ones_like(mesh["negative electrode"][0].nodes),
                4 * np.ones_like(mesh["separator"][0].nodes),
            ]
        )[:, np.newaxis]
        np.testing.assert_allclose(eqn_disc.evaluate(), expected_vector)

    def test_concatenation_2D(self):
        disc = get_1p1d_discretisation_for_testing(zpts=3)

        a = pybamm.Variable("a", domain=["negative electrode"])
        b = pybamm.Variable("b", domain=["separator"])
        c = pybamm.Variable("c", domain=["positive electrode"])
        conc = pybamm.Concatenation(a, b, c)
        disc.set_variable_slices([conc])
        self.assertEqual(
            disc.y_slices[a.id], [slice(0, 40), slice(100, 140), slice(200, 240)]
        )
        self.assertEqual(
            disc.y_slices[b.id], [slice(40, 65), slice(140, 165), slice(240, 265)]
        )
        self.assertEqual(
            disc.y_slices[c.id], [slice(65, 100), slice(165, 200), slice(265, 300)]
        )
        expr = disc.process_symbol(conc)
        self.assertIsInstance(expr, pybamm.DomainConcatenation)

        # Evaulate
        y = np.linspace(0, 1, 300)
        self.assertEqual(expr.children[0].evaluate(0, y).shape, (120, 1))
        self.assertEqual(expr.children[1].evaluate(0, y).shape, (75, 1))
        self.assertEqual(expr.children[2].evaluate(0, y).shape, (105, 1))
        np.testing.assert_equal(expr.evaluate(0, y), y[:, np.newaxis])

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
        model.variables = {c_n.name: pybamm.Concatenation(c_n, c_s)}
        disc.process_model(model)

        # check doesn't raise if broadcast
        model.variables = {
            c_n.name: pybamm.PrimaryBroadcast(
                pybamm.InputParameter("a"), ["negative electrode"]
            )
        }
        disc.process_model(model)

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

    def test_mass_matirx_inverse(self):
        # get mesh
        mesh = get_2p1d_mesh_for_testing(ypts=5, zpts=5)
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "current collector": pybamm.ScikitFiniteElement(),
        }
        # create model
        a = pybamm.Variable("a", domain="negative electrode")
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
        n = disc.mesh.combine_submeshes(*a.domain)[0].npts
        self.assertEqual(a_disc._expected_size, n)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
