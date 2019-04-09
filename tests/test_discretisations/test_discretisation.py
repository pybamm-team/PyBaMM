#
# Tests for the base model class
#
import pybamm

import numpy as np
import unittest
from tests import get_mesh_for_testing, get_discretisation_for_testing
from scipy.sparse import block_diag


class TestDiscretise(unittest.TestCase):
    def test_concatenate_init(self):
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

        disc._y_slices = {c.id: slice(0, 1), a.id: slice(2, 3), b.id: slice(3, 4)}
        result = disc._concatenate_init(initial_conditions)

        self.assertIsInstance(result, pybamm.NumpyConcatenation)
        self.assertEqual(result.children[0].evaluate(), 1)
        self.assertEqual(result.children[1].evaluate(), 2)
        self.assertEqual(result.children[2].evaluate(), 3)

        initial_conditions = {a: pybamm.Scalar(2), b: pybamm.Scalar(3)}
        with self.assertRaises(pybamm.ModelError):
            result = disc._concatenate_init(initial_conditions)

    def test_discretise_slicing(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        variables = [c]
        disc.set_variable_slices(variables)

        self.assertEqual(disc._y_slices, {c.id: slice(0, 100)})

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        c_true = combined_submesh[0].nodes ** 2
        y = c_true
        np.testing.assert_array_equal(y[disc._y_slices[c.id]], c_true)

        # Several variables
        d = pybamm.Variable("d", domain=whole_cell)
        jn = pybamm.Variable("jn", domain=["negative electrode"])
        variables = [c, d, jn]
        disc.set_variable_slices(variables)

        self.assertEqual(
            disc._y_slices,
            {c.id: slice(0, 100), d.id: slice(100, 200), jn.id: slice(200, 240)},
        )
        d_true = 4 * combined_submesh[0].nodes
        jn_true = mesh["negative electrode"][0].nodes ** 3
        y = np.concatenate([c_true, d_true, jn_true])
        np.testing.assert_array_equal(y[disc._y_slices[c.id]], c_true)
        np.testing.assert_array_equal(y[disc._y_slices[d.id]], d_true)
        np.testing.assert_array_equal(y[disc._y_slices[jn.id]], jn_true)

        # Variables with a concatenation
        js = pybamm.Variable("js", domain=["separator"])
        jp = pybamm.Variable("jp", domain=["positive electrode"])
        j = pybamm.Concatenation(jn, js, jp)
        variables = [c, d, j]
        disc.set_variable_slices(variables)
        self.assertEqual(
            disc._y_slices,
            {
                c.id: slice(0, 100),
                d.id: slice(100, 200),
                jn.id: slice(200, 240),
                js.id: slice(240, 265),
                jp.id: slice(265, 300),
            },
        )
        d_true = 4 * combined_submesh[0].nodes
        jn_true = mesh["negative electrode"][0].nodes ** 3
        y = np.concatenate([c_true, d_true, jn_true])
        np.testing.assert_array_equal(y[disc._y_slices[c.id]], c_true)
        np.testing.assert_array_equal(y[disc._y_slices[d.id]], d_true)
        np.testing.assert_array_equal(y[disc._y_slices[jn.id]], jn_true)

    def test_process_symbol_base(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # variable
        var = pybamm.Variable("var")
        disc._y_slices = {var.id: slice(53)}
        var_disc = disc.process_symbol(var)
        self.assertIsInstance(var_disc, pybamm.StateVector)
        self.assertEqual(var_disc._y_slice, disc._y_slices[var.id])
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

        un2 = abs(scal)
        un2_disc = disc.process_symbol(un2)
        self.assertIsInstance(un2_disc, pybamm.AbsoluteValue)
        self.assertIsInstance(un2_disc.children[0], pybamm.Scalar)

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

        disc._y_slices = {var1.id: slice(53), var2.id: slice(53, 59)}
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
            exp_disc.children[0].children[1].children[1].y_slice,
            disc._y_slices[var2.id],
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
            exp_disc.children[1].children[0].children[0].y_slice,
            disc._y_slices[var1.id],
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

    def test_core_NotImplementedErrors(self):
        # create spatial method
        spatial_method = pybamm.SpatialMethod(None)

        with self.assertRaises(NotImplementedError):
            spatial_method.gradient(None, None, {})
        with self.assertRaises(NotImplementedError):
            spatial_method.divergence(None, None, {})
        with self.assertRaises(NotImplementedError):
            spatial_method.integral(None, None)

    def test_process_dict(self):
        # one equation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        N = pybamm.grad(c)
        rhs = {c: pybamm.div(N)}
        initial_conditions = {c: pybamm.Scalar(3)}
        variables = {"c_squared": c ** 2}
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(0)}
        }

        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        y = combined_submesh[0].nodes ** 2
        disc._bcs = boundary_conditions

        disc.set_variable_slices(list(rhs.keys()))
        # rhs - grad and div are identity operators here
        processed_rhs = disc.process_dict(rhs)
        np.testing.assert_array_equal(y, processed_rhs[c].evaluate(None, y))
        # initial conditions
        y0 = disc.process_dict(initial_conditions)
        np.testing.assert_array_equal(
            y0[c].evaluate(0, None), 3 * np.ones_like(combined_submesh[0].nodes)
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
        )

        variables = list(rhs.keys())
        disc.set_variable_slices(variables)
        # rhs
        processed_rhs = disc.process_dict(rhs)
        np.testing.assert_array_equal(
            y[disc._y_slices[c.id]], processed_rhs[c].evaluate(None, y)
        )
        np.testing.assert_array_equal(
            y[disc._y_slices[T.id]], processed_rhs[T].evaluate(None, y)
        )
        # initial conditions
        y0 = disc.process_dict(initial_conditions)
        np.testing.assert_array_equal(
            y0[c].evaluate(0, None), 3 * np.ones_like(combined_submesh[0].nodes)
        )
        np.testing.assert_array_equal(
            y0[T].evaluate(0, None),
            5 * np.ones_like(mesh["negative electrode"][0].nodes),
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
            N: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(0)}
        }
        model.variables = {"c": c, "N": N}

        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        combined_submesh = mesh.combine_submeshes(*whole_cell)
        disc.process_model(model)

        y0 = model.concatenated_initial_conditions
        np.testing.assert_array_equal(y0, 3 * np.ones_like(combined_submesh[0].nodes))
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
            N: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(0)},
            q: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(0)},
            p: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(0)},
        }
        model.variables = {"ST": S * T}

        disc.process_model(model)
        y0 = model.concatenated_initial_conditions
        np.testing.assert_array_equal(
            y0,
            np.concatenate(
                [
                    2 * np.ones_like(combined_submesh[0].nodes),
                    5 * np.ones_like(mesh["negative electrode"][0].nodes),
                    8 * np.ones_like(mesh["negative electrode"][0].nodes),
                ]
            ),
        )
        # grad and div are identity operators here
        np.testing.assert_array_equal(y0, model.concatenated_rhs.evaluate(None, y0))
        c0, T0, S0 = np.split(
            y0,
            np.cumsum([combined_submesh[0].npts, mesh["negative electrode"][0].npts]),
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
        np.testing.assert_array_equal(
            np.eye(np.size(y0)), jacobian.toarray()
        )

        # test that not enough initial conditions raises an error
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N), T: pybamm.div(q), S: pybamm.div(p)}
        model.initial_conditions = {T: pybamm.Scalar(5), S: pybamm.Scalar(8)}
        model.boundary_conditions = {}
        model.variables = {"ST": S * T}
        with self.assertRaises(pybamm.ModelError):
            disc.process_model(model)

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
            N: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(0)}
        }
        model.variables = {"c": c, "N": N, "d": d}

        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        disc.process_model(model)
        combined_submesh = mesh.combine_submeshes(*whole_cell)

        y0 = model.concatenated_initial_conditions
        np.testing.assert_array_equal(
            y0,
            np.concatenate(
                [
                    3 * np.ones_like(combined_submesh[0].nodes),
                    6 * np.ones_like(combined_submesh[0].nodes),
                ]
            ),
        )

        # grad and div are identity operators here
        np.testing.assert_array_equal(
            y0[: combined_submesh[0].npts], model.concatenated_rhs.evaluate(None, y0)
        )

        np.testing.assert_array_equal(
            model.concatenated_algebraic.evaluate(None, y0),
            np.zeros_like(combined_submesh[0].nodes),
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
        disc.create_jacobian(model)
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

        model.boundary_conditions = {N: {"left": 0, "right": 0}}
        model.check_well_posedness()

        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        combined_submesh = mesh.combine_submeshes(
            "negative electrode", "separator", "positive electrode"
        )

        disc.process_model(model)
        y0 = model.concatenated_initial_conditions
        np.testing.assert_array_equal(y0, 3 * np.ones_like(combined_submesh[0].nodes))

        # grad and div are identity operators here
        np.testing.assert_array_equal(y0, model.concatenated_rhs.evaluate(None, y0))
        model.check_well_posedness()

    def test_broadcast(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]

        a = pybamm.Scalar(7)
        var = pybamm.Variable("var")

        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        # scalar
        broad = disc._spatial_methods[whole_cell[0]].broadcast(a, whole_cell)
        np.testing.assert_array_equal(
            broad.evaluate(), 7 * np.ones_like(combined_submesh[0].nodes)
        )
        self.assertEqual(broad.domain, whole_cell)

        broad_disc = disc.process_symbol(broad)
        self.assertIsInstance(broad_disc, pybamm.NumpyBroadcast)

        # process Broadcast variable
        disc._y_slices = {var.id: slice(53)}
        broad1 = pybamm.Broadcast(var, ["negative electrode"])
        broad1_disc = disc.process_symbol(broad1)
        self.assertIsInstance(broad1_disc, pybamm.NumpyBroadcast)
        self.assertIsInstance(broad1_disc.children[0], pybamm.StateVector)

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
        a = pybamm.Broadcast(5, ["negative electrode"])
        b = pybamm.Broadcast(4, ["positive electrode"])

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
                4 * np.ones_like(mesh["positive electrode"][0].nodes),
            ]
        )
        np.testing.assert_allclose(eqn_disc.evaluate(), expected_vector)

    def test_discretise_spatial_variable(self):
        # create discretisation
        disc = get_discretisation_for_testing()

        # space
        x1 = pybamm.SpatialVariable("x", ["negative electrode"])
        x1_disc = disc.process_symbol(x1)
        self.assertIsInstance(x1_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            x1_disc.evaluate(), disc.mesh["negative electrode"][0].nodes
        )

        x2 = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        x2_disc = disc.process_symbol(x2)
        self.assertIsInstance(x2_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            x2_disc.evaluate(),
            disc.mesh.combine_submeshes("negative electrode", "separator")[0].nodes,
        )

        r = 3 * pybamm.SpatialVariable("r", ["negative particle"])
        r_disc = disc.process_symbol(r)
        self.assertIsInstance(r_disc.children[1], pybamm.Vector)
        np.testing.assert_array_equal(
            r_disc.evaluate(), 3 * disc.mesh["negative particle"][0].nodes
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
