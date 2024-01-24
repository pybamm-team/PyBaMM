#
# Test for the Symbol class
#
from tests import TestCase
import os
import unittest
import unittest.mock as mock
from tempfile import TemporaryDirectory

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

import pybamm
from pybamm.expression_tree.binary_operators import _Heaviside
from pybamm.util import have_optional_dependency


class TestSymbol(TestCase):
    def test_symbol_init(self):
        sym = pybamm.Symbol("a symbol")
        self.assertEqual(sym.name, "a symbol")
        self.assertEqual(str(sym), "a symbol")

    def test_children(self):
        symc1 = pybamm.Symbol("child1")
        symc2 = pybamm.Symbol("child2")
        symp = pybamm.Symbol("parent", children=[symc1, symc2])

        # test tuples of children for equality based on their name
        def check_are_equal(children1, children2):
            self.assertEqual(len(children1), len(children2))
            for i in range(len(children1)):
                self.assertEqual(children1[i].name, children2[i].name)

        check_are_equal(symp.children, (symc1, symc2))

    def test_symbol_domains(self):
        a = pybamm.Symbol("a", domain="test")
        self.assertEqual(a.domain, ["test"])
        # test for updating domain with same as existing domain
        a.domains = {"primary": ["test"]}
        self.assertEqual(a.domains["primary"], ["test"])
        a = pybamm.Symbol("a", domain=["t", "e", "s"])
        self.assertEqual(a.domain, ["t", "e", "s"])
        with self.assertRaises(TypeError):
            a = pybamm.Symbol("a", domain=1)
        with self.assertRaisesRegex(
            pybamm.DomainError,
            "Domain levels must be filled in order",
        ):
            b = pybamm.Symbol("b", auxiliary_domains={"secondary": ["test sec"]})
        b = pybamm.Symbol(
            "b", domain="test", auxiliary_domains={"secondary": ["test sec"]}
        )

        with self.assertRaisesRegex(pybamm.DomainError, "keys must be one of"):
            b.domains = {"test": "test"}
        with self.assertRaisesRegex(ValueError, "Only one of 'domain' or 'domains'"):
            pybamm.Symbol("b", domain="test", domains={"primary": "test"})
        with self.assertRaisesRegex(
            ValueError, "Only one of 'auxiliary_domains' or 'domains'"
        ):
            pybamm.Symbol(
                "b",
                auxiliary_domains={"secondary": "other test"},
                domains={"test": "test"},
            )
        with self.assertRaisesRegex(NotImplementedError, "Cannot set domain directly"):
            b.domain = "test"

    def test_symbol_auxiliary_domains(self):
        a = pybamm.Symbol(
            "a",
            domain="test",
            auxiliary_domains={
                "secondary": "sec",
                "tertiary": "tert",
                "quaternary": "quat",
            },
        )
        self.assertEqual(a.domain, ["test"])
        self.assertEqual(a.secondary_domain, ["sec"])
        self.assertEqual(a.tertiary_domain, ["tert"])
        self.assertEqual(a.tertiary_domain, ["tert"])
        self.assertEqual(a.quaternary_domain, ["quat"])
        self.assertEqual(
            a.domains,
            {
                "primary": ["test"],
                "secondary": ["sec"],
                "tertiary": ["tert"],
                "quaternary": ["quat"],
            },
        )

        a = pybamm.Symbol("a", domain=["t", "e", "s"])
        self.assertEqual(a.domain, ["t", "e", "s"])
        with self.assertRaises(TypeError):
            a = pybamm.Symbol("a", domain=1)
        b = pybamm.Symbol("b", domain="test sec")
        with self.assertRaisesRegex(
            pybamm.DomainError, "All domains must be different"
        ):
            b.domains = {"primary": "test", "secondary": "test"}
        with self.assertRaisesRegex(
            pybamm.DomainError, "All domains must be different"
        ):
            b = pybamm.Symbol(
                "b",
                domain="test",
                auxiliary_domains={"secondary": ["test sec"], "tertiary": ["test sec"]},
            )

        with self.assertRaisesRegex(NotImplementedError, "auxiliary_domains"):
            a.auxiliary_domains

    def test_symbol_methods(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")

        # unary
        self.assertIsInstance(-a, pybamm.Negate)
        self.assertIsInstance(abs(a), pybamm.AbsoluteValue)
        # special cases
        self.assertEqual(-(-a), a)
        self.assertEqual(-(a - b), b - a)
        self.assertEqual(abs(abs(a)), abs(a))

        # binary - two symbols
        self.assertIsInstance(a + b, pybamm.Addition)
        self.assertIsInstance(a - b, pybamm.Subtraction)
        self.assertIsInstance(a * b, pybamm.Multiplication)
        self.assertIsInstance(a @ b, pybamm.MatrixMultiplication)
        self.assertIsInstance(a / b, pybamm.Division)
        self.assertIsInstance(a**b, pybamm.Power)
        self.assertIsInstance(a < b, _Heaviside)
        self.assertIsInstance(a <= b, _Heaviside)
        self.assertIsInstance(a > b, _Heaviside)
        self.assertIsInstance(a >= b, _Heaviside)
        self.assertIsInstance(a % b, pybamm.Modulo)

        # binary - symbol and number
        self.assertIsInstance(a + 2, pybamm.Addition)
        self.assertIsInstance(2 - a, pybamm.Subtraction)
        self.assertIsInstance(a * 2, pybamm.Multiplication)
        self.assertIsInstance(a @ 2, pybamm.MatrixMultiplication)
        self.assertIsInstance(2 / a, pybamm.Division)
        self.assertIsInstance(a**2, pybamm.Power)

        # binary - number and symbol
        self.assertIsInstance(3 + b, pybamm.Addition)
        self.assertEqual((3 + b).children[1], b)
        self.assertIsInstance(3 - b, pybamm.Subtraction)
        self.assertEqual((3 - b).children[1], b)
        self.assertIsInstance(3 * b, pybamm.Multiplication)
        self.assertEqual((3 * b).children[1], b)
        self.assertIsInstance(3 @ b, pybamm.MatrixMultiplication)
        self.assertEqual((3 @ b).children[1], b)
        self.assertIsInstance(3 / b, pybamm.Division)
        self.assertEqual((3 / b).children[1], b)
        self.assertIsInstance(3**b, pybamm.Power)
        self.assertEqual((3**b).children[1], b)

        # error raising
        with self.assertRaisesRegex(
            NotImplementedError, "BinaryOperator not implemented for symbols of type"
        ):
            a + "two"

    def test_symbol_create_copy(self):
        a = pybamm.Symbol("a")
        with self.assertRaisesRegex(NotImplementedError, "method self.new_copy()"):
            a.create_copy()

    def test_sigmoid(self):
        # Test that smooth heaviside is used when the setting is changed
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")

        pybamm.settings.heaviside_smoothing = 10

        self.assertEqual(str(a < b), str(pybamm.sigmoid(a, b, 10)))
        self.assertEqual(str(a <= b), str(pybamm.sigmoid(a, b, 10)))
        self.assertEqual(str(a > b), str(pybamm.sigmoid(b, a, 10)))
        self.assertEqual(str(a >= b), str(pybamm.sigmoid(b, a, 10)))

        # But exact heavisides should still be used if both variables are constant
        a = pybamm.Scalar(1)
        b = pybamm.Scalar(2)
        self.assertEqual(str(a < b), str(pybamm.Scalar(1)))
        self.assertEqual(str(a <= b), str(pybamm.Scalar(1)))
        self.assertEqual(str(a > b), str(pybamm.Scalar(0)))
        self.assertEqual(str(a >= b), str(pybamm.Scalar(0)))

        # Change setting back for other tests
        pybamm.settings.heaviside_smoothing = "exact"

    def test_smooth_absolute_value(self):
        # Test that smooth absolute value is used when the setting is changed
        a = pybamm.Symbol("a")
        pybamm.settings.abs_smoothing = 10
        self.assertEqual(str(abs(a)), str(pybamm.smooth_absolute_value(a, 10)))

        # But exact absolute value should still be used for constants
        a = pybamm.Scalar(-5)
        self.assertEqual(str(abs(a)), str(pybamm.Scalar(5)))

        # Change setting back for other tests
        pybamm.settings.abs_smoothing = "exact"

    def test_multiple_symbols(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        c = pybamm.Symbol("c")
        exp = a * c * (a * b * c + a - c * a)
        expected_preorder = [
            "*",
            "*",
            "a",
            "c",
            "-",
            "+",
            "*",
            "*",
            "a",
            "b",
            "c",
            "a",
            "*",
            "c",
            "a",
        ]
        for node, expect in zip(exp.pre_order(), expected_preorder):
            self.assertEqual(node.name, expect)

    def test_symbol_diff(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        self.assertIsInstance(a.diff(a), pybamm.Scalar)
        self.assertEqual(a.diff(a).evaluate(), 1)
        self.assertIsInstance(a.diff(b), pybamm.Scalar)
        self.assertEqual(a.diff(b).evaluate(), 0)

    def test_symbol_evaluation(self):
        a = pybamm.Symbol("a")
        with self.assertRaises(NotImplementedError):
            a.evaluate()

    def test_evaluate_ignoring_errors(self):
        self.assertIsNone(pybamm.t.evaluate_ignoring_errors(t=None))
        self.assertEqual(pybamm.t.evaluate_ignoring_errors(t=0), 0)
        self.assertIsNone(pybamm.Parameter("a").evaluate_ignoring_errors())
        self.assertIsNone(pybamm.StateVector(slice(0, 1)).evaluate_ignoring_errors())
        self.assertIsNone(pybamm.StateVectorDot(slice(0, 1)).evaluate_ignoring_errors())

        np.testing.assert_array_equal(
            pybamm.InputParameter("a").evaluate_ignoring_errors(), np.nan
        )

    def test_symbol_is_constant(self):
        a = pybamm.Variable("a")
        self.assertFalse(a.is_constant())

        a = pybamm.Parameter("a")
        self.assertFalse(a.is_constant())

        a = pybamm.Scalar(1) * pybamm.Variable("a")
        self.assertFalse(a.is_constant())

        a = pybamm.Scalar(1) * pybamm.StateVector(slice(10))
        self.assertFalse(a.is_constant())

        a = pybamm.Scalar(1) * pybamm.Vector(np.zeros(10))
        self.assertTrue(a.is_constant())

    def test_symbol_evaluates_to_number(self):
        a = pybamm.Scalar(3)
        self.assertTrue(a.evaluates_to_number())

        a = pybamm.Parameter("a")
        self.assertTrue(a.evaluates_to_number())

        a = pybamm.Scalar(3) * pybamm.Time()
        self.assertTrue(a.evaluates_to_number())
        # highlight difference between this function and isinstance(a, Scalar)
        self.assertNotIsInstance(a, pybamm.Scalar)

        a = pybamm.Variable("a")
        self.assertFalse(a.evaluates_to_number())

        a = pybamm.Scalar(3) - 2
        self.assertTrue(a.evaluates_to_number())

        a = pybamm.Vector(np.ones(5))
        self.assertFalse(a.evaluates_to_number())

        a = pybamm.Matrix(np.ones((4, 6)))
        self.assertFalse(a.evaluates_to_number())

        a = pybamm.StateVector(slice(0, 10))
        self.assertFalse(a.evaluates_to_number())

        # Time variable returns false
        a = 3 * pybamm.t + 2
        self.assertTrue(a.evaluates_to_number())

    def test_symbol_evaluates_to_constant_number(self):
        a = pybamm.Scalar(3)
        self.assertTrue(a.evaluates_to_constant_number())

        a = pybamm.Parameter("a")
        self.assertFalse(a.evaluates_to_constant_number())

        a = pybamm.Variable("a")
        self.assertFalse(a.evaluates_to_constant_number())

        a = pybamm.Scalar(3) - 2
        self.assertTrue(a.evaluates_to_constant_number())

        a = pybamm.Vector(np.ones(5))
        self.assertFalse(a.evaluates_to_constant_number())

        a = pybamm.Matrix(np.ones((4, 6)))
        self.assertFalse(a.evaluates_to_constant_number())

        a = pybamm.StateVector(slice(0, 10))
        self.assertFalse(a.evaluates_to_constant_number())

        # Time variable returns true
        a = 3 * pybamm.t + 2
        self.assertFalse(a.evaluates_to_constant_number())

    def test_simplify_if_constant(self):
        m = pybamm.Matrix(np.zeros((10, 10)))
        m_simp = pybamm.simplify_if_constant(m)
        self.assertIsInstance(m_simp, pybamm.Matrix)
        self.assertIsInstance(m_simp.entries, csr_matrix)

    def test_symbol_repr(self):
        """
        test that __repr___ returns the string
        `__class__(id, name, children, domain, auxiliary_domains)`
        """
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        c = pybamm.Symbol("c", domain=["test"])
        d = pybamm.Symbol(
            "d", domain=["test"], auxiliary_domains={"secondary": "other test"}
        )
        hex_regex = r"\-?0x[0-9,a-f]+"
        self.assertRegex(
            a.__repr__(),
            r"Symbol\(" + hex_regex + r", a, children\=\[\], domains\=\{\}\)",
        )
        self.assertRegex(
            b.__repr__(),
            r"Symbol\(" + hex_regex + r", b, children\=\[\], domains\=\{\}\)",
        )
        self.assertRegex(
            c.__repr__(),
            r"Symbol\("
            + hex_regex
            + r", c, children\=\[\], domains\=\{'primary': \['test'\]\}\)",
        )
        self.assertRegex(
            d.__repr__(),
            r"Symbol\("
            + hex_regex
            + r", d, children\=\[\], domains\=\{'primary': \['test'\], "
            + r"'secondary': \['other test'\]\}\)",
        )
        self.assertRegex(
            (a + b).__repr__(),
            r"Addition\(" + hex_regex + r", \+, children\=\['a', 'b'\], domains=\{\}",
        )
        self.assertRegex(
            (a * d).__repr__(),
            r"Multiplication\("
            + hex_regex
            + r", \*, children\=\['a', 'd'\], domains\=\{'primary': \['test'\], "
            + r"'secondary': \['other test'\]\}\)",
        )
        self.assertRegex(
            pybamm.grad(c).__repr__(),
            r"Gradient\("
            + hex_regex
            + r", grad, children\=\['c'\], domains\=\{'primary': \['test'\]}",
        )

    def test_symbol_visualise(self):
        with TemporaryDirectory() as dir_name:
            test_stub = os.path.join(dir_name, "test_visualize")
            test_name = f"{test_stub}.png"
            c = pybamm.Variable("c", "negative electrode")
            d = pybamm.Variable("d", "negative electrode")
            sym = pybamm.div(c * pybamm.grad(c)) + (c / d + c - d) ** 5
            sym.visualise(test_name)
            self.assertTrue(os.path.exists(test_name))
            with self.assertRaises(ValueError):
                sym.visualise(test_stub)

    def test_has_spatial_derivatives(self):
        var = pybamm.Variable("var", domain="test")
        grad_eqn = pybamm.grad(var)
        div_eqn = pybamm.div(pybamm.standard_spatial_vars.x_edge)
        grad_div_eqn = pybamm.div(grad_eqn)
        algebraic_eqn = 2 * var + 3
        self.assertTrue(grad_eqn.has_symbol_of_classes(pybamm.Gradient))
        self.assertFalse(grad_eqn.has_symbol_of_classes(pybamm.Divergence))
        self.assertFalse(div_eqn.has_symbol_of_classes(pybamm.Gradient))
        self.assertTrue(div_eqn.has_symbol_of_classes(pybamm.Divergence))
        self.assertTrue(grad_div_eqn.has_symbol_of_classes(pybamm.Gradient))
        self.assertTrue(grad_div_eqn.has_symbol_of_classes(pybamm.Divergence))
        self.assertFalse(algebraic_eqn.has_symbol_of_classes(pybamm.Gradient))
        self.assertFalse(algebraic_eqn.has_symbol_of_classes(pybamm.Divergence))

    def test_orphans(self):
        a = pybamm.Scalar(1)
        b = pybamm.Parameter("b")
        summ = a + b

        a_orp, b_orp = summ.orphans
        self.assertEqual(a, a_orp)
        self.assertEqual(b, b_orp)

    def test_shape(self):
        scal = pybamm.Scalar(1)
        self.assertEqual(scal.shape, ())
        self.assertEqual(scal.size, 1)

        state = pybamm.StateVector(slice(10))
        self.assertEqual(state.shape, (10, 1))
        self.assertEqual(state.size, 10)
        state = pybamm.StateVector(slice(10, 25))
        self.assertEqual(state.shape, (15, 1))

        # test with big object
        state = 2 * pybamm.StateVector(slice(100000))
        self.assertEqual(state.shape, (100000, 1))

    def test_shape_and_size_for_testing(self):
        scal = pybamm.Scalar(1)
        self.assertEqual(scal.shape_for_testing, scal.shape)
        self.assertEqual(scal.size_for_testing, scal.size)

        state = pybamm.StateVector(slice(10, 25), domain="test")
        state2 = pybamm.StateVector(slice(10, 25), domain="test 2")
        self.assertEqual(state.shape_for_testing, state.shape)

        param = pybamm.Parameter("a")
        self.assertEqual(param.shape_for_testing, ())

        func = pybamm.FunctionParameter("func", {"state": state})
        self.assertEqual(func.shape_for_testing, state.shape_for_testing)

        concat = pybamm.concatenation(state, state2)
        self.assertEqual(concat.shape_for_testing, (30, 1))
        self.assertEqual(concat.size_for_testing, 30)

        var = pybamm.Variable("var", domain="negative electrode")
        broadcast = pybamm.PrimaryBroadcast(0, "negative electrode")
        self.assertEqual(var.shape_for_testing, broadcast.shape_for_testing)
        self.assertEqual(
            (var + broadcast).shape_for_testing, broadcast.shape_for_testing
        )

        var = pybamm.Variable("var", domain=["random domain", "other domain"])
        broadcast = pybamm.PrimaryBroadcast(0, ["random domain", "other domain"])
        self.assertEqual(var.shape_for_testing, broadcast.shape_for_testing)
        self.assertEqual(
            (var + broadcast).shape_for_testing, broadcast.shape_for_testing
        )

        sym = pybamm.Symbol("sym")
        with self.assertRaises(NotImplementedError):
            sym.shape_for_testing

    def test_test_shape(self):
        # right shape, passes
        y1 = pybamm.StateVector(slice(0, 10))
        y1.test_shape()
        # bad shape, fails
        y2 = pybamm.StateVector(slice(0, 5))
        with self.assertRaises(pybamm.ShapeError):
            (y1 + y2).test_shape()

    def test_to_equation(self):
        sympy = have_optional_dependency("sympy")
        self.assertEqual(pybamm.Symbol("test").to_equation(), sympy.Symbol("test"))

    def test_numpy_array_ufunc(self):
        x = pybamm.Symbol("x")
        self.assertEqual(np.exp(x), pybamm.exp(x))

    def test_to_from_json(self):
        symc1 = pybamm.Symbol("child1", domain=["domain_1"])
        symc2 = pybamm.Symbol("child2", domain=["domain_2"])
        symp = pybamm.Symbol("parent", domain=["domain_3"], children=[symc1, symc2])

        json_dict = {
            "name": "parent",
            "id": mock.ANY,
            "domains": {
                "primary": ["domain_3"],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
        }

        self.assertEqual(symp.to_json(), json_dict)

        json_dict["children"] = [symc1, symc2]

        self.assertEqual(pybamm.Symbol._from_json(json_dict), symp)


class TestIsZero(TestCase):
    def test_is_scalar_zero(self):
        a = pybamm.Scalar(0)
        b = pybamm.Scalar(2)
        self.assertTrue(pybamm.is_scalar_zero(a))
        self.assertFalse(pybamm.is_scalar_zero(b))

    def test_is_matrix_zero(self):
        a = pybamm.Matrix(coo_matrix(np.zeros((10, 10))))
        b = pybamm.Matrix(coo_matrix(np.ones((10, 10))))
        c = pybamm.Matrix(coo_matrix(([1], ([0], [0])), shape=(5, 5)))
        self.assertTrue(pybamm.is_matrix_zero(a))
        self.assertFalse(pybamm.is_matrix_zero(b))
        self.assertFalse(pybamm.is_matrix_zero(c))

        a = pybamm.Matrix(np.zeros((10, 10)))
        b = pybamm.Matrix(np.ones((10, 10)))
        c = pybamm.Matrix([1, 0, 0])
        self.assertTrue(pybamm.is_matrix_zero(a))
        self.assertFalse(pybamm.is_matrix_zero(b))
        self.assertFalse(pybamm.is_matrix_zero(c))

    def test_bool(self):
        a = pybamm.Symbol("a")
        with self.assertRaisesRegex(NotImplementedError, "Boolean"):
            bool(a)
        # if statement calls Boolean
        with self.assertRaisesRegex(NotImplementedError, "Boolean"):
            if a > 1:
                print("a is greater than 1")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
