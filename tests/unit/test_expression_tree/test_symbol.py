#
# Test for the Symbol class
#
import pybamm

import unittest
import numpy as np
import os


class TestSymbol(unittest.TestCase):
    def test_symbol_init(self):
        sym = pybamm.Symbol("a symbol")
        self.assertEqual(sym.name, "a symbol")
        self.assertEqual(str(sym), "a symbol")

    def test_cached_children(self):
        symc1 = pybamm.Symbol("child1")
        symc2 = pybamm.Symbol("child2")
        symc3 = pybamm.Symbol("child3")
        symp = pybamm.Symbol("parent", children=[symc1, symc2])

        # test tuples of children for equality based on their name
        def check_are_equal(children1, children2):
            self.assertEqual(len(children1), len(children2))
            for i in range(len(children1)):
                self.assertEqual(children1[i].name, children2[i].name)

        check_are_equal(symp.children, super(pybamm.Symbol, symp).children)
        check_are_equal(symp.children, (symc1, symc2))

        # update children, since we cache the children they will be unchanged
        symc3.parent = symp
        check_are_equal(symp.children, (symc1, symc2))

        # check that the *actual* children are updated
        check_are_equal(super(pybamm.Symbol, symp).children, (symc1, symc2, symc3))

    def test_symbol_domains(self):
        a = pybamm.Symbol("a", domain="test")
        self.assertEqual(a.domain, ["test"])
        a = pybamm.Symbol("a", domain=["t", "e", "s"])
        self.assertEqual(a.domain, ["t", "e", "s"])
        with self.assertRaises(TypeError):
            a = pybamm.Symbol("a", domain=1)

    def test_symbol_methods(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")

        # unary
        self.assertIsInstance(-a, pybamm.Negate)
        self.assertIsInstance(abs(a), pybamm.AbsoluteValue)

        # binary - two symbols
        self.assertIsInstance(a + b, pybamm.Addition)
        self.assertIsInstance(a - b, pybamm.Subtraction)
        self.assertIsInstance(a * b, pybamm.Multiplication)
        self.assertIsInstance(a @ b, pybamm.MatrixMultiplication)
        self.assertIsInstance(a / b, pybamm.Division)
        self.assertIsInstance(a ** b, pybamm.Power)

        # binary - symbol and number
        self.assertIsInstance(a + 2, pybamm.Addition)
        self.assertIsInstance(a - 2, pybamm.Subtraction)
        self.assertIsInstance(a * 2, pybamm.Multiplication)
        self.assertIsInstance(a @ 2, pybamm.MatrixMultiplication)
        self.assertIsInstance(a / 2, pybamm.Division)
        self.assertIsInstance(a ** 2, pybamm.Power)

        # binary - number and symbol
        self.assertIsInstance(3 + b, pybamm.Addition)
        self.assertEqual((3 + b).children[1].id, b.id)
        self.assertIsInstance(3 - b, pybamm.Subtraction)
        self.assertEqual((3 - b).children[1].id, b.id)
        self.assertIsInstance(3 * b, pybamm.Multiplication)
        self.assertEqual((3 * b).children[1].id, b.id)
        self.assertIsInstance(3 @ b, pybamm.MatrixMultiplication)
        self.assertEqual((3 @ b).children[1].id, b.id)
        self.assertIsInstance(3 / b, pybamm.Division)
        self.assertEqual((3 / b).children[1].id, b.id)
        self.assertIsInstance(3 ** b, pybamm.Power)
        self.assertEqual((3 ** b).children[1].id, b.id)

        # error raising
        with self.assertRaises(NotImplementedError):
            a + "two"
        with self.assertRaises(NotImplementedError):
            a - "two"
        with self.assertRaises(NotImplementedError):
            a * "two"
        with self.assertRaises(NotImplementedError):
            a @ "two"
        with self.assertRaises(NotImplementedError):
            a / "two"
        with self.assertRaises(NotImplementedError):
            a ** "two"
        with self.assertRaises(NotImplementedError):
            "two" + a
        with self.assertRaises(NotImplementedError):
            "two" - a
        with self.assertRaises(NotImplementedError):
            "two" * a
        with self.assertRaises(NotImplementedError):
            "two" @ a
        with self.assertRaises(NotImplementedError):
            "two" / a
        with self.assertRaises(NotImplementedError):
            "two" ** a

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

    def test_symbol_is_constant(self):
        a = pybamm.Variable("a")
        self.assertFalse(a.is_constant())

        a = pybamm.Parameter("a")
        self.assertTrue(a.is_constant())

        a = pybamm.Scalar(1) * pybamm.Variable("a")
        self.assertFalse(a.is_constant())

        a = pybamm.Scalar(1) * pybamm.Parameter("a")
        self.assertTrue(a.is_constant())

        a = pybamm.Scalar(1) * pybamm.StateVector(slice(10))
        self.assertFalse(a.is_constant())

        a = pybamm.Scalar(1) * pybamm.Vector(np.zeros(10))
        self.assertTrue(a.is_constant())

    def test_symbol_evaluates_to_number(self):
        a = pybamm.Scalar(3)
        self.assertTrue(a.evaluates_to_number())

        a = pybamm.Parameter("a")
        self.assertFalse(a.evaluates_to_number())

        a = pybamm.Scalar(3) * pybamm.Scalar(2)
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

        # Time variable returns true
        a = 3 * pybamm.t + 2
        self.assertTrue(a.evaluates_to_number())

    def test_symbol_repr(self):
        """
        test that __repr___ returns the string
        `__class__(id, name, parent expression)`
        """
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        c = pybamm.Symbol("c", domain=["test"])
        d = pybamm.Symbol("d", domain=["test"])
        hex_regex = r"\-?0x[0-9,a-f]+"
        self.assertRegex(
            a.__repr__(),
            r"Symbol\(" + hex_regex + r", a, children\=\[\], domain\=\[\]\)",
        )
        self.assertRegex(
            b.__repr__(),
            r"Symbol\(" + hex_regex + r", b, children\=\[\], domain\=\[\]\)",
        )
        self.assertRegex(
            c.__repr__(),
            r"Symbol\(" + hex_regex + r", c, children\=\[\], domain\=\['test'\]\)",
        )
        self.assertRegex(
            d.__repr__(),
            r"Symbol\(" + hex_regex + r", d, children\=\[\], domain\=\['test'\]\)",
        )
        self.assertRegex(
            (a + b).__repr__(),
            r"Addition\(" + hex_regex + r", \+, children\=\['a', 'b'\], domain=\[\]\)",
        )
        self.assertRegex(
            (c * d).__repr__(),
            r"Multiplication\("
            + hex_regex
            + r", \*, children\=\['c', 'd'\], domain=\['test'\]\)",
        )
        self.assertRegex(
            pybamm.grad(a).__repr__(),
            r"Gradient\(" + hex_regex + ", grad, children\=\['a'\], domain=\[\]\)",
        )
        self.assertRegex(
            pybamm.grad(c).__repr__(),
            r"Gradient\("
            + hex_regex
            + ", grad, children\=\['c'\], domain=\['test'\]\)",
        )

    def test_symbol_visualise(self):
        param = pybamm.standard_parameters_lithium_ion

        c_e = pybamm.standard_variables.c_e
        variables = {"Electrolyte concentration": c_e}
        onen = pybamm.Broadcast(1, ["negative electrode"])
        onep = pybamm.Broadcast(1, ["positive electrode"])
        reactions = {
            "main": {"neg": {"s_plus": 1, "aj": onen}, "pos": {"s_plus": 1, "aj": onep}}
        }
        model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        model.set_differential_system(variables, reactions)
        rhs = model.rhs[c_e]
        rhs.visualise("StefanMaxwell_test.png")
        self.assertTrue(os.path.exists("StefanMaxwell_test.png"))
        with self.assertRaises(ValueError):
            rhs.visualise("StefanMaxwell_test")

    def test_has_spatial_derivatives(self):
        var = pybamm.Variable("var")
        grad_eqn = pybamm.grad(var)
        div_eqn = pybamm.div(var)
        grad_div_eqn = pybamm.div(grad_eqn)
        algebraic_eqn = 2 * var + 3
        self.assertTrue(grad_eqn.has_symbol_of_class(pybamm.Gradient))
        self.assertFalse(grad_eqn.has_symbol_of_class(pybamm.Divergence))
        self.assertFalse(div_eqn.has_symbol_of_class(pybamm.Gradient))
        self.assertTrue(div_eqn.has_symbol_of_class(pybamm.Divergence))
        self.assertTrue(grad_div_eqn.has_symbol_of_class(pybamm.Gradient))
        self.assertTrue(grad_div_eqn.has_symbol_of_class(pybamm.Divergence))
        self.assertFalse(algebraic_eqn.has_symbol_of_class(pybamm.Gradient))
        self.assertFalse(algebraic_eqn.has_symbol_of_class(pybamm.Divergence))

    def test_orphans(self):
        a = pybamm.Scalar(1)
        b = pybamm.Scalar(2)
        sum = a + b

        a_orp, b_orp = sum.orphans
        self.assertIsNone(a_orp.parent)
        self.assertIsNone(b_orp.parent)
        self.assertEqual(a.id, a_orp.id)
        self.assertEqual(b.id, b_orp.id)

    def test_shape(self):
        scal = pybamm.Scalar(1)
        self.assertEqual(scal.shape_for_testing, ())
        self.assertEqual(scal.size, 1)

        state = pybamm.StateVector(slice(10))
        self.assertEqual(state.shape_for_testing, (10, 1))
        self.assertEqual(state.size, 10)
        state = pybamm.StateVector(slice(10, 25))
        self.assertEqual(state.shape_for_testing, (15, 1))

    def test_shape_for_testing(self):
        scal = pybamm.Scalar(1)
        self.assertEqual(scal.shape_for_testing, scal.shape)

        state = pybamm.StateVector(slice(10, 25))
        self.assertEqual(state.shape_for_testing, state.shape)

        param = pybamm.Parameter("a")
        self.assertEqual(param.shape_for_testing, ())

        func = pybamm.FunctionParameter("func", state)
        self.assertEqual(func.shape_for_testing, state.shape_for_testing)

        concat = pybamm.Concatenation()
        self.assertEqual(concat.shape_for_testing, (0,))
        concat = pybamm.Concatenation(state, state)
        self.assertEqual(concat.shape_for_testing, (30, 1))

        var = pybamm.Variable("var", domain="negative electrode")
        broadcast = pybamm.Broadcast(0, domain="negative electrode")
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


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
