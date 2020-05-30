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
        self.assertEqual(a.domains["primary"], ["test"])
        a = pybamm.Symbol("a", domain=["t", "e", "s"])
        self.assertEqual(a.domain, ["t", "e", "s"])
        with self.assertRaises(TypeError):
            a = pybamm.Symbol("a", domain=1)
        with self.assertRaisesRegex(
            pybamm.DomainError,
            "Domain cannot be empty if auxiliary domains are not empty",
        ):
            b = pybamm.Symbol("b", auxiliary_domains={"sec": ["test sec"]})
        b = pybamm.Symbol("b", domain="test", auxiliary_domains={"sec": ["test sec"]})
        with self.assertRaisesRegex(
            pybamm.DomainError, "Domain cannot be the same as an auxiliary domain"
        ):
            b.domain = "test sec"

    def test_symbol_auxiliary_domains(self):
        a = pybamm.Symbol(
            "a",
            domain="test",
            auxiliary_domains={"secondary": "sec", "tertiary": "tert"},
        )
        self.assertEqual(a.domain, ["test"])
        self.assertEqual(
            a.auxiliary_domains, {"secondary": ["sec"], "tertiary": ["tert"]}
        )
        self.assertEqual(
            a.domains, {"primary": ["test"], "secondary": ["sec"], "tertiary": ["tert"]}
        )
        a = pybamm.Symbol("a", domain=["t", "e", "s"])
        self.assertEqual(a.domain, ["t", "e", "s"])
        with self.assertRaises(TypeError):
            a = pybamm.Symbol("a", domain=1)
        b = pybamm.Symbol("b", domain="test sec")
        with self.assertRaisesRegex(
            pybamm.DomainError, "Domain cannot be the same as an auxiliary domain"
        ):
            b.auxiliary_domains = {"sec": "test sec"}
        with self.assertRaisesRegex(
            pybamm.DomainError, "All auxiliary domains must be different"
        ):
            b = pybamm.Symbol(
                "b",
                domain="test",
                auxiliary_domains={"sec": ["test sec"], "tert": ["test sec"]},
            )

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
        self.assertIsInstance(a < b, pybamm.Heaviside)
        self.assertIsInstance(a <= b, pybamm.Heaviside)
        self.assertIsInstance(a > b, pybamm.Heaviside)
        self.assertIsInstance(a >= b, pybamm.Heaviside)

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
        with self.assertRaisesRegex(
            NotImplementedError, "'Addition' not implemented for symbols of type"
        ):
            a + "two"

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
        self.assertEqual(pybamm.InputParameter("a").evaluate_ignoring_errors(), 1)

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

        # Time variable returns true
        a = 3 * pybamm.t + 2
        self.assertTrue(a.evaluates_to_number())

    def test_symbol_repr(self):
        """
        test that __repr___ returns the string
        `__class__(id, name, children, domain, auxiliary_domains)`
        """
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        c = pybamm.Symbol("c", domain=["test"])
        d = pybamm.Symbol("d", domain=["test"], auxiliary_domains={"sec": "other test"})
        hex_regex = r"\-?0x[0-9,a-f]+"
        self.assertRegex(
            a.__repr__(),
            r"Symbol\("
            + hex_regex
            + r", a, children\=\[\], domain\=\[\], auxiliary_domains\=\{\}\)",
        )
        self.assertRegex(
            b.__repr__(),
            r"Symbol\("
            + hex_regex
            + r", b, children\=\[\], domain\=\[\], auxiliary_domains\=\{\}\)",
        )
        self.assertRegex(
            c.__repr__(),
            r"Symbol\("
            + hex_regex
            + r", c, children\=\[\], domain\=\['test'\], auxiliary_domains\=\{\}\)",
        )
        self.assertRegex(
            d.__repr__(),
            r"Symbol\("
            + hex_regex
            + r", d, children\=\[\], domain\=\['test'\]"
            + r", auxiliary_domains\=\{'sec': \"\['other test'\]\"\}\)",
        )
        self.assertRegex(
            (a + b).__repr__(),
            r"Addition\(" + hex_regex + r", \+, children\=\['a', 'b'\], domain=\[\]",
        )
        self.assertRegex(
            (c * d).__repr__(),
            r"Multiplication\("
            + hex_regex
            + r", \*, children\=\['c', 'd'\], domain=\['test'\]"
            + r", auxiliary_domains\=\{'sec': \"\['other test'\]\"\}\)",
        )
        self.assertRegex(
            pybamm.grad(c).__repr__(),
            r"Gradient\("
            + hex_regex
            + r", grad, children\=\['c'\], domain=\['test'\]"
            + r", auxiliary_domains\=\{\}\)",
        )

    def test_symbol_visualise(self):

        param = pybamm.standard_parameters_lithium_ion

        zero_n = pybamm.FullBroadcast(0, ["negative electrode"], "current collector")
        zero_s = pybamm.FullBroadcast(0, ["separator"], "current collector")
        zero_p = pybamm.FullBroadcast(0, ["positive electrode"], "current collector")

        zero_nsp = pybamm.Concatenation(zero_n, zero_s, zero_p)

        v_box = pybamm.Scalar(0)

        variables = {
            "Porosity": param.epsilon,
            "Electrolyte tortuosity": param.epsilon ** 1.5,
            "Porosity change": zero_nsp,
            "Electrolyte current density": zero_nsp,
            "Volume-averaged velocity": v_box,
            "Interfacial current density": zero_nsp,
            "Oxygen interfacial current density": zero_nsp,
            "Cell temperature": pybamm.Concatenation(zero_n, zero_s, zero_p),
            "Transverse volume-averaged acceleration": pybamm.Concatenation(
                zero_n, zero_s, zero_p
            ),
            "Sum of electrolyte reaction source terms": zero_nsp,
        }
        model = pybamm.electrolyte_diffusion.Full(param)
        variables.update(model.get_fundamental_variables())
        variables.update(model.get_coupled_variables(variables))

        model.set_rhs(variables)

        rhs = list(model.rhs.values())[0]
        rhs.visualise("StefanMaxwell_test.png")
        self.assertTrue(os.path.exists("StefanMaxwell_test.png"))
        with self.assertRaises(ValueError):
            rhs.visualise("StefanMaxwell_test")

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
        self.assertIsNone(a_orp.parent)
        self.assertIsNone(b_orp.parent)
        self.assertEqual(a.id, a_orp.id)
        self.assertEqual(b.id, b_orp.id)

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

        state = pybamm.StateVector(slice(10, 25))
        self.assertEqual(state.shape_for_testing, state.shape)

        param = pybamm.Parameter("a")
        self.assertEqual(param.shape_for_testing, ())

        func = pybamm.FunctionParameter("func", {"state": state})
        self.assertEqual(func.shape_for_testing, state.shape_for_testing)

        concat = pybamm.Concatenation()
        self.assertEqual(concat.shape_for_testing, (0,))
        concat = pybamm.Concatenation(state, state)
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


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
