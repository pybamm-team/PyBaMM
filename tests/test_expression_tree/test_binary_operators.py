#
# Tests for the Binary Operator classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np
import unittest


class TestBinaryOperators(unittest.TestCase):
    def test_binary_operator(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        bin = pybamm.BinaryOperator("binary test", a, b)
        self.assertEqual(bin.children[0].name, a.name)
        self.assertEqual(bin.children[1].name, b.name)

    def test_binary_operator_domains(self):
        # same domain
        a = pybamm.Symbol("a", domain=["negative electrode"])
        b = pybamm.Symbol("b", domain=["negative electrode"])
        bin1 = pybamm.BinaryOperator("binary test", a, b)
        self.assertEqual(bin1.domain, ["negative electrode"])
        # one empty domain
        c = pybamm.Symbol("c", domain=[])
        bin2 = pybamm.BinaryOperator("binary test", a, c)
        self.assertEqual(bin2.domain, ["negative electrode"])
        bin3 = pybamm.BinaryOperator("binary test", c, b)
        self.assertEqual(bin3.domain, ["negative electrode"])
        # mismatched domains
        d = pybamm.Symbol("d", domain=["positive electrode"])
        with self.assertRaises(pybamm.DomainError):
            pybamm.BinaryOperator("binary test", a, d)

    def test_addition(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        sum = pybamm.Addition(a, b)
        self.assertEqual(sum.children[0].name, a.name)
        self.assertEqual(sum.children[1].name, b.name)

    def test_power(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        pow = pybamm.Power(a, b)
        self.assertEqual(pow.name, "**")
        self.assertEqual(pow.children[0].name, a.name)
        self.assertEqual(pow.children[1].name, b.name)

        a = pybamm.Scalar(4)
        b = pybamm.Scalar(2)
        pow = pybamm.Power(a, b)
        self.assertEqual(pow.evaluate(), 16)

    def test_addition_printing(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        sum = pybamm.Addition(a, b)
        self.assertEqual(sum.name, "+")
        self.assertEqual(str(sum), "a + b")

    def test_id(self):
        a = pybamm.Scalar(4)
        b = pybamm.Scalar(5)
        bin1 = pybamm.BinaryOperator("test", a, b)
        bin2 = pybamm.BinaryOperator("test", a, b)
        bin3 = pybamm.BinaryOperator("new test", a, b)
        self.assertEqual(bin1.id, bin2.id)
        self.assertNotEqual(bin1.id, bin3.id)
        c = pybamm.Scalar(5)
        bin4 = pybamm.BinaryOperator("test", a, c)
        self.assertEqual(bin1.id, bin4.id)
        d = pybamm.Scalar(42)
        bin5 = pybamm.BinaryOperator("test", a, d)
        self.assertNotEqual(bin1.id, bin5.id)

    def test_number_overloading(self):
        a = pybamm.Scalar(4)
        prod = a * 3
        self.assertIsInstance(prod.children[1], pybamm.Scalar)
        self.assertEqual(prod.evaluate(), 12)

    def test_size_and_shape(self):
        M1 = pybamm.Matrix(np.ones((34, 67)))
        M2 = pybamm.Matrix(2 * np.ones((67, 18)))
        v1 = pybamm.Vector(3 * np.ones(67))
        v2 = pybamm.Vector(4 * np.ones(18))
        sv1 = pybamm.StateVector(slice(0, 67))
        sv2 = pybamm.StateVector(slice(67, 67 + 18))
        s = pybamm.Scalar(5)

        y = np.ones(67 + 18)
        # Elementwise
        for left, right in (
            (M1, M1),
            (v1, v1),
            (M1, v1),
            (v1, M1),
            (sv1, v1),
            (sv1, sv1),
            (M1, sv1),
            (sv1, M1),
            (s, M1),
            (M1, s),
        ):
            # Addition
            self.assertEqual(
                (left + right).shape, (left.evaluate(y=y) + right.evaluate(y=y)).shape
            )
            # Elementwise multiplication
            self.assertEqual(
                (left * right).shape, (left.evaluate(y=y) * right.evaluate(y=y)).shape
            )
            self.assertEqual(
                (left + right).size, (left.evaluate(y=y) + right.evaluate(y=y)).size
            )

        # Errors (mismatched sizes)
        for left, right in ((M1, M2), (M1, v2), (v2, M1), (v1, v2)):
            bad_prod = left * right
            with self.assertRaises(ValueError):
                bad_prod.shape

        # Matrix Multiplication
        for left, right in (
            (M1, M2),
            (M1, v1),
            (M2, v2),
            (v1, v1),
            (sv1, v1),
            (sv1, sv1),
            (M1, sv1),
            (M2, sv2),
        ):
            self.assertEqual(
                (left @ right).shape, (left.evaluate(y=y) @ right.evaluate(y=y)).shape
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
