#
# Binary operator classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class BinaryOperator(pybamm.Symbol):
    def __init__(self, name, left, right, parent=None):
        super().__init__(name, parent)
        self._left = left
        self._right = right

    def __str__(self):
        return "{!s} {} {!s}".format(self._left, self._name, self._right)


class Addition(BinaryOperator):
    def __init__(self, left, right, parent=None):
        super().__init__("+", left, right, parent)

    def evaluate(self, y):
        return self._left.evaluate(y) + self._right.evaluate(y)


class Subtraction(BinaryOperator):
    def __init__(self, left, right, parent=None):
        super().__init__("-", left, right, parent)

    def evaluate(self, y):
        return self._left.evaluate(y) - self._right.evaluate(y)


class Multiplication(BinaryOperator):
    def __init__(self, left, right, parent=None):
        super().__init__("*", left, right, parent)

    def evaluate(self, y):
        return self._left.evaluate(y) * self._right.evaluate(y)


class Division(BinaryOperator):
    def __init__(self, left, right, parent=None):
        super().__init__("/", left, right, parent)

    def evaluate(self, y):
        return self._left.evaluate(y) / self._right.evaluate(y)


class MatrixVectorMultiplication(BinaryOperator):
    def __init__(self, left, right, parent=None):
        super().__init__("@", left, right, parent)

    def evaluate(self, y):
        return self.left.evaluate(y) @ self.right.evaluate(y)
