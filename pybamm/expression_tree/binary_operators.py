#
# Binary operator classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import copy


class BinaryOperator(pybamm.Symbol):
    def __init__(self, name, left, right, parent=None):
        super().__init__(name, children=[left, right], parent=parent)

    def __str__(self):
        return "{!s} {} {!s}".format(self.children[0], self.name, self.children[1])


class Addition(BinaryOperator):
    def __init__(self, left, right, parent=None):
        super().__init__("+", left, right, parent)

    def evaluate(self, y):
        return self.children[0].evaluate(y) + self.children[1].evaluate(y)


class Subtraction(BinaryOperator):
    def __init__(self, left, right, parent=None):
        super().__init__("-", left, right, parent)

    def evaluate(self, y):
        return self.children[0].evaluate(y) - self.children[1].evaluate(y)


class Multiplication(BinaryOperator):
    def __init__(self, left, right, parent=None):
        super().__init__("*", left, right, parent)

    def evaluate(self, y):
        return self.children[0].evaluate(y) * self.children[1].evaluate(y)


class Division(BinaryOperator):
    def __init__(self, left, right, parent=None):
        super().__init__("/", left, right, parent)

    def evaluate(self, y):
        return self.children[0].evaluate(y) / self.children[1].evaluate(y)


class MatrixVectorMultiplication(BinaryOperator):
    def __init__(self, left, right, parent=None):
        super().__init__("@", left, right, parent)

    def evaluate(self, y):
        return self.children[0].evaluate(y) @ self.children[1].evaluate(y)
