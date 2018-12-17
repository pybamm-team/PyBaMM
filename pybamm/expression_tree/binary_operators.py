#
# Binary operator classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import copy


class BinaryOperator(pybamm.Symbol):
    """A node in the expression tree representing a binary operator (e.g. +, *)

    Derived classes will specify the particular operator

    Arguments:

    ``name`` (str)
        name of the node
    ``left`` (node)
        lhs child node
    ``right`` (node)
        rhs child node
    """

    def __init__(self, name, left, right):

        super().__init__(name, children=[left, right])

    def __str__(self):
        """return a string representation of the node and its children"""
        return "{!s} {} {!s}".format(self.children[0], self.name, self.children[1])


class Addition(BinaryOperator):
    """A node in the expression tree representing an addition operator

    *Extends:* :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__("+", left, right)

    def evaluate(self, y):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(y) + self.children[1].evaluate(y)


class Subtraction(BinaryOperator):
    """A node in the expression tree representing a subtraction operator

    *Extends:* :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """

        super().__init__("-", left, right)

    def evaluate(self, y):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(y) - self.children[1].evaluate(y)


class Multiplication(BinaryOperator):
    """A node in the expression tree representing a multiplication operator

    *Extends:* :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """

        super().__init__("*", left, right)

    def evaluate(self, y):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(y) * self.children[1].evaluate(y)


class Division(BinaryOperator):
    """A node in the expression tree representing a division operator

    *Extends:* :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__("/", left, right)

    def evaluate(self, y):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(y) / self.children[1].evaluate(y)


class MatrixVectorMultiplication(BinaryOperator):
    """A node in the expression tree representing a matrix vector multiplication operator

    *Extends:* :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """

        super().__init__("@", left, right)

    def evaluate(self, y):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(y) @ self.children[1].evaluate(y)
