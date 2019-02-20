#
# Binary operator classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numbers
import numpy as np


class BinaryOperator(pybamm.Symbol):
    """A node in the expression tree representing a binary operator (e.g. `+`, `*`)

    Derived classes will specify the particular operator

    **Extends**: :class:`Symbol`

    Parameters
    ----------

    name : str
        name of the node
    left : :class:`Symbol` or :class:`Number`
        lhs child node (converted to :class:`Scalar` if Number)
    right : :class:`Symbol` or :class:`Number`
        rhs child node (converted to :class:`Scalar` if Number)

    """

    def __init__(self, name, left, right):
        assert isinstance(left, (pybamm.Symbol, numbers.Number)) and isinstance(
            right, (pybamm.Symbol, numbers.Number)
        ), TypeError(
            """left and right must both be Symbols or Numbers
                but they are {} and {}""".format(
                type(left), type(right)
            )
        )
        if isinstance(left, numbers.Number):
            left = pybamm.Scalar(left)
        if isinstance(right, numbers.Number):
            right = pybamm.Scalar(right)

        domain = self.get_children_domains(left.domain, right.domain)
        super().__init__(name, children=[left, right], domain=domain)

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        return "{!s} {} {!s}".format(self.children[0], self.name, self.children[1])

    def get_children_domains(self, ldomain, rdomain):
        if ldomain == rdomain:
            return ldomain
        elif ldomain == []:
            return rdomain
        elif rdomain == []:
            return ldomain
        else:
            raise pybamm.DomainError("""children must have same (or empty) domains""")


class Power(BinaryOperator):
    """A node in the expression tree representing a `**` power operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__("**", left, right)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(t, y) ** self.children[1].evaluate(t, y)


class Addition(BinaryOperator):
    """A node in the expression tree representing an addition operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__("+", left, right)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(t, y) + self.children[1].evaluate(t, y)


class Subtraction(BinaryOperator):
    """A node in the expression tree representing a subtraction operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """

        super().__init__("-", left, right)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(t, y) - self.children[1].evaluate(t, y)


class Multiplication(BinaryOperator):
    """A node in the expression tree representing a multiplication operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """

        super().__init__("*", left, right)

    @property
    def size(self):
        """Get the size of the array, based on the size of children"""
        return np.prod(self.shape)

    @property
    def shape(self):
        """Get the shape of the array, based on the shape of children"""
        # Scalars are "invisible" for shape
        if isinstance(self.children[0], pybamm.Scalar):
            return self.children[1].shape
        elif isinstance(self.children[1], pybamm.Scalar):
            return self.children[0].shape

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(t, y) * self.children[1].evaluate(t, y)


class MatrixMultiplication(BinaryOperator):
    """A node in the expression tree representing a matrix multiplication operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """

        super().__init__("*", left, right)

    @property
    def size(self):
        """Get the size of the array, based on the size of children"""
        return np.prod(self.shape)

    @property
    def shape(self):
        """Get the shape of the array, based on the shape of children"""
        # Scalars are "invisible" for shape
        if isinstance(self.children[0], pybamm.Scalar):
            return self.children[1].shape
        elif isinstance(self.children[1], pybamm.Scalar):
            return self.children[0].shape

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(t, y) @ self.children[1].evaluate(t, y)


class Division(BinaryOperator):
    """A node in the expression tree representing a division operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__("/", left, right)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(t, y) / self.children[1].evaluate(t, y)
