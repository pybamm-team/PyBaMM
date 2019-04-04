#
# Binary operator classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numbers
import autograd.numpy as np


def is_zero(expr):
    """
    Utility function to test if an expression evaluates to a constant scalar zero
    """
    if expr.is_constant():
        result = expr.evaluate_ignoring_errors()
        return isinstance(result, numbers.Number) and result == 0
    else:
        return False


def is_one(expr):
    """
    Utility function to test if an expression evaluates to a constant scalar one
    """
    if expr.is_constant():
        result = expr.evaluate_ignoring_errors()
        return isinstance(result, numbers.Number) and result == 1
    else:
        return False


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

    def simplify(self):
        """ See :meth:`pybamm.Symbol.simplify()`. """
        left = self.children[0].simplify()
        right = self.children[1].simplify()

        # _binary_simplify defined in derived classes for specific rules
        new_node = self._binary_simplify(left, right)

        return pybamm.simplify_if_constant(new_node)


class Power(BinaryOperator):
    """A node in the expression tree representing a `**` power operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__("**", left, right)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            # apply chain rule and power rule
            base, exponent = self.orphans
            return base ** (exponent - 1) * (
                exponent * base.diff(variable)
                + base * pybamm.Function(np.log, base) * exponent.diff(variable)
            )

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            # apply chain rule and power rule
            base, exponent = self.orphans
            if base.evaluates_to_number() and exponent.evaluates_to_number():
                return pybamm.Scalar(0)
            elif exponent.evaluates_to_number():
                return pybamm.Diagonal(exponent * base ** (exponent - 1)) @ base.jac(
                    variable
                )
            elif base.evaluates_to_number():
                return pybamm.Diagonal(
                    base ** exponent * pybamm.Function(np.log, base)
                ) @ exponent.jac(variable)
            else:
                return pybamm.Diagonal(base ** (exponent - 1)) @ (
                    exponent @ base.jac(variable)
                    + pybamm.Diagonal(base)
                    @ pybamm.Diagonal(pybamm.Function(np.log, base))
                    @ exponent.jac(variable)
                )

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(t, y) ** self.children[1].evaluate(t, y)

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.simplify()`. """

        # anything to the power of zero is one
        if is_zero(right):
            return pybamm.Scalar(1)

        # anything to the power of one is itself
        if is_zero(left):
            return left

        return self.__class__(left, right)


class Addition(BinaryOperator):
    """A node in the expression tree representing an addition operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__("+", left, right)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            return self.children[0].diff(variable) + self.children[1].diff(variable)

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            return self.children[0].jac(variable) + self.children[1].jac(variable)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(t, y) + self.children[1].evaluate(t, y)

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.simplify()`. """

        # anything added by a scalar zero returns the other child
        if is_zero(left):
            return right
        if is_zero(right):
            return left

        return self.__class__(left, right)


class Subtraction(BinaryOperator):
    """A node in the expression tree representing a subtraction operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """

        super().__init__("-", left, right)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            return self.children[0].diff(variable) - self.children[1].diff(variable)

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        return self.children[0].jac(variable) - self.children[1].jac(variable)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(t, y) - self.children[1].evaluate(t, y)

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.simplify()`. """

        # anything added by a scalar zero returns the other child
        if is_zero(left):
            return -right
        if is_zero(right):
            return left

        return self.__class__(left, right)


class Multiplication(BinaryOperator):
    """A node in the expression tree representing a multiplication operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """

        super().__init__("*", left, right)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            # apply product rule
            left, right = self.orphans
            return left.diff(variable) * right + left * right.diff(variable)

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        # apply product rule
        left, right = self.orphans
        if left.evaluates_to_number() and right.evaluates_to_number():
            return pybamm.Scalar(0)
        elif left.evaluates_to_number():
            return left * right.jac(variable)
        elif right.evaluates_to_number():
            return right * left.jac(variable)
        else:
            return pybamm.Diagonal(right) @ left.jac(variable) + pybamm.Diagonal(
                left
            ) @ right.jac(variable)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(t, y) * self.children[1].evaluate(t, y)

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.simplify()`. """

        # anything multiplied by a scalar zero returns a scalar zero

        if is_zero(left) or is_zero(right):
            return pybamm.Scalar(0)

        # anything multiplied by a scalar one returns itself
        if is_one(left):
            return right
        if is_one(right):
            return left

        return self.__class__(left, right)


class MatrixMultiplication(BinaryOperator):
    """A node in the expression tree representing a matrix multiplication operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """

        super().__init__("*", left, right)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        # We shouldn't need this
        raise NotImplementedError

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        # I think we only need the case where left is a matrix and right
        # is a (slice of a) state vector, e.g. for discretised spatial
        # operators of the form D @ u
        left, right = self.orphans
        if isinstance(left, pybamm.Matrix):
            return left @ right.jac(variable)
        else:
            raise NotImplementedError

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        left, right = self.orphans
        return self.children[0].evaluate(t, y) @ self.children[1].evaluate(t, y)

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.simplify()`. """
        # anything multiplied by a scalar zero returns a scalar zero
        if is_zero(left) or is_zero(right):
            return pybamm.Scalar(0)

        return self.__class__(left, right)


class Division(BinaryOperator):
    """A node in the expression tree representing a division operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__("/", left, right)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            # apply quotient rule
            top, bottom = self.orphans
            return (
                top.diff(variable) * bottom - top * bottom.diff(variable)
            ) / bottom ** 2

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        # apply quotient rule
        top, bottom = self.orphans
        if top.evaluates_to_number() and bottom.evaluates_to_number():
            return pybamm.Scalar(0)
        elif top.evaluates_to_number():
            return -top * pybamm.Diagonal(1 / bottom ** 2) @ bottom.jac(variable)
        elif bottom.evaluates_to_number():
            return top.jac(variable) / bottom
        else:
            return pybamm.Diagonal(1 / bottom ** 2) @ (
                pybamm.Diagonal(bottom) @ top.jac(variable)
                - pybamm.Diagonal(top) @ bottom.jac(variable)
            )

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(t, y) / self.children[1].evaluate(t, y)

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.simplify()`. """

        # zero divided by zero returns nan scalar
        if is_zero(left) and is_zero(right):
            return pybamm.Scalar(np.nan)

        # zero divided by anything returns zero
        if is_zero(left):
            return pybamm.Scalar(0)

        # anything divided by zero returns inf
        if is_zero(right):
            return pybamm.Scalar(np.inf)

        # anything divided by one is itself
        if is_one(right):
            return left

        return self.__class__(left, right)
