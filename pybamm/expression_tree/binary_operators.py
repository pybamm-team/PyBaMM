#
# Binary operator classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numbers
import autograd.numpy as np
import copy


def simplify_addition_subtraction(myclass, left, right):
    """
    some common simplifications for addition and subtraction

    if children are associative (addition, subtraction, etc) then try to find
    pairs of constant children and simplify them
    """
    for child1, child2 in zip([left, right], [right, left]):
        if child1.is_constant():
            if isinstance(child2, pybamm.Addition) or \
                    isinstance(child2, pybamm.Subtraction):
                # don't care about ordering
                for i in range(2):
                    if child2.children[i].is_constant():
                        tmp = copy.deepcopy(child2.children[i])
                        tmp.parent = None
                        left = myclass(child1, tmp)
                        left = pybamm.simplify_if_constant(left)
                        right = copy.deepcopy(child2.children[1 - i])
                        right.parent = None
                        return child2.__class__(left, right)

    return myclass(left, right)


def simplify_multiplication_division(myclass, left, right):
    """
    some common simplifications for multiplication and division

    if children are associative (multiply, division, etc) then try to find
    pairs of constant children and simplify them
    """
    numerator_constant = []
    denominator_constant = []
    numerator_nonconstant = []
    denominator_nonconstant = []
    matrix_multiplys = []
    in_numerator = True
    for child in [left, right]:
        if isinstance(child, pybamm.Multiplication) or isinstance(child, pybamm.Division):
            for i in range(2):
                tmp = copy.deepcopy(child.children[i])
                tmp.parent = None
                if in_numerator:
                    if tmp.is_constant():
                        numerator_constant.append(tmp)
                    else:
                        numerator_nonconstant.append(tmp)
                else:
                    if tmp.is_constant():
                        denominator_constant.append(tmp)
                    else:
                        denominator_nonconstant.append(tmp)
                if i == 0 and isinstance(child, pybamm.Division):
                    in_numerator = not in_numerator
        if child == left and myclass == pybamm.Division:
            in_numerator = not in_numerator

        elif isinstance(child, pybamm.MatrixMultiplication):
            if not in_numerator:
                raise pybamm.ModelError('matrix multiplication found on denominator!')
            matrix_multiplys.append(child)



    new_numerator = pybamm.Scalar(1)
    for child in numerator_constant:
        new_numerator = new_numerator * child
    new_numerator = pybamm.simplify_if_constant(new_numerator)
    for child in numerator_nonconstant:
        new_numerator = new_numerator * child

    new_denominator = pybamm.Scalar(1)
    for child in denominator_constant:
        new_denominator = new_denominator * child
    new_denominator= pybamm.simplify_if_constant(new_denominator)
    for child in denominator_constant:
        new_denominator = new_denominator * child

    new_mat_muls = 1

    new_expression = new_numerator / new_denominator




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

        return simplify_addition_subtraction(self.__class__, left, right)


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

        return simplify_addition_subtraction(self.__class__, left, right)


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

        return simplify_multiplication_division(self.__class__, left, right)


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

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """

        return self.children[0].evaluate(t, y) @ self.children[1].evaluate(t, y)

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.simplify()`. """

        # anything multiplied by a scalar zero returns a scalar zero
        if is_zero(left) or is_zero(right):
            return pybamm.Scalar(0)

        # if children are associative (multiply, division, etc) then try to find
        # pairs of constant children and simplify them
        for child1, child2 in zip([left, right], [right, left]):
            if child1.is_constant():
                if isinstance(child2, pybamm.Multiplication) or \
                        isinstance(child2, pybamm.Division) or \
                        isinstance(child2, pybamm.MatrixMultiplication):
                    # we care about ordering here
                    if child1 == left and child2.children[0].is_constant():
                        tmp = copy.deepcopy(child2.children[0])
                        tmp.parent = None
                        left = self.__class__(child1, tmp)
                        left = pybamm.simplify_if_constant(left)
                        right = copy.deepcopy(child2.children[1])
                        right.parent = None
                        return child2.__class__(left, right)
                    elif child1 == right and child2.children[1].is_constant():
                        tmp = copy.deepcopy(child2.children[1])
                        tmp.parent = None
                        left = copy.deepcopy(child2.children[0])
                        left.parent = None
                        right = self.__class__(tmp, child1)
                        right = pybamm.simplify_if_constant(right)
                        return child2.__class__(left, right)

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

        return simplify_multiplication_division(self.__class__, left, right)
