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
    pairs of constant children (that produce a value) and simplify them

    """
    numerator = []
    numerator_types = []

    # recursive function to flatten a term involving only additions or subtractions
    def flatten(previous_class, this_class, left_child, right_child):
        for child in [left_child, right_child]:
            if isinstance(child, pybamm.Addition) \
                    or isinstance(child, pybamm.Subtraction):
                left = copy.deepcopy(child.children[0])
                left.parent = None
                right = copy.deepcopy(child.children[1])
                right .parent = None
                if child == left_child:
                    flatten(previous_class, child.__class__, left, right)
                else:
                    flatten(this_class, child.__class__, left, right)

            else:
                numerator.append(child)
                if child == left_child:
                    numerator_types.append(previous_class)
                else:
                    numerator_types.append(this_class)

    flatten(None, myclass, left, right)

    # function to partition a source list of symbols into those that return a constant
    # value, and those that do not
    def partition_by_constant(source, types):
        constant = []
        nonconstant = []
        constant_t = []
        nonconstant_t = []

        for child, t in zip(source, types):
            if child.is_constant():
                result = child.evaluate_ignoring_errors()
                if result is not None:
                    constant.append(child)
                    constant_t.append(t)
                else:
                    nonconstant.append(child)
                    nonconstant_t.append(t)
            else:
                nonconstant.append(child)
                nonconstant_t.append(t)
        return constant, nonconstant, constant_t, nonconstant_t

    def fold_add_subtract(array, types):
        ret = None
        if len(array) > 0:
            ret = array[0]
            for child, t in zip(array[1:], types[1:]):
                if t == pybamm.Addition:
                    ret += child
                else:
                    ret -= child
        return ret

    # can reorder the numerator
    (constant, nonconstant,
     constant_t, nonconstant_t) = partition_by_constant(numerator, numerator_types)

    constant_expr = fold_add_subtract(constant, constant_t)
    nonconstant_expr = fold_add_subtract(nonconstant, nonconstant_t)

    if constant_expr is not None and nonconstant_expr is None:
        # might be no nonconstants
        new_expression = pybamm.simplify_if_constant(constant_expr)
    elif constant_expr is None and nonconstant_expr is not None:
        # might be no constants
        new_expression = nonconstant_expr
    else:
        # or mix of both
        constant_expr = pybamm.simplify_if_constant(constant_expr)
        if constant_t[0] is None and nonconstant_t[0] == pybamm.Addition:
            new_expression = constant_expr + nonconstant_expr
        elif constant_t[0] is None and nonconstant_t[0] == pybamm.Subtraction:
            new_expression = constant_expr - nonconstant_expr
        elif nonconstant_t[0] is None and constant_t[0] == pybamm.Addition:
            new_expression = nonconstant_expr + constant_expr
        else:
            assert constant_t[0] == pybamm.Subtraction
            new_expression = nonconstant_expr - constant_expr

    return new_expression


def simplify_multiplication_division(myclass, left, right):
    """
    some common simplifications for multiplication and division

    if children are associative (multiply, division, etc) then try to find
    pairs of constant children (that produce a value) and simplify them

    can handle matrix multiplication. If there are any on the numerator, it will only
    try to simplify pairs of neighbouring constant children. If there are any matrix
    multiplications on the denominator an exception is raised
    """
    numerator = []
    denominator = []
    numerator_types = []

    # recursive function to flatten a term involving only multiplications or divisions
    def flatten(previous_class, this_class, left_child, right_child, in_numerator):
        for child in [left_child, right_child]:

            if isinstance(child, pybamm.Multiplication) \
                    or isinstance(child, pybamm.Division) \
                    or isinstance(child, pybamm.MatrixMultiplication):
                left = copy.deepcopy(child.children[0])
                left.parent = None
                right = copy.deepcopy(child.children[1])
                right .parent = None
                if child == left_child:
                    flatten(previous_class, child.__class__, left, right, in_numerator)
                else:
                    flatten(this_class, child.__class__, left, right, in_numerator)
            else:
                if in_numerator:
                    numerator.append(child)
                    if child == left_child:
                        numerator_types.append(previous_class)
                    else:
                        numerator_types.append(this_class)
                else:
                    denominator.append(child)
                    if this_class == pybamm.MatrixMultiplication:
                        raise pybamm.ModelError(
                            'matrix multiplication on the denominator')

            if child == left_child and this_class == pybamm.Division:
                in_numerator = not in_numerator

    flatten(None, myclass, left, right, True)

    # check if there is a matrix multiply in the numerator (if so we can't reorder it)
    has_matrix_multiply = myclass == pybamm.MatrixMultiplication
    for t in numerator_types:
        has_matrix_multiply |= t == pybamm.MatrixMultiplication

    # function to partition a source list of symbols into those that return a constant
    # value, and those that do not
    def partition_by_constant(source, types=None):
        constant = []
        nonconstant = []

        for child in source:
            if child.is_constant():
                result = child.evaluate_ignoring_errors()
                if result is not None:
                    constant.append(child)
                else:
                    nonconstant.append(child)
            else:
                nonconstant.append(child)
        return constant, nonconstant

    # assume everything in denominator is a scalar so can reorder
    denominator_constant, denominator_nonconstant = partition_by_constant(denominator)

    def fold_multiply(array, types=None):
        ret = None
        if len(array) > 0:
            ret = array[0]
            if types is None:
                for child in array[1:]:
                    ret *= child
            else:
                for child, t in zip(array[1:], types[1:]):
                    if t == pybamm.MatrixMultiplication:
                        ret = ret @ child
                    else:
                        ret *= child
        return ret

    # compact denominator_constant to a constant scalar
    constant_denominator_expr = fold_multiply(denominator_constant)

    # see if there is a constant term in the numerator.
    # if so combine with constant denominator term
    found_a_constant = False
    for i, child in enumerate(numerator):
        if child.is_constant():
            result = child.evaluate_ignoring_errors()
            if result is not None:
                if constant_denominator_expr is not None:
                    numerator[i] = \
                        pybamm.simplify_if_constant(child / constant_denominator_expr)
                found_a_constant = True

    if not found_a_constant:
        # not possible to simplify numerator, just return as is
        new_numerator = fold_multiply(numerator, numerator_types)

        # there has to be at least one numerator
        if constant_denominator_expr is not None:
            # better to invert the constant denominator to get rid of the divide
            invert_constant_denom = \
                pybamm.simplify_if_constant(1 / constant_denominator_expr)
            new_numerator = invert_constant_denom * new_numerator

    # here we have determined that there is at least one constant in the numerator and
    # we've combined all the constants in the denominator with it
    elif has_matrix_multiply:
        # only consider neighbouring children for numerator as we can't reorder mat muls
        new_numerator = [numerator[0]]
        new_numerator_t = [numerator_types[0]]
        for child, t in zip(numerator[1:], numerator_types[1:]):
            if new_numerator[-1].is_constant() and child.is_constant() \
                    and new_numerator[-1].evaluate_ignoring_errors() is not None \
                    and child.evaluate_ignoring_errors() is not None:
                if t == pybamm.MatrixMultiplication:
                    new_numerator[-1] = new_numerator[-1] @ child
                else:
                    new_numerator[-1] *= child
                new_numerator[-1] = pybamm.simplify_if_constant(new_numerator[-1])
            else:
                new_numerator.append(child)
                new_numerator_t.append(t)
        new_numerator = fold_multiply(new_numerator, new_numerator_t)

    else:
        # can reorder the numerator since no matrix multiplies
        numerator_constant, numerator_nonconstant = partition_by_constant(numerator)

        constant_numerator_expr = fold_multiply(numerator_constant)
        nonconstant_numerator_expr = fold_multiply(numerator_nonconstant)

        # there is at least one constant in the numerator
        constant_numerator_expr = pybamm.simplify_if_constant(constant_numerator_expr)

        # might be no nonconstant numerator terms
        if nonconstant_numerator_expr is None:
            new_numerator = constant_numerator_expr
        else:
            new_numerator = constant_numerator_expr * nonconstant_numerator_expr

    # combine new numerator with the nonconstant denominator terms (if they exist) and
    # return the simplified expression
    new_expression = new_numerator
    new_denominator = fold_multiply(denominator_nonconstant)
    if new_denominator is not None:
        new_expression /= new_denominator

    return new_expression


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

        return simplify_multiplication_division(self.__class__, left, right)


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
