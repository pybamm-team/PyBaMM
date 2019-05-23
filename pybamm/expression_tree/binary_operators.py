#
# Binary operator classes
#
import pybamm

import autograd.numpy as np
import numbers
from scipy.sparse import issparse, csr_matrix


def is_scalar_zero(expr):
    """
    Utility function to test if an expression evaluates to a constant scalar zero
    """
    if expr.is_constant():
        result = expr.evaluate_ignoring_errors()
        return isinstance(result, numbers.Number) and result == 0
    else:
        return False


def is_matrix_zero(expr):
    """
    Utility function to test if an expression evaluates to a constant scalar zero
    """
    if expr.is_constant():
        result = expr.evaluate_ignoring_errors()
        return (issparse(result) and result.count_nonzero() == 0) or (
            isinstance(result, np.ndarray) and np.all(result == 0)
        )
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
        self.left = self.children[0]
        self.right = self.children[1]

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        return "{!s} {} {!s}".format(self.left, self.name, self.right)

    def get_children_domains(self, ldomain, rdomain):
        if ldomain == rdomain:
            return ldomain
        elif ldomain == []:
            return rdomain
        elif rdomain == []:
            return ldomain
        else:
            raise pybamm.DomainError(
                """
                children must have same (or empty) domains, but left.domain is '{}'
                and right.domain is '{}'
                """.format(
                    ldomain, rdomain
                )
            )

    def _binary_evaluate(self, left, right):
        """ Perform binary operation on nodes 'left' and 'right'. """
        raise NotImplementedError

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        # process children
        new_left = self.left.new_copy()
        new_right = self.right.new_copy()
        # make new symbol, ensure domain remains the same
        out = self.__class__(new_left, new_right)
        out.domain = self.domain
        return out

    def evaluate(self, t=None, y=None, known_evals=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        if known_evals is not None:
            id = self.id
            try:
                return known_evals[id], known_evals
            except KeyError:
                left, known_evals = self.left.evaluate(t, y, known_evals)
                right, known_evals = self.right.evaluate(t, y, known_evals)
                value = self._binary_evaluate(left, right)
                known_evals[id] = value
                return value, known_evals
        else:
            left = self.left.evaluate(t, y)
            right = self.right.evaluate(t, y)
            return self._binary_evaluate(left, right)


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
                return (exponent * base ** (exponent - 1)) * base.jac(variable)
            elif base.evaluates_to_number():
                return (
                    base ** exponent * pybamm.Function(np.log, base)
                ) * exponent.jac(variable)
            else:
                return (base ** (exponent - 1)) * (
                    exponent * base.jac(variable)
                    + base * pybamm.Function(np.log, base) * exponent.jac(variable)
                )

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """
        return left ** right

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.simplify()`. """

        # anything to the power of zero is one
        if is_scalar_zero(right):
            return pybamm.Scalar(1)

        # anything to the power of one is itself
        if is_scalar_zero(left):
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
            return self.left.diff(variable) + self.right.diff(variable)

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            return self.left.jac(variable) + self.right.jac(variable)

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """
        return left + right

    def _binary_simplify(self, left, right):
        """
        See :meth:`pybamm.BinaryOperator.simplify()`.

        Note
        ----
        We check for scalars first, then matrices. This is because
        (Zero Matrix) + (Zero Scalar)
        should return (Zero Matrix), not (Zero Scalar).
        """

        # anything added by a scalar zero returns the other child
        if is_scalar_zero(left):
            return right
        if is_scalar_zero(right):
            return left
        # Check matrices after checking scalars
        if is_matrix_zero(left):
            return right
        if is_matrix_zero(right):
            return left

        return pybamm.simplify_addition_subtraction(self.__class__, left, right)


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
            return self.left.diff(variable) - self.right.diff(variable)

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        return self.left.jac(variable) - self.right.jac(variable)

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """
        return left - right

    def _binary_simplify(self, left, right):
        """
        See :meth:`pybamm.BinaryOperator.simplify()`.

        Note
        ----
        We check for scalars first, then matrices. This is because
        (Zero Matrix) - (Zero Scalar)
        should return (Zero Matrix), not -(Zero Scalar).
        """

        # anything added by a scalar zero returns the other child
        if is_scalar_zero(left):
            return -right
        if is_scalar_zero(right):
            return left
        # Check matrices after checking scalars
        if is_matrix_zero(left):
            return -right
        if is_matrix_zero(right):
            return left

        return pybamm.simplify_addition_subtraction(self.__class__, left, right)


class Multiplication(BinaryOperator):
    """
    A node in the expression tree representing a multiplication operator
    (Hadamard product). Overloads cases where the "*" operator would usually return a
    matrix multiplication (e.g. scipy.sparse.coo.coo_matrix)

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
            return right * left.jac(variable) + left * right.jac(variable)

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """

        if issparse(left):
            return left.multiply(right)
        elif issparse(right):
            # Hadamard product is commutative, so we can switch right and left
            return right.multiply(left)
        else:
            return left * right

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.simplify()`. """

        # anything multiplied by a scalar zero returns a scalar zero
        if is_scalar_zero(left):
            if isinstance(right, pybamm.Array):
                return pybamm.Array(np.zeros(right.shape))
            else:
                return pybamm.Scalar(0)
        if is_scalar_zero(right):
            if isinstance(left, pybamm.Array):
                return pybamm.Array(np.zeros(left.shape))
            else:
                return pybamm.Scalar(0)

        # if one of the children is a zero matrix, we have to be careful about shapes
        if is_matrix_zero(left) or is_matrix_zero(right):
            shape = (left * right).shape
            if len(shape) == 1 or shape[1] == 1:
                return pybamm.Vector(np.zeros(shape))
            else:
                return pybamm.Matrix(csr_matrix(shape))

        # anything multiplied by a scalar one returns itself
        if is_one(left):
            return right
        if is_one(right):
            return left

        return pybamm.simplify_multiplication_division(self.__class__, left, right)


class MatrixMultiplication(BinaryOperator):
    """A node in the expression tree representing a matrix multiplication operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """

        super().__init__("@", left, right)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        # We shouldn't need this
        raise NotImplementedError

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        # I think we only need the case where left is an array and right
        # is a (slice of a) state vector, e.g. for discretised spatial
        # operators of the form D @ u
        left, right = self.orphans
        if isinstance(left, pybamm.Array):
            return left @ right.jac(variable)
        else:
            raise NotImplementedError

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """
        return left @ right

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.simplify()`. """
        # anything multiplied by a scalar zero returns a scalar zero
        if is_scalar_zero(left) or is_scalar_zero(right):
            return pybamm.Scalar(0)

        return pybamm.simplify_multiplication_division(self.__class__, left, right)


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
            return -top / bottom ** 2 * bottom.jac(variable)
        elif bottom.evaluates_to_number():
            return top.jac(variable) / bottom
        else:
            return (
                bottom * top.jac(variable) - top * bottom.jac(variable)
            ) / bottom ** 2

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """

        if issparse(left):
            return left.multiply(1 / right)
        else:
            return left / right

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.simplify()`. """

        # zero divided by zero returns nan scalar
        if is_scalar_zero(left) and is_scalar_zero(right):
            return pybamm.Scalar(np.nan)

        # zero divided by anything returns zero
        if is_scalar_zero(left):
            return pybamm.Scalar(0)

        # anything divided by zero returns inf
        if is_scalar_zero(right):
            return pybamm.Scalar(np.inf)

        # anything divided by one is itself
        if is_one(right):
            return left

        return pybamm.simplify_multiplication_division(self.__class__, left, right)


class Outer(BinaryOperator):
    """A node in the expression tree representing an outer product

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__("outer product", left, right)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        # to do

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        # to do

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """

        return np.outer(left, right).reshape(-1, 1)

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.simplify()`. """

        return pybamm.simplify_if_constant(self)


def outer(left, right):
    """
    Return outer product of two symbols. If the symbols have the same domain, the outer
    product is just a multiplication. If they have different domains, make a copy of the
    left child with same domain as right child, and then take outer product.
    """
    try:
        return left * right
    except pybamm.DomainError:
        left = left.new_copy()
        left.domain = right.domain
        return pybamm.Outer(left, right)
