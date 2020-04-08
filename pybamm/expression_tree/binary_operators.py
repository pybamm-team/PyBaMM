#
# Binary operator classes
#
import pybamm

import numpy as np
import numbers
from scipy.sparse import issparse, csr_matrix


def is_scalar_zero(expr):
    """
    Utility function to test if an expression evaluates to a constant scalar zero
    """
    if expr.is_constant():
        result = expr.evaluate_ignoring_errors(t=None)
        return isinstance(result, numbers.Number) and result == 0
    else:
        return False


def is_matrix_zero(expr):
    """
    Utility function to test if an expression evaluates to a constant matrix zero
    """
    if expr.is_constant():
        result = expr.evaluate_ignoring_errors(t=None)
        return (issparse(result) and result.count_nonzero() == 0) or (
            isinstance(result, np.ndarray) and np.all(result == 0)
        )
    else:
        return False


def is_scalar_one(expr):
    """
    Utility function to test if an expression evaluates to a constant scalar one
    """
    if expr.is_constant():
        result = expr.evaluate_ignoring_errors(t=None)
        return isinstance(result, numbers.Number) and result == 1
    else:
        return False


def zeros_of_shape(shape):
    """
    Utility function to create a scalar zero, or a vector or matrix of zeros of
    the correct shape
    """
    if shape == ():
        return pybamm.Scalar(0)
    else:
        if len(shape) == 1 or shape[1] == 1:
            return pybamm.Vector(np.zeros(shape))
        else:
            return pybamm.Matrix(csr_matrix(shape))


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
        left, right = self.format(left, right)

        domain = self.get_children_domains(left.domain, right.domain)
        auxiliary_domains = self.get_children_auxiliary_domains([left, right])
        super().__init__(
            name,
            children=[left, right],
            domain=domain,
            auxiliary_domains=auxiliary_domains,
        )
        self.left = self.children[0]
        self.right = self.children[1]

    def format(self, left, right):
        "Format children left and right into compatible form"
        # Turn numbers into scalars
        if isinstance(left, numbers.Number):
            left = pybamm.Scalar(left)
        if isinstance(right, numbers.Number):
            right = pybamm.Scalar(right)

        # Check both left and right are pybamm Symbols
        if not (isinstance(left, pybamm.Symbol) and isinstance(right, pybamm.Symbol)):
            raise NotImplementedError(
                """'{}' not implemented for symbols of type {} and {}""".format(
                    self.__class__.__name__, type(left), type(right)
                )
            )

        # Do some broadcasting in special cases, to avoid having to do this manually
        if left.domain != [] and right.domain != []:
            if (
                left.domain != right.domain
                and "secondary" in right.auxiliary_domains
                and left.domain == right.auxiliary_domains["secondary"]
            ):
                left = pybamm.PrimaryBroadcast(left, right.domain)
            if (
                right.domain != left.domain
                and "secondary" in left.auxiliary_domains
                and right.domain == left.auxiliary_domains["secondary"]
            ):
                right = pybamm.PrimaryBroadcast(right, left.domain)

        return left, right

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        return "{!s} {} {!s}".format(self.left, self.name, self.right)

    def get_children_domains(self, ldomain, rdomain):
        "Combine domains from children in appropriate way"
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

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """

        # process children
        new_left = self.left.new_copy()
        new_right = self.right.new_copy()

        # make new symbol, ensure domain(s) remain the same
        out = self._binary_new_copy(new_left, new_right)
        out.copy_domains(self)

        return out

    def _binary_new_copy(self, left, right):
        "Default behaviour for new_copy"
        return self.__class__(left, right)

    def evaluate(self, t=None, y=None, y_dot=None, inputs=None, known_evals=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        if known_evals is not None:
            id = self.id
            try:
                return known_evals[id], known_evals
            except KeyError:
                left, known_evals = self.left.evaluate(t, y, y_dot, inputs, known_evals)
                right, known_evals = self.right.evaluate(
                    t, y, y_dot, inputs, known_evals
                )
                value = self._binary_evaluate(left, right)
                known_evals[id] = value
                return value, known_evals
        else:
            left = self.left.evaluate(t, y, y_dot, inputs)
            right = self.right.evaluate(t, y, y_dot, inputs)
            return self._binary_evaluate(left, right)

    def _evaluate_for_shape(self):
        """ See :meth:`pybamm.Symbol.evaluate_for_shape()`. """
        left = self.children[0].evaluate_for_shape()
        right = self.children[1].evaluate_for_shape()
        return self._binary_evaluate(left, right)

    def _binary_jac(self, left_jac, right_jac):
        """ Calculate the jacobian of a binary operator. """
        raise NotImplementedError

    def _binary_simplify(self, new_left, new_right):
        """ Simplify a binary operator. Default behaviour: unchanged"""
        return self._binary_new_copy(new_left, new_right)

    def _binary_evaluate(self, left, right):
        """ Perform binary operation on nodes 'left' and 'right'. """
        raise NotImplementedError

    def evaluates_on_edges(self):
        """ See :meth:`pybamm.Symbol.evaluates_on_edges()`. """
        return self.left.evaluates_on_edges() or self.right.evaluates_on_edges()


class Power(BinaryOperator):
    """A node in the expression tree representing a `**` power operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__("**", left, right)

    def _diff(self, variable):
        """ See :meth:`pybamm.Symbol._diff()`. """
        # apply chain rule and power rule
        base, exponent = self.orphans
        # derivative if variable is in the base
        diff = exponent * (base ** (exponent - 1)) * base.diff(variable)
        # derivative if variable is in the exponent (rare, check separately to avoid
        # unecessarily big tree)
        if any(variable.id == x.id for x in exponent.pre_order()):
            diff += (base ** exponent) * pybamm.log(base) * exponent.diff(variable)
        return diff

    def _binary_jac(self, left_jac, right_jac):
        """ See :meth:`pybamm.BinaryOperator._binary_jac()`. """
        # apply chain rule and power rule
        left, right = self.orphans
        if left.evaluates_to_number() and right.evaluates_to_number():
            return pybamm.Scalar(0)
        elif right.evaluates_to_number():
            return (right * left ** (right - 1)) * left_jac
        elif left.evaluates_to_number():
            return (left ** right * pybamm.log(left)) * right_jac
        else:
            return (left ** (right - 1)) * (
                right * left_jac + left * pybamm.log(left) * right_jac
            )

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """
        # don't raise RuntimeWarning for NaNs
        with np.errstate(invalid="ignore"):
            return left ** right

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_simplify()`. """

        # anything to the power of zero is one
        if is_scalar_zero(right):
            return pybamm.Scalar(1)

        # zero to the power of anything is zero
        if is_scalar_zero(left):
            return pybamm.Scalar(0)

        # anything to the power of one is itself
        if is_scalar_one(right):
            return left

        return self.__class__(left, right)


class Addition(BinaryOperator):
    """A node in the expression tree representing an addition operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__("+", left, right)

    def _diff(self, variable):
        """ See :meth:`pybamm.Symbol._diff()`. """
        return self.left.diff(variable) + self.right.diff(variable)

    def _binary_jac(self, left_jac, right_jac):
        """ See :meth:`pybamm.BinaryOperator._binary_jac()`. """
        return left_jac + right_jac

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """
        return left + right

    def _binary_simplify(self, left, right):
        """
        See :meth:`pybamm.BinaryOperator._binary_simplify()`.

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
            if isinstance(right, pybamm.Scalar):
                return pybamm.Array(right.value * np.ones(left.shape_for_testing))
            else:
                return right
        if is_matrix_zero(right):
            if isinstance(left, pybamm.Scalar):
                return pybamm.Array(left.value * np.ones(right.shape_for_testing))
            else:
                return left

        return pybamm.simplify_addition_subtraction(self.__class__, left, right)


class Subtraction(BinaryOperator):
    """A node in the expression tree representing a subtraction operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """

        super().__init__("-", left, right)

    def _diff(self, variable):
        """ See :meth:`pybamm.Symbol._diff()`. """
        return self.left.diff(variable) - self.right.diff(variable)

    def _binary_jac(self, left_jac, right_jac):
        """ See :meth:`pybamm.BinaryOperator._binary_jac()`. """
        return left_jac - right_jac

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """
        return left - right

    def _binary_simplify(self, left, right):
        """
        See :meth:`pybamm.BinaryOperator._binary_simplify()`.

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
            if isinstance(right, pybamm.Scalar):
                return pybamm.Array(-right.value * np.ones(left.shape_for_testing))
            else:
                return -right
        if is_matrix_zero(right):
            if isinstance(left, pybamm.Scalar):
                return pybamm.Array(left.value * np.ones(right.shape_for_testing))
            else:
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

    def _diff(self, variable):
        """ See :meth:`pybamm.Symbol._diff()`. """
        # apply product rule
        left, right = self.orphans
        return left.diff(variable) * right + left * right.diff(variable)

    def _binary_jac(self, left_jac, right_jac):
        """ See :meth:`pybamm.BinaryOperator._binary_jac()`. """
        # apply product rule
        left, right = self.orphans
        if left.evaluates_to_number() and right.evaluates_to_number():
            return pybamm.Scalar(0)
        elif left.evaluates_to_number():
            return left * right_jac
        elif right.evaluates_to_number():
            return right * left_jac
        else:
            return right * left_jac + left * right_jac

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """

        if issparse(left):
            return csr_matrix(left.multiply(right))
        elif issparse(right):
            # Hadamard product is commutative, so we can switch right and left
            return csr_matrix(right.multiply(left))
        else:
            return left * right

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_simplify()`. """

        # simplify multiply by scalar zero, being careful about shape
        if is_scalar_zero(left):
            return zeros_of_shape(right.shape_for_testing)
        if is_scalar_zero(right):
            return zeros_of_shape(left.shape_for_testing)

        # if one of the children is a zero matrix, we have to be careful about shapes
        if is_matrix_zero(left) or is_matrix_zero(right):
            shape = (left * right).shape
            return zeros_of_shape(shape)

        # anything multiplied by a scalar one returns itself
        if is_scalar_one(left):
            return right
        if is_scalar_one(right):
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
        raise NotImplementedError(
            "diff not implemented for symbol of type 'MatrixMultiplication'"
        )

    def _binary_jac(self, left_jac, right_jac):
        """ See :meth:`pybamm.BinaryOperator._binary_jac()`. """
        # We only need the case where left is an array and right
        # is a (slice of a) state vector, e.g. for discretised spatial
        # operators of the form D @ u (also catch cases of (-D) @ u)
        left, right = self.orphans
        if isinstance(left, pybamm.Array) or (
            isinstance(left, pybamm.Negate) and isinstance(left.child, pybamm.Array)
        ):
            left = pybamm.Matrix(csr_matrix(left.evaluate()))
            return left @ right_jac
        else:
            raise NotImplementedError(
                """jac of 'MatrixMultiplication' is only
             implemented for left of type 'pybamm.Array',
             not {}""".format(
                    left.__class__
                )
            )

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """
        return left @ right

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_simplify()`. """
        if is_matrix_zero(left) or is_matrix_zero(right):
            shape = (left @ right).shape
            return zeros_of_shape(shape)

        return pybamm.simplify_multiplication_division(self.__class__, left, right)


class Division(BinaryOperator):
    """A node in the expression tree representing a division operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__("/", left, right)

    def _diff(self, variable):
        """ See :meth:`pybamm.Symbol._diff()`. """
        # apply quotient rule
        top, bottom = self.orphans
        return (top.diff(variable) * bottom - top * bottom.diff(variable)) / bottom ** 2

    def _binary_jac(self, left_jac, right_jac):
        """ See :meth:`pybamm.BinaryOperator._binary_jac()`. """
        # apply quotient rule
        left, right = self.orphans
        if left.evaluates_to_number() and right.evaluates_to_number():
            return pybamm.Scalar(0)
        elif left.evaluates_to_number():
            return -left / right ** 2 * right_jac
        elif right.evaluates_to_number():
            return left_jac / right
        else:
            return (right * left_jac - left * right_jac) / right ** 2

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """

        if issparse(left):
            return csr_matrix(left.multiply(1 / right))
        else:
            if isinstance(right, numbers.Number) and right == 0:
                # don't raise RuntimeWarning for NaNs
                with np.errstate(invalid="ignore"):
                    return left * np.inf
            else:
                return left / right

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_simplify()`. """

        # zero divided by zero returns nan scalar
        if is_scalar_zero(left) and is_scalar_zero(right):
            return pybamm.Scalar(np.nan)

        # zero divided by anything returns zero (being careful about shape)
        if is_scalar_zero(left):
            return zeros_of_shape(right.shape_for_testing)

        # matrix zero divided by anything returns matrix zero (i.e. itself)
        if is_matrix_zero(left):
            return left

        # anything divided by zero returns inf
        if is_scalar_zero(right):
            if left.shape_for_testing == ():
                return pybamm.Scalar(np.inf)
            else:
                return pybamm.Array(np.inf * np.ones(left.shape_for_testing))

        # anything divided by one is itself
        if is_scalar_one(right):
            return left

        return pybamm.simplify_multiplication_division(self.__class__, left, right)


class Inner(BinaryOperator):
    """
    A node in the expression tree which represents the inner (or dot) product. This
    operator should be used to take the inner product of two mathematical vectors
    (as opposed to the computational vectors arrived at post-discretisation) of the
    form v = v_x e_x + v_y e_y + v_z e_z where v_x, v_y, v_z are scalars
    and e_x, e_y, e_z are x-y-z-directional unit vectors. For v and w mathematical
    vectors, inner product returns v_x * w_x + v_y * w_y + v_z * w_z. In addition,
    for some spatial discretisations mathematical vector quantities (such as
    i = grad(phi) ) are evaluated on a different part of the grid to mathematical
    scalars (e.g. for finite volume mathematical scalars are evaluated on the nodes but
    mathematical vectors are evaluated on cell edges). Therefore, inner also transfers
    the inner product of the vector onto the scalar part of the grid if required
    by a particular discretisation.

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__("inner product", left, right)

    def _diff(self, variable):
        """ See :meth:`pybamm.Symbol._diff()`. """
        # apply product rule
        left, right = self.orphans
        return left.diff(variable) * right + left * right.diff(variable)

    def _binary_jac(self, left_jac, right_jac):
        """ See :meth:`pybamm.BinaryOperator._binary_jac()`. """
        # apply product rule
        left, right = self.orphans
        if left.evaluates_to_number() and right.evaluates_to_number():
            return pybamm.Scalar(0)
        elif left.evaluates_to_number():
            return left * right_jac
        elif right.evaluates_to_number():
            return right * left_jac
        else:
            return right * left_jac + left * right_jac

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
        """ See :meth:`pybamm.BinaryOperator._binary_simplify()`. """

        # simplify multiply by scalar zero, being careful about shape
        if is_scalar_zero(left):
            return zeros_of_shape(right.shape_for_testing)
        if is_scalar_zero(right):
            return zeros_of_shape(left.shape_for_testing)

        # if one of the children is a zero matrix, we have to be careful about shapes
        if is_matrix_zero(left) or is_matrix_zero(right):
            shape = (left * right).shape
            return zeros_of_shape(shape)

        # anything multiplied by a scalar one returns itself
        if is_scalar_one(left):
            return right
        if is_scalar_one(right):
            return left

        return pybamm.simplify_multiplication_division(self.__class__, left, right)

    def evaluates_on_edges(self):
        """ See :meth:`pybamm.Symbol.evaluates_on_edges()`. """
        return False


def inner(left, right):
    """
    Return inner product of two symbols.
    """
    return pybamm.Inner(left, right)


class Heaviside(BinaryOperator):
    """A node in the expression tree representing a heaviside step function.

    Adding this operation to the rhs or algebraic equations in a model can often cause a
    discontinuity in the solution. For the specific cases listed below, this will be
    automatically handled by the solver. In the general case, you can explicitly tell
    the solver of discontinuities by adding a :class:`Event` object with
    :class:`EventType` DISCONTINUITY to the model's list of events.

    In the case where the Heaviside function is of the form `pybamm.t < x`, `pybamm.t <=
    x`, `x < pybamm.t`, or `x <= pybamm.t`, where `x` is any constant equation, this
    DISCONTINUITY event will automatically be added by the solver.

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, name, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__(name, left, right)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        # Heaviside should always be multiplied by something else so hopefully don't
        # need to worry about shape
        return pybamm.Scalar(0)

    def _binary_jac(self, left_jac, right_jac):
        """ See :meth:`pybamm.BinaryOperator._binary_jac()`. """
        # Heaviside should always be multiplied by something else so hopefully don't
        # need to worry about shape
        return pybamm.Scalar(0)


class EqualHeaviside(Heaviside):
    "A heaviside function with equality (return 1 when left = right)"

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__("<=", left, right)

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        return "{!s} <= {!s}".format(self.left, self.right)

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """
        # don't raise RuntimeWarning for NaNs
        with np.errstate(invalid="ignore"):
            return left <= right


class NotEqualHeaviside(Heaviside):
    "A heaviside function without equality (return 0 when left = right)"

    def __init__(self, left, right):
        super().__init__("<", left, right)

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        return "{!s} < {!s}".format(self.left, self.right)

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """
        # don't raise RuntimeWarning for NaNs
        with np.errstate(invalid="ignore"):
            return left < right


class Minimum(BinaryOperator):
    " Returns the smaller of two objects "

    def __init__(self, left, right):
        super().__init__("minimum", left, right)

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        return "minimum({!s}, {!s})".format(self.left, self.right)

    def _diff(self, variable):
        """ See :meth:`pybamm.Symbol._diff()`. """
        left, right = self.orphans
        return (left <= right) * left.diff(variable) + (left > right) * right.diff(
            variable
        )

    def _binary_jac(self, left_jac, right_jac):
        """ See :meth:`pybamm.BinaryOperator._binary_jac()`. """
        left, right = self.orphans
        return (left <= right) * left_jac + (left > right) * right_jac

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """
        # don't raise RuntimeWarning for NaNs
        return np.minimum(left, right)


class Maximum(BinaryOperator):
    " Returns the smaller of two objects "

    def __init__(self, left, right):
        super().__init__("maximum", left, right)

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        return "maximum({!s}, {!s})".format(self.left, self.right)

    def _diff(self, variable):
        """ See :meth:`pybamm.Symbol._diff()`. """
        left, right = self.orphans
        return (left >= right) * left.diff(variable) + (left < right) * right.diff(
            variable
        )

    def _binary_jac(self, left_jac, right_jac):
        """ See :meth:`pybamm.BinaryOperator._binary_jac()`. """
        left, right = self.orphans
        return (left >= right) * left_jac + (left < right) * right_jac

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """
        # don't raise RuntimeWarning for NaNs
        return np.maximum(left, right)


def minimum(left, right):
    """
    Returns the smaller of two objects. Not to be confused with :meth:`pybamm.min`,
    which returns min function of child.
    """
    return pybamm.simplify_if_constant(Minimum(left, right), keep_domains=True)


def maximum(left, right):
    """
    Returns the larger of two objects. Not to be confused with :meth:`pybamm.max`,
    which returns max function of child.
    """
    return pybamm.simplify_if_constant(Maximum(left, right), keep_domains=True)


def source(left, right, boundary=False):
    """A convinience function for creating (part of) an expression tree representing
    a source term. This is necessary for spatial methods where the mass matrix
    is not the identity (e.g. finite element formulation with piecwise linear
    basis functions). The left child is the symbol representing the source term
    and the right child is the symbol of the equation variable (currently, the
    finite element formulation in PyBaMM assumes all functions are constructed
    using the same basis, and the matrix here is constructed accoutning for the
    boundary conditions of the right child). The method returns the matrix-vector
    product of the mass matrix (adjusted to account for any Dirichlet boundary
    conditions imposed the the right symbol) and the discretised left symbol.

    Parameters
    ----------

    left : :class:`Symbol`
        The left child node, which represents the expression for the source term.
    right : :class:`Symbol`
        The right child node. This is the symbol whose boundary conditions are
        accounted for in the construction of the mass matrix.
    boundary : bool, optional
        If True, then the mass matrix should is assembled over the boundary,
        corresponding to a source term which only acts on the boundary of the
        domain. If False (default), the matrix is assembled over the entire domain,
        corresponding to a source term in the bulk.

    """
    # Broadcast if left is number
    if isinstance(left, numbers.Number):
        left = pybamm.PrimaryBroadcast(left, "current collector")

    if left.domain != ["current collector"] or right.domain != ["current collector"]:
        raise pybamm.DomainError(
            """'source' only implemented in the 'current collector' domain,
            but symbols have domains {} and {}""".format(
                left.domain, right.domain
            )
        )
    if boundary:
        return pybamm.BoundaryMass(right) @ left
    else:
        return pybamm.Mass(right) @ left
