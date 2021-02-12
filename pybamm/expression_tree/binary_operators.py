#
# Binary operator classes
#
import pybamm

import numpy as np
import numbers
from scipy.sparse import issparse, csr_matrix


def preprocess_binary(left, right):
    if isinstance(left, numbers.Number):
        left = pybamm.Scalar(left)
    if isinstance(right, numbers.Number):
        right = pybamm.Scalar(right)

    # Check both left and right are pybamm Symbols
    if not (isinstance(left, pybamm.Symbol) and isinstance(right, pybamm.Symbol)):
        raise NotImplementedError(
            """BinaryOperator not implemented for symbols of type {} and {}""".format(
                type(left), type(right)
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


def get_binary_children_domains(ldomain, rdomain):
    """Combine domains from children in appropriate way."""
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
        left, right = preprocess_binary(left, right)

        domain = get_binary_children_domains(left.domain, right.domain)
        auxiliary_domains = self.get_children_auxiliary_domains([left, right])
        super().__init__(
            name,
            children=[left, right],
            domain=domain,
            auxiliary_domains=auxiliary_domains,
        )
        self.left = self.children[0]
        self.right = self.children[1]

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        # Possibly add brackets for clarity
        if isinstance(self.left, pybamm.BinaryOperator) and not (
            (self.left.name == self.name)
            or (self.left.name == "*" and self.name == "/")
            or (self.left.name == "+" and self.name == "-")
            or self.name == "+"
        ):
            left_str = "({!s})".format(self.left)
        else:
            left_str = "{!s}".format(self.left)
        if isinstance(self.right, pybamm.BinaryOperator) and not (
            (self.name == "*" and self.right.name in ["*", "/"]) or self.name == "+"
        ):
            right_str = "({!s})".format(self.right)
        else:
            right_str = "{!s}".format(self.right)
        return "{} {} {}".format(left_str, self.name, right_str)

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
        """
        Default behaviour for new_copy.
        This copies the behaviour of `_binary_evaluate`, but since `left` and `right`
        are symbols creates a new symbol instead of returning a value.
        """
        return self._binary_evaluate(left, right)

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

    def _binary_evaluate(self, left, right):
        """ Perform binary operation on nodes 'left' and 'right'. """
        raise NotImplementedError

    def _evaluates_on_edges(self, dimension):
        """ See :meth:`pybamm.Symbol._evaluates_on_edges()`. """
        return self.left.evaluates_on_edges(dimension) or self.right.evaluates_on_edges(
            dimension
        )

    def is_constant(self):
        """ See :meth:`pybamm.Symbol.is_constant()`. """
        return self.left.is_constant() and self.right.is_constant()


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
        if right.evaluates_to_constant_number():
            return (right * left ** (right - 1)) * left_jac
        elif left.evaluates_to_constant_number():
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
        if left.evaluates_to_constant_number():
            return left * right_jac
        elif right.evaluates_to_constant_number():
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
        if left.evaluates_to_constant_number():
            return -left / right ** 2 * right_jac
        elif right.evaluates_to_constant_number():
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
        if left.evaluates_to_constant_number():
            return left * right_jac
        elif right.evaluates_to_constant_number():
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

    def _binary_new_copy(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_new_copy()`. """
        return pybamm.inner(left, right)

    def _evaluates_on_edges(self, dimension):
        """ See :meth:`pybamm.Symbol._evaluates_on_edges()`. """
        return False


def inner(left, right):
    """Return inner product of two symbols."""
    left, right = preprocess_binary(left, right)
    # simplify multiply by scalar zero, being careful about shape
    if pybamm.is_scalar_zero(left):
        return pybamm.zeros_like(right)
    if pybamm.is_scalar_zero(right):
        return pybamm.zeros_like(left)

    # if one of the children is a zero matrix, we have to be careful about shapes
    if pybamm.is_matrix_zero(left) or pybamm.is_matrix_zero(right):
        return pybamm.zeros_like(pybamm.Inner(left, right))

    # anything multiplied by a scalar one returns itself
    if pybamm.is_scalar_one(left):
        return right
    if pybamm.is_scalar_one(right):
        return left

    return pybamm.simplify_if_constant(pybamm.Inner(left, right))


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
    """A heaviside function with equality (return 1 when left = right)"""

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
    """A heaviside function without equality (return 0 when left = right)"""

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


class Modulo(BinaryOperator):
    """Calculates the remainder of an integer division."""

    def __init__(self, left, right):
        super().__init__("%", left, right)

    def _diff(self, variable):
        """ See :meth:`pybamm.Symbol._diff()`. """
        # apply chain rule and power rule
        left, right = self.orphans
        # derivative if variable is in the base
        diff = left.diff(variable)
        # derivative if variable is in the right term (rare, check separately to avoid
        # unecessarily big tree)
        if any(variable.id == x.id for x in right.pre_order()):
            diff += -pybamm.Floor(left / right) * right.diff(variable)
        return diff

    def _binary_jac(self, left_jac, right_jac):
        """ See :meth:`pybamm.BinaryOperator._binary_jac()`. """
        # apply chain rule and power rule
        left, right = self.orphans
        if right.evaluates_to_constant_number():
            return left_jac
        elif left.evaluates_to_constant_number():
            return -right_jac * pybamm.Floor(left / right)
        else:
            return left_jac - right_jac * pybamm.Floor(left / right)

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        return "{!s} mod {!s}".format(self.left, self.right)

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """
        return left % right


class Minimum(BinaryOperator):
    """Returns the smaller of two objects."""

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

    def _binary_new_copy(self, left, right):
        "See :meth:`pybamm.BinaryOperator._binary_new_copy()`. "
        return pybamm.minimum(left, right)


class Maximum(BinaryOperator):
    """Returns the smaller of two objects."""

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

    def _binary_new_copy(self, left, right):
        "See :meth:`pybamm.BinaryOperator._binary_new_copy()`. "
        return pybamm.maximum(left, right)


def simplify_elementwise_binary_broadcasts(left, right):
    left, right = preprocess_binary(left, right)

    # No need to broadcast if the other symbol already has the shape that is being
    # broadcasted to
    if left.domains == right.domains and all(
        left.evaluates_on_edges(dim) == right.evaluates_on_edges(dim)
        for dim in ["primary", "secondary", "tertiary"]
    ):
        if isinstance(left, pybamm.Broadcast) and left.child.domain == []:
            left = left.orphans[0]
        elif isinstance(right, pybamm.Broadcast) and right.child.domain == []:
            right = right.orphans[0]

    return left, right


def simplified_power(left, right):
    left, right = simplify_elementwise_binary_broadcasts(left, right)

    # Broadcast commutes with power operator
    if isinstance(left, pybamm.Broadcast) and right.domain == []:
        return left._unary_new_copy(left.orphans[0] ** right)
    elif isinstance(right, pybamm.Broadcast) and left.domain == []:
        return right._unary_new_copy(left ** right.orphans[0])

    # anything to the power of zero is one
    if pybamm.is_scalar_zero(right):
        return pybamm.ones_like(left)

    # zero to the power of anything is zero
    if pybamm.is_scalar_zero(left):
        return pybamm.Scalar(0)

    # anything to the power of one is itself
    if pybamm.is_scalar_one(right):
        return left

    if isinstance(left, Multiplication):
        # Simplify (a * b) ** c to (a ** c) * (b ** c)
        # if (a ** c) is constant or (b ** c) is constant
        if left.left.is_constant() or left.right.is_constant():
            l_left, l_right = left.orphans
            new_left = l_left ** right
            new_right = l_right ** right
            if new_left.is_constant() or new_right.is_constant():
                return new_left * new_right
    elif isinstance(left, Division):
        # Simplify (a / b) ** c to (a ** c) / (b ** c)
        # if (a ** c) is constant or (b ** c) is constant
        if left.left.is_constant() or left.right.is_constant():
            l_left, l_right = left.orphans
            new_left = l_left ** right
            new_right = l_right ** right
            if new_left.is_constant() or new_right.is_constant():
                return new_left / new_right

    return pybamm.simplify_if_constant(pybamm.Power(left, right))


def simplified_addition(left, right):
    """
    Note
    ----
    We check for scalars first, then matrices. This is because
    (Zero Matrix) + (Zero Scalar)
    should return (Zero Matrix), not (Zero Scalar).
    """
    left, right = simplify_elementwise_binary_broadcasts(left, right)

    # Broadcast commutes with addition operator
    if isinstance(left, pybamm.Broadcast) and right.domain == []:
        return left._unary_new_copy(left.orphans[0] + right)
    elif isinstance(right, pybamm.Broadcast) and left.domain == []:
        return right._unary_new_copy(left + right.orphans[0])

    # anything added by a scalar zero returns the other child
    elif pybamm.is_scalar_zero(left):
        return right
    elif pybamm.is_scalar_zero(right):
        return left
    # Check matrices after checking scalars
    elif pybamm.is_matrix_zero(left):
        if right.evaluates_to_number():
            return right * pybamm.ones_like(left)
        # If left object is zero and has size smaller than or equal to right object in
        # all dimensions, we can safely return the right object. For example, adding a
        # zero vector a matrix, we can just return the matrix
        elif all(
            left_dim_size <= right_dim_size
            for left_dim_size, right_dim_size in zip(
                left.shape_for_testing, right.shape_for_testing
            )
        ) and all(
            left.evaluates_on_edges(dim) == right.evaluates_on_edges(dim)
            for dim in ["primary", "secondary", "tertiary"]
        ):
            return right
    elif pybamm.is_matrix_zero(right):
        if left.evaluates_to_number():
            return left * pybamm.ones_like(right)
        # See comment above
        elif all(
            left_dim_size >= right_dim_size
            for left_dim_size, right_dim_size in zip(
                left.shape_for_testing, right.shape_for_testing
            )
        ) and all(
            left.evaluates_on_edges(dim) == right.evaluates_on_edges(dim)
            for dim in ["primary", "secondary", "tertiary"]
        ):
            return left

    # Simplify A @ c + B @ c to (A + B) @ c if (A + B) is constant
    # This is a common construction that appears from discretisation of spatial
    # operators
    elif (
        isinstance(left, MatrixMultiplication)
        and isinstance(right, MatrixMultiplication)
        and left.right.id == right.right.id
    ):
        l_left, l_right = left.orphans
        r_left = right.orphans[0]
        new_left = l_left + r_left
        if new_left.is_constant():
            new_sum = new_left @ l_right
            new_sum.copy_domains(pybamm.Addition(left, right))
            return new_sum

    return pybamm.simplify_if_constant(pybamm.Addition(left, right))


def simplified_subtraction(left, right):
    """
     Note
    ----
    We check for scalars first, then matrices. This is because
    (Zero Matrix) - (Zero Scalar)
    should return (Zero Matrix), not -(Zero Scalar).
    """
    left, right = simplify_elementwise_binary_broadcasts(left, right)

    # Broadcast commutes with subtraction operator
    if isinstance(left, pybamm.Broadcast) and right.domain == []:
        return left._unary_new_copy(left.orphans[0] - right)
    elif isinstance(right, pybamm.Broadcast) and left.domain == []:
        return right._unary_new_copy(left - right.orphans[0])

    # anything added by a scalar zero returns the other child
    if pybamm.is_scalar_zero(left):
        return -right
    if pybamm.is_scalar_zero(right):
        return left
    # Check matrices after checking scalars
    if pybamm.is_matrix_zero(left):
        if right.evaluates_to_number():
            return -right * pybamm.ones_like(left)
        # See comments in simplified_addition
        elif all(
            left_dim_size <= right_dim_size
            for left_dim_size, right_dim_size in zip(
                left.shape_for_testing, right.shape_for_testing
            )
        ) and all(
            left.evaluates_on_edges(dim) == right.evaluates_on_edges(dim)
            for dim in ["primary", "secondary", "tertiary"]
        ):
            return -right
    if pybamm.is_matrix_zero(right):
        if left.evaluates_to_number():
            return left * pybamm.ones_like(right)
        # See comments in simplified_addition
        elif all(
            left_dim_size >= right_dim_size
            for left_dim_size, right_dim_size in zip(
                left.shape_for_testing, right.shape_for_testing
            )
        ) and all(
            left.evaluates_on_edges(dim) == right.evaluates_on_edges(dim)
            for dim in ["primary", "secondary", "tertiary"]
        ):
            return left

    # a symbol minus itself is 0s of the same shape
    if left.id == right.id:
        return pybamm.zeros_like(left)

    return pybamm.simplify_if_constant(pybamm.Subtraction(left, right))


def simplified_multiplication(left, right):
    left, right = simplify_elementwise_binary_broadcasts(left, right)

    # Broadcast commutes with multiplication operator
    if isinstance(left, pybamm.Broadcast) and right.domain == []:
        return left._unary_new_copy(left.orphans[0] * right)
    elif isinstance(right, pybamm.Broadcast) and left.domain == []:
        return right._unary_new_copy(left * right.orphans[0])

    # simplify multiply by scalar zero, being careful about shape
    if pybamm.is_scalar_zero(left):
        return pybamm.zeros_like(right)
    if pybamm.is_scalar_zero(right):
        return pybamm.zeros_like(left)

    # if one of the children is a zero matrix, we have to be careful about shapes
    if pybamm.is_matrix_zero(left) or pybamm.is_matrix_zero(right):
        return pybamm.zeros_like(pybamm.Multiplication(left, right))

    # anything multiplied by a scalar one returns itself
    if pybamm.is_scalar_one(left):
        return right
    if pybamm.is_scalar_one(right):
        return left

    # anything multiplied by a matrix one returns itself if
    # - the shapes are the same
    # - both left and right evaluate on edges, or both evaluate on nodes, in all
    # dimensions
    # (and possibly more generally, but not implemented here)
    try:
        if left.shape_for_testing == right.shape_for_testing and all(
            left.evaluates_on_edges(dim) == right.evaluates_on_edges(dim)
            for dim in ["primary", "secondary", "tertiary"]
        ):
            if pybamm.is_matrix_one(left):
                return right
            elif pybamm.is_matrix_one(right):
                return left
    except NotImplementedError:
        pass

    # Return constant if both sides are constant
    if left.is_constant() and right.is_constant():
        return pybamm.simplify_if_constant(pybamm.Multiplication(left, right))

    # Simplify (B @ c) * a to (a * B) @ c if (a * B) is constant
    # This is a common construction that appears from discretisation of spatial
    # operators
    if (
        isinstance(left, MatrixMultiplication)
        and right.is_constant()
        and left.left.is_constant()
    ):
        l_left, l_right = left.orphans
        new_left = right * l_left
        # be careful about domains to avoid weird errors
        new_left.clear_domains()
        new_mul = new_left @ l_right
        # Keep the domain of the old left
        new_mul.copy_domains(left)
        return new_mul

    elif isinstance(left, Multiplication) and right.is_constant():
        # Simplify (a * b) * c to (a * c) * b if (a * c) is constant
        if left.left.is_constant():
            l_left, l_right = left.orphans
            new_left = l_left * right
            return new_left * l_right
        # Simplify (a * b) * c to a * (b * c) if (b * c) is constant
        elif left.right.is_constant():
            l_left, l_right = left.orphans
            new_right = l_right * right
            return l_left * new_right
    elif isinstance(left, Division) and right.is_constant():
        # Simplify (a / b) * c to a * (c / b) if (c / b) is constant
        if left.right.is_constant():
            l_left, l_right = left.orphans
            new_right = right / l_right
            return l_left * new_right

    # Simplify a * (B @ c) to (a * B) @ c if (a * B) is constant
    if (
        isinstance(right, MatrixMultiplication)
        and left.is_constant()
        and right.left.is_constant()
    ):
        r_left, r_right = right.orphans
        new_left = left * r_left
        # be careful about domains to avoid weird errors
        new_left.clear_domains()
        new_mul = new_left @ r_right
        # Keep the domain of the old right
        new_mul.copy_domains(right)
        return new_mul

    elif isinstance(right, Multiplication) and left.is_constant():
        # Simplify a * (b * c) to (a * b) * c if (a * b) is constant
        if right.left.is_constant():
            r_left, r_right = right.orphans
            new_left = left * r_left
            return new_left * r_right
        # Simplify a * (b * c) to (a * c) * b if (a * c) is constant
        elif right.right.is_constant():
            r_left, r_right = right.orphans
            new_left = left * r_right
            return new_left * r_left
    elif isinstance(right, Division) and left.is_constant():
        # Simplify a * (b / c) to (a / c) * b if (a / c) is constant
        if right.right.is_constant():
            r_left, r_right = right.orphans
            new_left = left / r_right
            return new_left * r_left

    return pybamm.Multiplication(left, right)


def simplified_division(left, right):
    left, right = simplify_elementwise_binary_broadcasts(left, right)

    # Broadcast commutes with division operator
    if isinstance(left, pybamm.Broadcast) and right.domain == []:
        return left._unary_new_copy(left.orphans[0] / right)
    elif isinstance(right, pybamm.Broadcast) and left.domain == []:
        return right._unary_new_copy(left / right.orphans[0])

    # zero divided by anything returns zero (being careful about shape)
    if pybamm.is_scalar_zero(left):
        return pybamm.zeros_like(right)

    # matrix zero divided by anything returns matrix zero (i.e. itself)
    if pybamm.is_matrix_zero(left):
        return pybamm.zeros_like(pybamm.Division(left, right))

    # anything divided by zero raises error
    if pybamm.is_scalar_zero(right):
        raise ZeroDivisionError

    # anything divided by one is itself
    if pybamm.is_scalar_one(right):
        return left

    # a symbol divided by itself is 1s of the same shape
    if left.id == right.id:
        return pybamm.ones_like(left)

    # Simplify (B @ c) / a to (B / a) @ c if (B / a) is constant
    # This is a common construction that appears from discretisation of averages
    elif isinstance(left, MatrixMultiplication) and right.is_constant():
        l_left, l_right = left.orphans
        new_left = l_left / right
        if new_left.is_constant():
            # be careful about domains to avoid weird errors
            new_left.clear_domains()
            new_division = new_left @ l_right
            # Keep the domain of the old left
            new_division.copy_domains(left)
            return new_division

    if isinstance(left, Multiplication):
        # Simplify (a * b) / c to (a / c) * b if (a / c) is constant
        if left.left.is_constant():
            l_left, l_right = left.orphans
            new_left = l_left / right
            if new_left.is_constant():
                return new_left * l_right
        # Simplify (a * b) / c to a * (b / c) if (b / c) is constant
        elif left.right.is_constant():
            l_left, l_right = left.orphans
            new_right = l_right / right
            if new_right.is_constant():
                return l_left * new_right

    return pybamm.simplify_if_constant(pybamm.Division(left, right))


def simplified_matrix_multiplication(left, right):
    left, right = preprocess_binary(left, right)
    if pybamm.is_matrix_zero(left) or pybamm.is_matrix_zero(right):
        return pybamm.zeros_like(pybamm.MatrixMultiplication(left, right))

    if isinstance(right, Multiplication) and left.is_constant():
        # Simplify A @ (b * c) to (A * b) @ c if (A * b) is constant
        if right.left.evaluates_to_constant_number():
            r_left, r_right = right.orphans
            new_left = left * r_left
            return new_left @ r_right
        # Simplify A @ (b * c) to (A * c) @ b if (A * c) is constant
        elif right.right.evaluates_to_constant_number():
            r_left, r_right = right.orphans
            new_left = left * r_right
            return new_left @ r_left
    elif isinstance(right, Division) and left.is_constant():
        # Simplify A @ (b / c) to (A / c) @ b if (A / c) is constant
        if right.right.evaluates_to_constant_number():
            r_left, r_right = right.orphans
            new_left = left / r_right
            return new_left @ r_left

    # Simplify A @ (B @ c) to (A @ B) @ c if (A @ B) is constant
    # This is a common construction that appears from discretisation of spatial
    # operators
    if (
        isinstance(right, MatrixMultiplication)
        and right.left.is_constant()
        and left.is_constant()
    ):
        r_left, r_right = right.orphans
        new_left = left @ r_left
        # be careful about domains to avoid weird errors
        new_left.clear_domains()
        new_mul = new_left @ r_right
        # Keep the domain of the old right
        new_mul.copy_domains(right)
        return new_mul

    # Simplify A @ (b + c) to (A @ b) + (A @ c) if (A @ b) or (A @ c) is constant
    # This is a common construction that appears from discretisation of spatial
    # operators
    # Don't do this if either b or c is a number as this will lead to matmul errors
    elif isinstance(right, Addition):
        if (right.left.is_constant() or right.right.is_constant()) and not (
            right.left.size_for_testing == 1 or right.right.size_for_testing == 1
        ):
            r_left, r_right = right.orphans
            return (left @ r_left) + (left @ r_right)

    return pybamm.simplify_if_constant(pybamm.MatrixMultiplication(left, right))


def minimum(left, right):
    """
    Returns the smaller of two objects, possibly with a smoothing approximation.
    Not to be confused with :meth:`pybamm.min`, which returns min function of child.
    """
    k = pybamm.settings.min_smoothing
    # Return exact approximation if that is the setting or the outcome is a constant
    # (i.e. no need for smoothing)
    if k == "exact" or (pybamm.is_constant(left) and pybamm.is_constant(right)):
        out = Minimum(left, right)
    else:
        out = pybamm.softminus(left, right, k)
    return pybamm.simplify_if_constant(out)


def maximum(left, right):
    """
    Returns the larger of two objects, possibly with a smoothing approximation.
    Not to be confused with :meth:`pybamm.max`, which returns max function of child.
    """
    k = pybamm.settings.max_smoothing
    # Return exact approximation if that is the setting or the outcome is a constant
    # (i.e. no need for smoothing)
    if k == "exact" or (pybamm.is_constant(left) and pybamm.is_constant(right)):
        out = Maximum(left, right)
    else:
        out = pybamm.softplus(left, right, k)
    return pybamm.simplify_if_constant(out)


def softminus(left, right, k):
    """
    Softplus approximation to the minimum function. k is the smoothing parameter,
    set by `pybamm.settings.min_smoothing`. The recommended value is k=10.
    """
    return pybamm.log(pybamm.exp(-k * left) + pybamm.exp(-k * right)) / -k


def softplus(left, right, k):
    """
    Softplus approximation to the maximum function. k is the smoothing parameter,
    set by `pybamm.settings.max_smoothing`. The recommended value is k=10.
    """
    return pybamm.log(pybamm.exp(k * left) + pybamm.exp(k * right)) / k


def sigmoid(left, right, k):
    """
    Sigmoidal approximation to the heaviside function. k is the smoothing parameter,
    set by `pybamm.settings.heaviside_smoothing`. The recommended value is k=10.
    Note that the concept of deciding which side to pick when left=right does not apply
    for this smooth approximation. When left=right, the value is (left+right)/2.
    """
    return (1 + pybamm.tanh(k * (right - left))) / 2


def source(left, right, boundary=False):
    """A convenience function for creating (part of) an expression tree representing
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
