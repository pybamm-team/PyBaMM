#
# Binary operator classes
#
from __future__ import annotations
import numbers

import numpy as np
import sympy
from scipy.sparse import csr_matrix, issparse
import functools

import pybamm

from typing import Callable, cast

# create type alias(s)
from pybamm.type_definitions import ChildSymbol, ChildValue, Numeric


def _preprocess_binary(
    left: ChildSymbol, right: ChildSymbol
) -> tuple[pybamm.Symbol, pybamm.Symbol]:
    if isinstance(left, (float, int, np.number)):
        left = pybamm.Scalar(left)
    elif isinstance(left, np.ndarray):
        if left.ndim > 1:
            raise ValueError("left must be a 1D array")
        left = pybamm.Vector(left)
    if isinstance(right, (float, int, np.number)):
        right = pybamm.Scalar(right)
    elif isinstance(right, np.ndarray):
        if right.ndim > 1:
            raise ValueError("right must be a 1D array")
        right = pybamm.Vector(right)

    # Check both left and right are pybamm Symbols
    if not (isinstance(left, pybamm.Symbol) and isinstance(right, pybamm.Symbol)):
        raise NotImplementedError(
            f"BinaryOperator not implemented for symbols of type {type(left)} and {type(right)}"
        )

    # Do some broadcasting in special cases, to avoid having to do this manually
    if left.domain != [] and right.domain != [] and left.domain != right.domain:
        if left.domain == right.secondary_domain:
            left = pybamm.PrimaryBroadcast(left, right.domain)
        elif right.domain == left.secondary_domain:
            right = pybamm.PrimaryBroadcast(right, left.domain)

    return left, right


class BinaryOperator(pybamm.Symbol):
    """
    A node in the expression tree representing a binary operator (e.g. `+`, `*`)

    Derived classes will specify the particular operator

    Parameters
    ----------

    name : str
        name of the node
    left : :class:`Symbol` or :class:`Number`
        lhs child node (converted to :class:`Scalar` if Number)
    right : :class:`Symbol` or :class:`Number`
        rhs child node (converted to :class:`Scalar` if Number)
    """

    def __init__(
        self, name: str, left_child: ChildSymbol, right_child: ChildSymbol
    ) -> None:
        left, right = _preprocess_binary(left_child, right_child)

        domains = self.get_children_domains([left, right])
        super().__init__(name, children=[left, right], domains=domains)
        self.left = self.children[0]
        self.right = self.children[1]

    @classmethod
    def _from_json(cls, snippet: dict):
        """Use to instantiate when deserialising; discretisation has
        already occured so pre-processing of binaries is not necessary."""

        instance = cls.__new__(cls)

        super(BinaryOperator, instance).__init__(
            snippet["name"],
            children=[snippet["children"][0], snippet["children"][1]],
            domains=snippet["domains"],
        )
        instance.left = instance.children[0]
        instance.right = instance.children[1]

        return instance

    def __str__(self):
        """See :meth:`pybamm.Symbol.__str__()`."""
        # Possibly add brackets for clarity
        if isinstance(self.left, pybamm.BinaryOperator) and not (
            (self.left.name == self.name)
            or (self.left.name == "*" and self.name == "/")
            or (self.left.name == "+" and self.name == "-")
            or self.name == "+"
        ):
            left_str = f"({self.left!s})"
        else:
            left_str = f"{self.left!s}"
        if isinstance(self.right, pybamm.BinaryOperator) and not (
            (self.name == "*" and self.right.name in ["*", "/"]) or self.name == "+"
        ):
            right_str = f"({self.right!s})"
        else:
            right_str = f"{self.right!s}"
        return f"{left_str} {self.name} {right_str}"

    def create_copy(
        self,
        new_children: list[pybamm.Symbol] | None = None,
        perform_simplifications: bool = True,
    ):
        """See :meth:`pybamm.Symbol.new_copy()`."""

        if new_children and len(new_children) != 2:
            raise ValueError(
                f"Symbol of type {type(self)} must have exactly two children."
            )
        children = self._children_for_copying(new_children)

        if not perform_simplifications:
            out = self.__class__(children[0], children[1])
        else:
            # creates a new instance using the overloaded binary operator to perform
            # additional simplifications, rather than just calling the constructor
            out = self._binary_new_copy(children[0], children[1])

        out.copy_domains(self)

        return out

    def _binary_new_copy(self, left: ChildSymbol, right: ChildSymbol):
        """
        Performs the overloaded binary operation on the two symbols `left` and `right`,
        to create a binary class instance after performing appropriate simplifying
        checks.

        Default behaviour for _binary_new_copy copies the behaviour of `_binary_evaluate`,
        but since `left` and `right` are symbols this creates a new symbol instead of
        returning a value.
        """
        return self._binary_evaluate(left, right)

    def evaluate(
        self,
        t: float | None = None,
        y: np.ndarray | None = None,
        y_dot: np.ndarray | None = None,
        inputs: dict | str | None = None,
    ):
        """See :meth:`pybamm.Symbol.evaluate()`."""
        left = self.left.evaluate(t, y, y_dot, inputs)
        right = self.right.evaluate(t, y, y_dot, inputs)
        return self._binary_evaluate(left, right)

    def _evaluate_for_shape(self):
        """See :meth:`pybamm.Symbol.evaluate_for_shape()`."""
        left = self.children[0].evaluate_for_shape()
        right = self.children[1].evaluate_for_shape()
        return self._binary_evaluate(left, right)

    def _binary_jac(self, left_jac, right_jac):
        """Calculate the Jacobian of a binary operator."""
        raise NotImplementedError

    def _binary_evaluate(self, left, right):
        """Perform binary operation on nodes 'left' and 'right'."""
        raise NotImplementedError(
            f"{self.__class__} does not implement _binary_evaluate."
        )

    def _evaluates_on_edges(self, dimension: str) -> bool:
        """See :meth:`pybamm.Symbol._evaluates_on_edges()`."""
        return self.left.evaluates_on_edges(dimension) or self.right.evaluates_on_edges(
            dimension
        )

    def is_constant(self):
        """See :meth:`pybamm.Symbol.is_constant()`."""
        return self.left.is_constant() and self.right.is_constant()

    def _sympy_operator(self, left, right):
        """Apply appropriate SymPy operators."""
        return self._binary_evaluate(left, right)

    def to_equation(self):
        """Convert the node and its subtree into a SymPy equation."""
        if self.print_name is not None:
            return sympy.Symbol(self.print_name)
        else:
            child1, child2 = self.children
            eq1 = child1.to_equation()
            eq2 = child2.to_equation()
            return self._sympy_operator(eq1, eq2)

    def to_json(self):
        """
        Method to serialise a BinaryOperator object into JSON.
        """

        json_dict = {"name": self.name, "id": self.id, "domains": self.domains}

        return json_dict


class Power(BinaryOperator):
    """
    A node in the expression tree representing a `**` power operator.
    """

    def __init__(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        """See :meth:`pybamm.BinaryOperator.__init__()`."""
        super().__init__("**", left, right)

    def _diff(self, variable: pybamm.Symbol):
        """See :meth:`pybamm.Symbol._diff()`."""
        # apply chain rule and power rule
        base, exponent = self.orphans
        # derivative if variable is in the base
        diff = exponent * (base ** (exponent - 1)) * base.diff(variable)
        # derivative if variable is in the exponent (rare, check separately to avoid
        # unecessarily big tree)
        if any(variable == x for x in exponent.pre_order()):
            diff += (base**exponent) * pybamm.log(base) * exponent.diff(variable)
        return diff

    def _binary_jac(self, left_jac, right_jac):
        """See :meth:`pybamm.BinaryOperator._binary_jac()`."""
        # apply chain rule and power rule
        left, right = self.orphans
        if right.evaluates_to_constant_number():
            return (right * left ** (right - 1)) * left_jac
        elif left.evaluates_to_constant_number():
            return (left**right * pybamm.log(left)) * right_jac
        else:
            return (left ** (right - 1)) * (
                right * left_jac + left * pybamm.log(left) * right_jac
            )

    def _binary_evaluate(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        """See :meth:`pybamm.BinaryOperator._binary_evaluate()`."""
        # don't raise RuntimeWarning for NaNs
        with np.errstate(invalid="ignore"):
            return left**right


class Addition(BinaryOperator):
    """
    A node in the expression tree representing an addition operator.
    """

    def __init__(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        """See :meth:`pybamm.BinaryOperator.__init__()`."""
        super().__init__("+", left, right)

    def _diff(self, variable: pybamm.Symbol):
        """See :meth:`pybamm.Symbol._diff()`."""
        return self.left.diff(variable) + self.right.diff(variable)

    def _binary_jac(self, left_jac: ChildValue, right_jac: ChildValue):
        """See :meth:`pybamm.BinaryOperator._binary_jac()`."""
        return left_jac + right_jac

    def _binary_evaluate(self, left: ChildValue, right: ChildValue):
        """See :meth:`pybamm.BinaryOperator._binary_evaluate()`."""
        return left + right


class Subtraction(BinaryOperator):
    """
    A node in the expression tree representing a subtraction operator.
    """

    def __init__(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        """See :meth:`pybamm.BinaryOperator.__init__()`."""

        super().__init__("-", left, right)

    def _diff(self, variable: pybamm.Symbol):
        """See :meth:`pybamm.Symbol._diff()`."""
        return self.left.diff(variable) - self.right.diff(variable)

    def _binary_jac(self, left_jac, right_jac):
        """See :meth:`pybamm.BinaryOperator._binary_jac()`."""
        return left_jac - right_jac

    def _binary_evaluate(self, left: ChildValue, right: ChildValue):
        """See :meth:`pybamm.BinaryOperator._binary_evaluate()`."""
        return left - right


class Multiplication(BinaryOperator):
    """
    A node in the expression tree representing a multiplication operator
    (Hadamard product). Overloads cases where the "*" operator would usually return a
    matrix multiplication (e.g. scipy.sparse.coo.coo_matrix)
    """

    def __init__(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        """See :meth:`pybamm.BinaryOperator.__init__()`."""

        super().__init__("*", left, right)

    def _diff(self, variable: pybamm.Symbol):
        """See :meth:`pybamm.Symbol._diff()`."""
        # apply product rule
        left, right = self.orphans
        return left.diff(variable) * right + left * right.diff(variable)

    def _binary_jac(self, left_jac, right_jac):
        """See :meth:`pybamm.BinaryOperator._binary_jac()`."""
        # apply product rule
        left, right = self.orphans
        if left.evaluates_to_constant_number():
            return left * right_jac
        else:
            return right * left_jac + left * right_jac

    def _binary_evaluate(self, left, right):
        """See :meth:`pybamm.BinaryOperator._binary_evaluate()`."""

        if issparse(left):
            return csr_matrix(left.multiply(right))
        elif issparse(right):
            # Hadamard product is commutative, so we can switch right and left
            return csr_matrix(right.multiply(left))
        else:
            return left * right


class MatrixMultiplication(BinaryOperator):
    """
    A node in the expression tree representing a matrix multiplication operator.
    """

    def __init__(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        """See :meth:`pybamm.BinaryOperator.__init__()`."""
        super().__init__("@", left, right)

    def diff(self, variable):
        """See :meth:`pybamm.Symbol.diff()`."""
        # We shouldn't need this
        raise NotImplementedError(
            "diff not implemented for symbol of type 'MatrixMultiplication'"
        )

    def _binary_jac(self, left_jac, right_jac):
        """See :meth:`pybamm.BinaryOperator._binary_jac()`."""
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
                f"jac of 'MatrixMultiplication' is only "
                "implemented for left of type 'pybamm.Array', "
                f"not {left.__class__}"
            )

    def _binary_evaluate(self, left, right):
        """See :meth:`pybamm.BinaryOperator._binary_evaluate()`."""
        return left @ right

    def _sympy_operator(self, left, right):
        """Override :meth:`pybamm.BinaryOperator._sympy_operator`"""
        left = sympy.Matrix(left)
        right = sympy.Matrix(right)
        return left * right


class Division(BinaryOperator):
    """
    A node in the expression tree representing a division operator.
    """

    def __init__(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        """See :meth:`pybamm.BinaryOperator.__init__()`."""
        super().__init__("/", left, right)

    def _diff(self, variable: pybamm.Symbol):
        """See :meth:`pybamm.Symbol._diff()`."""
        # apply quotient rule
        top, bottom = self.orphans
        return (top.diff(variable) * bottom - top * bottom.diff(variable)) / bottom**2

    def _binary_jac(self, left_jac, right_jac):
        """See :meth:`pybamm.BinaryOperator._binary_jac()`."""
        # apply quotient rule
        left, right = self.orphans
        if left.evaluates_to_constant_number():
            return -left / right**2 * right_jac
        else:
            return (right * left_jac - left * right_jac) / right**2

    def _binary_evaluate(self, left, right):
        """See :meth:`pybamm.BinaryOperator._binary_evaluate()`."""

        if issparse(left):
            return csr_matrix(left.multiply(1 / right))
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
    """

    def __init__(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        """See :meth:`pybamm.BinaryOperator.__init__()`."""
        super().__init__("inner product", left, right)

    def _diff(self, variable: pybamm.Symbol):
        """See :meth:`pybamm.Symbol._diff()`."""
        # apply product rule
        left, right = self.orphans
        return left.diff(variable) * right + left * right.diff(variable)

    def _binary_jac(self, left_jac, right_jac):
        """See :meth:`pybamm.BinaryOperator._binary_jac()`."""
        # apply product rule
        left, right = self.orphans
        if left.evaluates_to_constant_number():
            return left * right_jac
        elif right.evaluates_to_constant_number():
            return right * left_jac
        else:
            return right * left_jac + left * right_jac

    def _binary_evaluate(self, left, right):
        """See :meth:`pybamm.BinaryOperator._binary_evaluate()`."""

        if issparse(left):
            return left.multiply(right)
        elif issparse(right):
            # Hadamard product is commutative, so we can switch right and left
            return right.multiply(left)
        else:
            return left * right

    def _binary_new_copy(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        """See :meth:`pybamm.BinaryOperator._binary_new_copy()`."""
        return pybamm.inner(left, right)

    def _evaluates_on_edges(self, dimension: str) -> bool:
        """See :meth:`pybamm.Symbol._evaluates_on_edges()`."""
        return False


def inner(left_child, right_child):
    """Return inner product of two symbols."""
    left, right = _preprocess_binary(left_child, right_child)
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


class Equality(BinaryOperator):
    """
    A node in the expression tree representing an equality comparison between two
    nodes. Returns 1 if the two nodes evaluate to the same thing and 0 otherwise.
    """

    def __init__(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        """See :meth:`pybamm.BinaryOperator.__init__()`."""
        super().__init__("==", left, right)

    def diff(self, variable):
        """See :meth:`pybamm.Symbol.diff()`."""
        # Equality should always be multiplied by something else so hopefully don't
        # need to worry about shape
        return pybamm.Scalar(0)

    def _binary_jac(self, left_jac, right_jac):
        """See :meth:`pybamm.BinaryOperator._binary_jac()`."""
        # Equality should always be multiplied by something else so hopefully don't
        # need to worry about shape
        return pybamm.Scalar(0)

    def _binary_evaluate(self, left, right):
        """See :meth:`pybamm.BinaryOperator._binary_evaluate()`."""
        # numpy 1.25 deprecation warning: extract value from numpy arrays
        if isinstance(right, np.ndarray):
            return int(left == right.item())
        else:
            return int(left == right)

    def _binary_new_copy(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        """
        Overwrites `pybamm.BinaryOperator._binary_new_copy()` to return a new instance of
        `pybamm.Equality` rather than using `binary_evaluate` to return a value.
        """
        return pybamm.Equality(left, right)


class _Heaviside(BinaryOperator):
    """
    A node in the expression tree representing a heaviside step function.
    This class is semi-private and should not be called directly, use `EqualHeaviside`
    or `NotEqualHeaviside` instead, or `<` or `<=`.

    Adding this operation to the rhs or algebraic equations in a model can often cause a
    discontinuity in the solution. For the specific cases listed below, this will be
    automatically handled by the solver. In the general case, you can explicitly tell
    the solver of discontinuities by adding a :class:`Event` object with
    :class:`EventType` DISCONTINUITY to the model's list of events.

    In the case where the Heaviside function is of the form `pybamm.t < x`, `pybamm.t <=
    x`, `x < pybamm.t`, or `x <= pybamm.t`, where `x` is any constant equation, this
    DISCONTINUITY event will automatically be added by the solver.
    """

    def __init__(
        self,
        name: str,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        """See :meth:`pybamm.BinaryOperator.__init__()`."""
        super().__init__(name, left, right)

    def diff(self, variable):
        """See :meth:`pybamm.Symbol.diff()`."""
        # Heaviside should always be multiplied by something else so hopefully don't
        # need to worry about shape
        return pybamm.Scalar(0)

    def _binary_jac(self, left_jac, right_jac):
        """See :meth:`pybamm.BinaryOperator._binary_jac()`."""
        # Heaviside should always be multiplied by something else so hopefully don't
        # need to worry about shape
        return pybamm.Scalar(0)

    def _evaluate_for_shape(self):
        """
        Returns an array of NaNs of the correct shape.
        See :meth:`pybamm.Symbol.evaluate_for_shape()`.
        """
        left = self.children[0].evaluate_for_shape()
        right = self.children[1].evaluate_for_shape()
        # _binary_evaluate will return an array of bools, so we multiply by NaN to get
        # an array of NaNs
        return self._binary_evaluate(left, right) * np.nan


class EqualHeaviside(_Heaviside):
    """A heaviside function with equality (return 1 when left = right)"""

    def __init__(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        """See :meth:`pybamm.BinaryOperator.__init__()`."""
        super().__init__("<=", left, right)

    def __str__(self):
        """See :meth:`pybamm.Symbol.__str__()`."""
        return f"{self.left!s} <= {self.right!s}"

    def _binary_evaluate(self, left, right):
        """See :meth:`pybamm.BinaryOperator._binary_evaluate()`."""
        # don't raise RuntimeWarning for NaNs
        with np.errstate(invalid="ignore"):
            return left <= right


class NotEqualHeaviside(_Heaviside):
    """A heaviside function without equality (return 0 when left = right)"""

    def __init__(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        super().__init__("<", left, right)

    def __str__(self):
        """See :meth:`pybamm.Symbol.__str__()`."""
        return f"{self.left!s} < {self.right!s}"

    def _binary_evaluate(self, left, right):
        """See :meth:`pybamm.BinaryOperator._binary_evaluate()`."""
        # don't raise RuntimeWarning for NaNs
        with np.errstate(invalid="ignore"):
            return left < right


class Modulo(BinaryOperator):
    """Calculates the remainder of an integer division."""

    def __init__(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        super().__init__("%", left, right)

    def _diff(self, variable: pybamm.Symbol):
        """See :meth:`pybamm.Symbol._diff()`."""
        # apply chain rule and power rule
        left, right = self.orphans
        # derivative if variable is in the base
        diff = left.diff(variable)
        # derivative if variable is in the right term (rare, check separately to avoid
        # unecessarily big tree)
        if any(variable == x for x in right.pre_order()):
            diff += -pybamm.Floor(left / right) * right.diff(variable)
        return diff

    def _binary_jac(self, left_jac, right_jac):
        """See :meth:`pybamm.BinaryOperator._binary_jac()`."""
        # apply chain rule and power rule
        left, right = self.orphans
        if right.evaluates_to_constant_number():
            return left_jac
        elif left.evaluates_to_constant_number():
            return -right_jac * pybamm.Floor(left / right)
        else:
            return left_jac - right_jac * pybamm.Floor(left / right)

    def __str__(self):
        """See :meth:`pybamm.Symbol.__str__()`."""
        return f"{self.left!s} mod {self.right!s}"

    def _binary_evaluate(self, left, right):
        """See :meth:`pybamm.BinaryOperator._binary_evaluate()`."""
        return left % right


class Minimum(BinaryOperator):
    """Returns the smaller of two objects."""

    def __init__(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        super().__init__("minimum", left, right)

    def __str__(self):
        """See :meth:`pybamm.Symbol.__str__()`."""
        return f"minimum({self.left!s}, {self.right!s})"

    def _diff(self, variable: pybamm.Symbol):
        """See :meth:`pybamm.Symbol._diff()`."""
        left, right = self.orphans
        return (left <= right) * left.diff(variable) + (left > right) * right.diff(
            variable
        )

    def _binary_jac(self, left_jac, right_jac):
        """See :meth:`pybamm.BinaryOperator._binary_jac()`."""
        left, right = self.orphans
        return (left <= right) * left_jac + (left > right) * right_jac

    def _binary_evaluate(self, left, right):
        """See :meth:`pybamm.BinaryOperator._binary_evaluate()`."""
        # don't raise RuntimeWarning for NaNs
        return np.minimum(left, right)

    def _binary_new_copy(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        """See :meth:`pybamm.BinaryOperator._binary_new_copy()`."""
        return pybamm.minimum(left, right)

    def _sympy_operator(self, left, right):
        """Override :meth:`pybamm.BinaryOperator._sympy_operator`"""
        return sympy.Min(left, right)


class Maximum(BinaryOperator):
    """Returns the greater of two objects."""

    def __init__(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        super().__init__("maximum", left, right)

    def __str__(self):
        """See :meth:`pybamm.Symbol.__str__()`."""
        return f"maximum({self.left!s}, {self.right!s})"

    def _diff(self, variable: pybamm.Symbol):
        """See :meth:`pybamm.Symbol._diff()`."""
        left, right = self.orphans
        return (left >= right) * left.diff(variable) + (left < right) * right.diff(
            variable
        )

    def _binary_jac(self, left_jac, right_jac):
        """See :meth:`pybamm.BinaryOperator._binary_jac()`."""
        left, right = self.orphans
        return (left >= right) * left_jac + (left < right) * right_jac

    def _binary_evaluate(self, left, right):
        """See :meth:`pybamm.BinaryOperator._binary_evaluate()`."""
        # don't raise RuntimeWarning for NaNs
        return np.maximum(left, right)

    def _binary_new_copy(
        self,
        left: ChildSymbol,
        right: ChildSymbol,
    ):
        """See :meth:`pybamm.BinaryOperator._binary_new_copy()`."""
        return pybamm.maximum(left, right)

    def _sympy_operator(self, left, right):
        """Override :meth:`pybamm.BinaryOperator._sympy_operator`"""
        return sympy.Max(left, right)


def _simplify_elementwise_binary_broadcasts(
    left_child: ChildSymbol,
    right_child: ChildSymbol,
) -> tuple[pybamm.Symbol, pybamm.Symbol]:
    left, right = _preprocess_binary(left_child, right_child)

    def unpack_broadcast_recursive(symbol: pybamm.Symbol) -> pybamm.Symbol:
        if isinstance(symbol, pybamm.Broadcast):
            if symbol.child.domain == []:
                return symbol.orphans[0]
            elif (
                isinstance(symbol.child, pybamm.Broadcast)
                and symbol.child.broadcasts_to_nodes
            ):
                out = unpack_broadcast_recursive(symbol.orphans[0])
                if out.domain == []:
                    return out
        return symbol

    # No need to broadcast if the other symbol already has the shape that is being
    # broadcasted to
    # Do this recursively
    if left.domains == right.domains:
        if isinstance(left, pybamm.Broadcast) and left.broadcasts_to_nodes:
            left = unpack_broadcast_recursive(left)
        elif isinstance(right, pybamm.Broadcast) and right.broadcasts_to_nodes:
            right = unpack_broadcast_recursive(right)

    return left, right


def _simplified_binary_broadcast_concatenation(
    left: pybamm.Symbol,
    right: pybamm.Symbol,
    operator: Callable,
) -> pybamm.Broadcast | None:
    """
    Check if there are concatenations or broadcasts that we can commute the operator
    with
    """
    # Broadcast commutes with elementwise operators
    if isinstance(left, pybamm.Broadcast) and right.domain == []:
        return left.create_copy([operator(left.orphans[0], right)])
    elif isinstance(right, pybamm.Broadcast) and left.domain == []:
        return right.create_copy([operator(left, right.orphans[0])])

    # Concatenation commutes with elementwise operators
    # If one of the sides is constant then commute concatenation with the operator
    # Don't do this for ConcatenationVariable objects as these will
    # be simplified differently later on
    if isinstance(left, pybamm.Concatenation) and not isinstance(
        left, pybamm.ConcatenationVariable
    ):
        if right.evaluates_to_constant_number():
            return left.create_copy([operator(child, right) for child in left.orphans])
        elif isinstance(right, pybamm.Concatenation) and not isinstance(
            right, pybamm.ConcatenationVariable
        ):
            if len(left.orphans) == len(right.orphans):
                return left.create_copy(
                    [
                        operator(left_child, right_child)
                        for left_child, right_child in zip(left.orphans, right.orphans)
                    ]
                )
            else:
                raise AssertionError(
                    "Concatenations must have the same number of children"
                )
    if isinstance(right, pybamm.Concatenation) and not isinstance(
        right, pybamm.ConcatenationVariable
    ):
        if left.evaluates_to_constant_number():
            return right.create_copy([operator(left, child) for child in right.orphans])
    return None


def simplified_power(
    left: ChildSymbol,
    right: ChildSymbol,
):
    left, right = _simplify_elementwise_binary_broadcasts(left, right)

    # Check for Concatenations and Broadcasts
    out = _simplified_binary_broadcast_concatenation(left, right, simplified_power)
    if out is not None:
        return out

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
            new_left = l_left**right
            new_right = l_right**right
            if new_left.is_constant() or new_right.is_constant():
                return new_left * new_right
    elif isinstance(left, Division):
        # Simplify (a / b) ** c to (a ** c) / (b ** c)
        # if (a ** c) is constant or (b ** c) is constant
        if left.left.is_constant() or left.right.is_constant():
            l_left, l_right = left.orphans
            new_left = l_left**right
            new_right = l_right**right
            if new_left.is_constant() or new_right.is_constant():
                return new_left / new_right

    return pybamm.simplify_if_constant(pybamm.Power(left, right))


def add(left: ChildSymbol, right: ChildSymbol):
    """
    Note
    ----
    We check for scalars first, then matrices. This is because
    (Zero Matrix) + (Zero Scalar)
    should return (Zero Matrix), not (Zero Scalar).
    """
    left, right = _simplify_elementwise_binary_broadcasts(left, right)

    # Move constant to always be on the left
    if right.is_constant() and not left.is_constant():
        left, right = right, left

    # Check for Concatenations and Broadcasts
    out = _simplified_binary_broadcast_concatenation(left, right, add)
    if out is not None:
        return out

    # anything added by a scalar zero returns the other child
    if pybamm.is_scalar_zero(left):
        return right
    # Check matrices after checking scalars
    if pybamm.is_matrix_zero(left):
        if right.evaluates_to_number():
            return right * pybamm.ones_like(left)
        # If left object is zero and has size smaller than or equal to right object in
        # all dimensions, we can safely return the right object. For example, adding a
        # zero vector a matrix, we can just return the matrix.
        # When checking evaluation on edges, check dimensions of left object only
        elif all(
            left_dim_size <= right_dim_size
            for left_dim_size, right_dim_size in zip(
                left.shape_for_testing, right.shape_for_testing
            )
        ) and all(
            left.evaluates_on_edges(dim) == right.evaluates_on_edges(dim)
            for dim in left.domains.keys()
        ):
            return right

    # Return constant if both sides are constant
    if left.is_constant() and right.is_constant():
        return pybamm.simplify_if_constant(Addition(left, right))

    # Simplify A @ c + B @ c to (A + B) @ c if (A + B) is constant
    # This is a common construction that appears from discretisation of spatial
    # operators
    elif (
        isinstance(left, MatrixMultiplication)
        and isinstance(right, MatrixMultiplication)
        and left.right == right.right
    ):
        l_left, l_right = left.orphans
        r_left = right.orphans[0]
        new_left = l_left + r_left
        if new_left.is_constant():
            new_sum = new_left @ l_right
            new_sum.copy_domains(Addition(left, right))
            return new_sum

    # Turn a + (-b) into a - b
    if isinstance(right, pybamm.Negate):
        return left - right.orphans[0]
    # Turn (-a) + b into b - a
    # check for is_constant() to avoid infinite recursion
    if isinstance(left, pybamm.Negate) and not left.is_constant():
        return right - left.orphans[0]

    if left.is_constant():
        if isinstance(right, (Addition, Subtraction)) and right.left.is_constant():
            # Simplify a + (b +- c) to (a + b) +- c if (a + b) is constant
            r_left, r_right = right.orphans
            return right.create_copy([left + r_left, r_right])
    if isinstance(left, Subtraction):
        if right == left.right:
            # Simplify (a - b) + b to a
            # Make sure shape is preserved
            return left.left * pybamm.ones_like(left.right)
    if isinstance(right, Subtraction):
        if left == right.right:
            # Simplify a + (b - a) to b
            # Make sure shape is preserved
            return right.left * pybamm.ones_like(right.right)

    return pybamm.simplify_if_constant(Addition(left, right))


def subtract(
    left: ChildSymbol,
    right: ChildSymbol,
):
    """
    Note
    ----
    We check for scalars first, then matrices. This is because
    (Zero Matrix) - (Zero Scalar)
    should return (Zero Matrix), not -(Zero Scalar).
    """
    left, right = _simplify_elementwise_binary_broadcasts(left, right)

    # Move constant to always be on the left
    # For a subtraction, this means (var - constant) becomes (-constant + var)
    if right.is_constant() and not left.is_constant():
        return -right + left

    # Check for Concatenations and Broadcasts
    out = _simplified_binary_broadcast_concatenation(left, right, subtract)
    if out is not None:
        return out

    # anything added by a scalar zero returns the other child
    if pybamm.is_scalar_zero(left):
        return -right
    # Check matrices after checking scalars
    if pybamm.is_matrix_zero(left):
        if right.evaluates_to_number():
            return -right * pybamm.ones_like(left)
        # See comments in add
        elif all(
            left_dim_size <= right_dim_size
            for left_dim_size, right_dim_size in zip(
                left.shape_for_testing, right.shape_for_testing
            )
        ) and all(
            left.evaluates_on_edges(dim) == right.evaluates_on_edges(dim)
            for dim in left.domains.keys()
        ):
            return -right

    # Return constant if both sides are constant
    if left.is_constant() and right.is_constant():
        return pybamm.simplify_if_constant(Subtraction(left, right))

    # a symbol minus itself is 0s of the same shape
    if left == right:
        return pybamm.zeros_like(left)

    # Turn a - (-b) into a + b
    if isinstance(right, pybamm.Negate):
        return left + right.orphans[0]

    if left.is_constant():
        if isinstance(right, (Addition, Subtraction)) and right.left.is_constant():
            # Simplify a - (b +- c) to (a - b) -+ c if (a - b) is constant
            r_left, r_right = right.orphans
            return right.create_copy([left - r_left, -r_right])
    elif isinstance(left, Addition):
        if right == left.right:
            # Simplify (b + a) - a to b
            return left.left
        if right == left.left:
            # Simplify (a + b) - a to b
            return left.right
    elif isinstance(left, Subtraction):
        if right == left.left:
            # Simplify (b - a) - b to -a
            return -left.right
    elif isinstance(right, Addition):
        if left == right.left:
            # Simplify a - (a + b) to -b
            return -right.right
        if left == right.right:
            # Simplify a - (b + a) to -b
            return -right.left
    elif isinstance(right, Subtraction):
        if left == right.left:
            # Simplify a - (a - b) to b
            return right.right

    return pybamm.simplify_if_constant(Subtraction(left, right))


def multiply(
    left: ChildSymbol,
    right: ChildSymbol,
):
    left, right = _simplify_elementwise_binary_broadcasts(left, right)

    # Move constant to always be on the left
    if right.is_constant() and not left.is_constant():
        left, right = right, left

    # Check for Concatenations and Broadcasts
    out = _simplified_binary_broadcast_concatenation(left, right, multiply)
    if out is not None:
        return out

    # simplify multiply by scalar zero, being careful about shape
    if pybamm.is_scalar_zero(left):
        return pybamm.zeros_like(right)

    # if one of the children is a zero matrix, we have to be careful about shapes
    if pybamm.is_matrix_zero(left):
        return pybamm.zeros_like(Multiplication(left, right))

    # anything multiplied by a scalar one returns itself
    if pybamm.is_scalar_one(left):
        return right

    # anything multiplied by a scalar negative one returns negative itself
    if pybamm.is_scalar_minus_one(left):
        return -right

    # Return constant if both sides are constant
    if left.is_constant() and right.is_constant():
        return pybamm.simplify_if_constant(Multiplication(left, right))

    # anything multiplied by a matrix one returns itself if
    # - the shapes are the same
    # - both left and right evaluate on edges, or both evaluate on nodes, in all
    # dimensions
    # (and possibly more generally, but not implemented here)
    try:
        if left.shape_for_testing == right.shape_for_testing and all(
            left.evaluates_on_edges(dim) == right.evaluates_on_edges(dim)
            for dim in left.domains.keys()
        ):
            if pybamm.is_matrix_one(left):
                return right
            # also check for negative one
            if pybamm.is_matrix_minus_one(left):
                return -right

    except NotImplementedError:
        pass

    if left.is_constant():
        # Simplify a * (B @ c) to (a * B) @ c if (a * B) is constant
        if (
            isinstance(right, MatrixMultiplication)
            and right.left.is_constant()
            and not (left.ndim_for_testing == 2 and left.shape_for_testing[1] > 1)
        ):
            r_left, r_right = right.orphans
            new_left = left * r_left
            # be careful about domains to avoid weird errors
            new_left.clear_domains()
            new_mul = new_left @ r_right
            # Keep the domain of the old right
            new_mul.copy_domains(right)
            return new_mul

        elif isinstance(right, Multiplication):
            # Simplify a * (b * c) to (a * b) * c if (a * b) is constant
            if right.left.is_constant():
                r_left, r_right = right.orphans
                return (left * r_left) * r_right
        elif isinstance(right, Division):
            # Simplify a * (b / c) to (a * b) / c if (a * c) is constant
            if right.left.is_constant():
                r_left, r_right = right.orphans
                return (left * r_left) / r_right

        # Simplify a * (b + c) to (a * b) + (a * c) if (a * b) is constant
        # This is a common construction that appears from discretisation of spatial
        # operators
        # Also do this for cases like a * (b @ c + d) where (a * b) is constant
        elif isinstance(right, (Addition, Subtraction)):
            mul_classes = (Multiplication, MatrixMultiplication)
            if (
                right.left.is_constant()
                or (
                    isinstance(right.left, mul_classes)
                    and right.left.left.is_constant()
                )
                or (
                    isinstance(right.right, mul_classes)
                    and right.right.left.is_constant()
                )
            ):
                r_left, r_right = right.orphans
                if (r_left.domain == right.domain or r_left.domain == []) and (
                    r_right.domain == right.domain or r_right.domain == []
                ):
                    if isinstance(right, Addition):
                        return (left * r_left) + (left * r_right)
                    elif isinstance(right, Subtraction):
                        return (left * r_left) - (left * r_right)

    # Cancelling out common terms
    if isinstance(left, Division):
        # Simplify (a / b) * b to a
        if left.right == right:
            return left.left
    if isinstance(right, Division):
        # Simplify a * (b / a) to b
        if left == right.right:
            return right.left

    # Negation simplifications
    if isinstance(left, pybamm.Negate) and isinstance(right, pybamm.Negate):
        # Double negation cancels out
        return left.orphans[0] * right.orphans[0]
    elif isinstance(right, pybamm.Negate) and left.is_constant():
        # Simplify a * (-b) to (-a) * b if (-a) is constant
        return (-left) * right.orphans[0]

    return Multiplication(left, right)


def divide(
    left: ChildSymbol,
    right: ChildSymbol,
):
    left, right = _simplify_elementwise_binary_broadcasts(left, right)

    # anything divided by zero raises error
    if pybamm.is_scalar_zero(right):
        raise ZeroDivisionError

    # Move constant to always be on the left
    # For a division, this means (var / constant) becomes (1/constant * var)
    if right.is_constant() and not left.is_constant():
        return (1 / right) * left

    # Check for Concatenations and Broadcasts
    out = _simplified_binary_broadcast_concatenation(left, right, divide)
    if out is not None:
        return out

    # zero divided by anything returns zero (being careful about shape)
    if pybamm.is_scalar_zero(left):
        return pybamm.zeros_like(right)

    # matrix zero divided by anything returns matrix zero (i.e. itself)
    if pybamm.is_matrix_zero(left):
        return pybamm.zeros_like(Division(left, right))

    # a symbol divided by itself is 1s of the same shape
    if left == right:
        return pybamm.ones_like(left)

    # Return constant if both sides are constant
    if left.is_constant() and right.is_constant():
        return pybamm.simplify_if_constant(Division(left, right))

    if left.is_constant():
        if isinstance(right, (Multiplication, Division)) and right.left.is_constant():
            r_left, r_right = right.orphans
            # Simplify a / (b */ c) to (a / b) /* c if (a / b) is constant
            if isinstance(right, Multiplication):
                return (left / r_left) / r_right
            elif isinstance(right, Division):
                return (left / r_left) * r_right

    # Cancelling out common terms
    if isinstance(left, Multiplication):
        if left.left == right:
            # Make sure shape is preserved
            return left.right * pybamm.ones_like(left.left)
        elif left.right == right:
            return left.left
        elif isinstance(right, Multiplication):
            if left.left == right.left:
                _, l_right = left.orphans
                _, r_right = right.orphans
                return l_right / r_right
            if left.right == right.right:
                l_left, _ = left.orphans
                r_left, _ = right.orphans
                return l_left / r_left

    # Negation simplifications
    if isinstance(right, pybamm.Negate):
        if isinstance(left, pybamm.Negate):
            # Double negation cancels out
            return left.orphans[0] / right.orphans[0]
        elif left.is_constant():
            # Simplify a / (-b) to (-a) / b if (-a) is constant
            return (-left) / right.orphans[0]

    return pybamm.simplify_if_constant(Division(left, right))


def matmul(
    left_child: ChildSymbol,
    right_child: ChildSymbol,
):
    left, right = _preprocess_binary(left_child, right_child)
    if pybamm.is_matrix_zero(left) or pybamm.is_matrix_zero(right):
        return pybamm.zeros_like(MatrixMultiplication(left, right))

    if isinstance(right, Multiplication) and left.is_constant():
        # Simplify A @ (b * c) to (A * b) @ c if (A * b) is constant
        if right.left.evaluates_to_constant_number():
            r_left, r_right = right.orphans
            return (left * r_left) @ r_right

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

    elif left.is_constant() and isinstance(right, (Addition, Subtraction)):
        # Simplify A @ (b +- c) to (A @ b) +- (A @ c) if (A @ b) or (A @ c) is constant
        # This is a common construction that appears from discretisation of spatial
        # operators
        # Or simplify A @ (B @ b +- C @ c) to (A @ B @ b) +- (A @ C @ c) if (A @ B)
        # and (A @ C) are constant
        # Don't do this if either b or c is a number as this will lead to matmul errors
        if (
            (right.left.is_constant() or right.right.is_constant())
            # these lines should work but don't, possibly because of poorly
            # conditioned model?
            or (
                isinstance(right.left, MatrixMultiplication)
                and right.left.left.is_constant()
                and isinstance(right.right, MatrixMultiplication)
                and right.right.left.is_constant()
            )
        ) and not (
            right.left.size_for_testing == 1 or right.right.size_for_testing == 1
        ):
            r_left, r_right = right.orphans
            r_left.domains = right.domains
            r_right.domains = right.domains
            if isinstance(right, Addition):
                return (left @ r_left) + (left @ r_right)
            elif isinstance(right, Subtraction):
                return (left @ r_left) - (left @ r_right)

    return pybamm.simplify_if_constant(MatrixMultiplication(left, right))


def minimum(
    left: ChildSymbol,
    right: ChildSymbol,
) -> pybamm.Symbol:
    """
    Returns the smaller of two objects, possibly with a smoothing approximation.
    Not to be confused with :meth:`pybamm.min`, which returns min function of child.
    """
    # Check for Concatenations and Broadcasts
    left, right = _simplify_elementwise_binary_broadcasts(left, right)
    concat_out = _simplified_binary_broadcast_concatenation(left, right, minimum)
    if concat_out is not None:
        return concat_out

    mode = pybamm.settings.min_max_mode
    k = pybamm.settings.min_max_smoothing
    # Return exact approximation if that is the setting or the outcome is a constant
    # (i.e. no need for smoothing)
    if mode == "exact" or (left.is_constant() and right.is_constant()):
        out = Minimum(left, right)
    elif mode == "smooth":
        out = pybamm.smooth_min(left, right, k)
    else:
        out = pybamm.softminus(left, right, k)
    return pybamm.simplify_if_constant(out)


def maximum(
    left: ChildSymbol,
    right: ChildSymbol,
):
    """
    Returns the larger of two objects, possibly with a smoothing approximation.
    Not to be confused with :meth:`pybamm.max`, which returns max function of child.
    """
    # Check for Concatenations and Broadcasts
    left, right = _simplify_elementwise_binary_broadcasts(left, right)
    concat_out = _simplified_binary_broadcast_concatenation(left, right, maximum)
    if concat_out is not None:
        return concat_out

    mode = pybamm.settings.min_max_mode
    k = pybamm.settings.min_max_smoothing
    # Return exact approximation if that is the setting or the outcome is a constant
    # (i.e. no need for smoothing)
    if mode == "exact" or (left.is_constant() and right.is_constant()):
        out = Maximum(left, right)
    elif mode == "smooth":
        out = pybamm.smooth_max(left, right, k)
    else:
        out = pybamm.softplus(left, right, k)
    return pybamm.simplify_if_constant(out)


def _heaviside(left: ChildSymbol, right: ChildSymbol, equal):
    """return a :class:`EqualHeaviside` object, or a smooth approximation."""
    # Check for Concatenations and Broadcasts
    left, right = _simplify_elementwise_binary_broadcasts(left, right)
    concat_out = _simplified_binary_broadcast_concatenation(
        left, right, functools.partial(_heaviside, equal=equal)
    )
    if concat_out is not None:
        return concat_out

    if (
        left.is_constant()
        and isinstance(right, BinaryOperator)
        and right.left.is_constant()
    ):
        if isinstance(right, Addition):
            # simplify heaviside(a, b + var) to heaviside(a - b, var)
            return _heaviside(left - right.left, right.right, equal=equal)
        # elif isinstance(right, Multiplication):
        #     # simplify heaviside(a, b * var) to heaviside(a/b, var)
        #     if right.left.evaluate() > 0:
        #         return _heaviside(left / right.left, right.right, equal=equal)
        #     else:
        #         # maintain the sign of each side
        #         return _heaviside(left / -right.left, -right.right, equal=equal)

    k = pybamm.settings.heaviside_smoothing
    # Return exact approximation if that is the setting or the outcome is a constant
    # (i.e. no need for smoothing)
    if k == "exact" or (left.is_constant() and right.is_constant()):
        if equal is True:
            out: pybamm.EqualHeaviside = pybamm.EqualHeaviside(left, right)
        else:
            out: pybamm.NotEqualHeaviside = pybamm.NotEqualHeaviside(left, right)  # type: ignore[no-redef]
    else:
        out = pybamm.sigmoid(left, right, k)
    return pybamm.simplify_if_constant(out)


def softminus(
    left: pybamm.Symbol,
    right: pybamm.Symbol,
    k: float,
):
    """
    Softminus approximation to the minimum function. k is the smoothing parameter,
    set by `pybamm.settings.min_max_smoothing`. The recommended value is k=10.
    """
    return pybamm.log(pybamm.exp(-k * left) + pybamm.exp(-k * right)) / -k


def softplus(
    left: pybamm.Symbol,
    right: pybamm.Symbol,
    k: float,
):
    """
    Softplus approximation to the maximum function. k is the smoothing parameter,
    set by `pybamm.settings.min_max_smoothing`. The recommended value is k=10.
    """
    return pybamm.log(pybamm.exp(k * left) + pybamm.exp(k * right)) / k


def smooth_min(left, right, k):
    """
    Smooth_min approximation to the minimum function. k is the smoothing parameter,
    set by `pybamm.settings.min_max_smoothing`. The recommended value is k=100.
    """
    sigma = (1.0 / k) ** 2
    return ((left + right) - (pybamm.sqrt((left - right) ** 2 + sigma))) / 2


def smooth_max(left, right, k):
    """
    Smooth_max approximation to the maximum function. k is the smoothing parameter,
    set by `pybamm.settings.min_max_smoothing`. The recommended value is k=100.
    """
    sigma = (1.0 / k) ** 2
    return (pybamm.sqrt((left - right) ** 2 + sigma) + (left + right)) / 2


def sigmoid(
    left: pybamm.Symbol,
    right: pybamm.Symbol,
    k: float,
):
    """
    Sigmoidal approximation to the heaviside function. k is the smoothing parameter,
    set by `pybamm.settings.heaviside_smoothing`. The recommended value is k=10.
    Note that the concept of deciding which side to pick when left=right does not apply
    for this smooth approximation. When left=right, the value is (left+right)/2.
    """
    return (1 + pybamm.tanh(k * (right - left))) / 2


def source(
    left: Numeric | pybamm.Symbol,
    right: pybamm.Symbol,
    boundary=False,
):
    """
    A convenience function for creating (part of) an expression tree representing
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

    left : :class:`Symbol`, numeric
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

    # force type cast for mypy
    left = cast(pybamm.Symbol, left)

    if left.domain != ["current collector"] or right.domain != ["current collector"]:
        raise pybamm.DomainError(
            "'source' only implemented in the 'current collector' domain, "
            f"but symbols have domains {left.domain} and {right.domain}"
        )
    if boundary:
        return pybamm.BoundaryMass(right) @ left
    else:
        return pybamm.Mass(right) @ left
