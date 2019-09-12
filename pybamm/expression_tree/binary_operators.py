#
# Binary operator classes
#
import pybamm

import autograd.numpy as np
import numbers
from scipy.sparse import issparse, csr_matrix, kron


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
        # Turn numbers into scalars
        if isinstance(left, numbers.Number):
            left = pybamm.Scalar(left)
        if isinstance(right, numbers.Number):
            right = pybamm.Scalar(right)

        # Check and process domains, except for Outer symbol which takes the outer
        # product of two smbols in different domains, and gives it the domain of the
        # right child.
        if isinstance(self, (pybamm.Outer, pybamm.Kron)):
            domain = right.domain
            auxiliary_domains = {"secondary": left.domain}
        else:
            domain = self.get_children_domains(left.domain, right.domain)
            auxiliary_domains = self.get_children_auxiliary_domains(
                left.auxiliary_domains, right.auxiliary_domains
            )
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

    def get_children_auxiliary_domains(self, l_aux_domains, r_aux_domains):
        "Combine auxiliary domains from children, at all levels"
        aux_domains = {}
        for level in set(l_aux_domains.keys()).union(r_aux_domains.keys()):
            ldomain = l_aux_domains.get(level, [])
            rdomain = r_aux_domains.get(level, [])
            aux_domains[level] = self.get_children_domains(ldomain, rdomain)
        return aux_domains

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

    def evaluate_for_shape(self):
        """ See :meth:`pybamm.Symbol.evaluate_for_shape()`. """
        left = self.children[0].evaluate_for_shape()
        right = self.children[1].evaluate_for_shape()
        return self._binary_evaluate(left, right)

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
        return base ** (exponent - 1) * (
            exponent * base.diff(variable)
            + base * pybamm.log(base) * exponent.diff(variable)
        )

    def _jac(self, variable):
        """ See :meth:`pybamm.Symbol._jac()`. """
        # apply chain rule and power rule
        base, exponent = self.orphans
        if base.evaluates_to_number() and exponent.evaluates_to_number():
            return pybamm.Scalar(0)
        elif exponent.evaluates_to_number():
            return (exponent * base ** (exponent - 1)) * base.jac(variable)
        elif base.evaluates_to_number():
            return (base ** exponent * pybamm.log(base)) * exponent.jac(variable)
        else:
            return (base ** (exponent - 1)) * (
                exponent * base.jac(variable)
                + base * pybamm.log(base) * exponent.jac(variable)
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

    def _diff(self, variable):
        """ See :meth:`pybamm.Symbol._diff()`. """
        return self.left.diff(variable) + self.right.diff(variable)

    def _jac(self, variable):
        """ See :meth:`pybamm.Symbol._jac()`. """
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

    def _diff(self, variable):
        """ See :meth:`pybamm.Symbol._diff()`. """
        return self.left.diff(variable) - self.right.diff(variable)

    def _jac(self, variable):
        """ See :meth:`pybamm.Symbol._jac()`. """
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

    def _diff(self, variable):
        """ See :meth:`pybamm.Symbol._diff()`. """
        # apply product rule
        left, right = self.orphans
        return left.diff(variable) * right + left * right.diff(variable)

    def _jac(self, variable):
        """ See :meth:`pybamm.Symbol._jac()`. """
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
            return csr_matrix(left.multiply(right))
        elif issparse(right):
            # Hadamard product is commutative, so we can switch right and left
            return csr_matrix(right.multiply(left))
        else:
            return left * right

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.simplify()`. """

        # anything multiplied by a scalar zero returns a scalar zero
        if is_scalar_zero(left):
            if right.shape_for_testing == ():
                return pybamm.Scalar(0)
            else:
                return pybamm.Array(np.zeros(right.shape_for_testing))
        if is_scalar_zero(right):
            if left.shape_for_testing == ():
                return pybamm.Scalar(0)
            else:
                return pybamm.Array(np.zeros(left.shape_for_testing))

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
        raise NotImplementedError(
            "diff not implemented for symbol of type 'MatrixMultiplication'"
        )

    def _jac(self, variable):
        """ See :meth:`pybamm.Symbol._jac()`. """
        # We only need the case where left is an array and right
        # is a (slice of a) state vector, e.g. for discretised spatial
        # operators of the form D @ u (also catch cases of (-D) @ u)
        left, right = self.orphans
        if isinstance(left, pybamm.Array) or (
            isinstance(left, pybamm.Negate) and isinstance(left.child, pybamm.Array)
        ):
            left = pybamm.Matrix(csr_matrix(left.evaluate()))
            return left @ right.jac(variable)
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

    def _diff(self, variable):
        """ See :meth:`pybamm.Symbol._diff()`. """
        # apply quotient rule
        top, bottom = self.orphans
        return (top.diff(variable) * bottom - top * bottom.diff(variable)) / bottom ** 2

    def _jac(self, variable):
        """ See :meth:`pybamm.Symbol._jac()`. """
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
            return csr_matrix(left.multiply(1 / right))
        else:
            if isinstance(right, numbers.Number) and right == 0:
                return left * np.inf
            else:
                return left / right

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.simplify()`. """

        # zero divided by zero returns nan scalar
        if is_scalar_zero(left) and is_scalar_zero(right):
            return pybamm.Scalar(np.nan)

        # zero divided by anything returns zero
        if is_scalar_zero(left):
            if right.shape_for_testing == ():
                return pybamm.Scalar(0)
            else:
                return pybamm.Array(np.zeros(right.shape_for_testing))

        # anything divided by zero returns inf
        if is_scalar_zero(right):
            if left.shape_for_testing == ():
                return pybamm.Scalar(np.inf)
            else:
                return pybamm.Array(np.inf * np.ones(left.shape_for_testing))

        # anything divided by one is itself
        if is_one(right):
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

    def _jac(self, variable):
        """ See :meth:`pybamm.Symbol._jac()`. """
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

    def evaluates_on_edges(self):
        """ See :meth:`pybamm.Symbol.evaluates_on_edges()`. """
        return False


def inner(left, right):
    """
    Return inner product of two symbols.
    """
    return pybamm.Inner(left, right)


class Outer(BinaryOperator):
    """A node in the expression tree representing an outer product.
    This takes a 1D vector in the current collector domain of size (n,1) and a 1D
    variable of size (m,1), takes their outer product, and reshapes this into a vector
    of size (nm,1). It can also take in a vector in a single particle and a vector
    of the electrolyte domain to repeat that particle.
    Note: this class might be a bit dangerous, so at the moment it is very restrictive
    in what symbols can be passed to it

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        # cannot have Variable, StateVector or Matrix in the right symbol, as these
        # can already be 2D objects (so we can't take an outer product with them)
        if right.has_symbol_of_classes(
            (pybamm.Variable, pybamm.StateVector, pybamm.Matrix)
        ):
            raise TypeError(
                "right child must only contain SpatialVariable and scalars" ""
            )

        super().__init__("outer product", left, right)

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        return "outer({!s}, {!s})".format(self.left, self.right)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        raise NotImplementedError("diff not implemented for symbol of type 'Outer'")

    def _jac(self, variable):
        """ See :meth:`pybamm.Symbol._jac()`. """
        # right cannot be a StateVector, so no need for product rule
        left, right = self.orphans
        # make sure left child keeps same domain
        left.domain = self.left.domain
        return pybamm.Kron(left.jac(variable), right)

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """

        return np.outer(left, right).reshape(-1, 1)

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.simplify()`. """
        # Make sure left child keeps same domain
        left.domain = self.left.domain
        return pybamm.Outer(left, right)


class Kron(BinaryOperator):
    """A node in the expression tree representing a (sparse) kronecker product operator

    **Extends:** :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """

        super().__init__("kronecker product", left, right)

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        return "kron({!s}, {!s})".format(self.left, self.right)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        raise NotImplementedError("diff not implemented for symbol of type 'Kron'")

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        raise NotImplementedError("jac not implemented for symbol of type 'Kron'")

    def _binary_evaluate(self, left, right):
        """ See :meth:`pybamm.BinaryOperator._binary_evaluate()`. """
        return csr_matrix(kron(left, right))

    def _binary_simplify(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.simplify()`. """
        return pybamm.Kron(left, right)


def outer(left, right):
    """
    Return outer product of two symbols. If the symbols have the same domain, the outer
    product is just a multiplication. If they have different domains, make a copy of the
    left child with same domain as right child, and then take outer product.
    """
    try:
        return left * right
    except pybamm.DomainError:
        return pybamm.Outer(left, right)


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
        left = pybamm.Broadcast(left, "current collector")

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
