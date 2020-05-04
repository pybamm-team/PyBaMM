#
# Simplify a symbol
#
import pybamm

import numpy as np
import numbers
from scipy.sparse import issparse, csr_matrix


def simplify_if_constant(symbol, keep_domains=False):
    """
    Utility function to simplify an expression tree if it evalutes to a constant
    scalar, vector or matrix
    """
    if keep_domains is True:
        domain = symbol.domain
        auxiliary_domains = symbol.auxiliary_domains
    else:
        domain = None
        auxiliary_domains = None
    if symbol.is_constant():
        result = symbol.evaluate_ignoring_errors()
        if result is not None:
            if (
                isinstance(result, numbers.Number)
                or (isinstance(result, np.ndarray) and result.ndim == 0)
                or isinstance(result, np.bool_)
            ):
                return pybamm.Scalar(result)
            elif isinstance(result, np.ndarray) or issparse(result):
                if result.ndim == 1 or result.shape[1] == 1:
                    return pybamm.Vector(
                        result, domain=domain, auxiliary_domains=auxiliary_domains
                    )
                else:
                    # Turn matrix of zeros into sparse matrix
                    if isinstance(result, np.ndarray) and np.all(result == 0):
                        result = csr_matrix(result)
                    return pybamm.Matrix(
                        result, domain=domain, auxiliary_domains=auxiliary_domains
                    )

    return symbol


def simplify_addition_subtraction(myclass, left, right):
    """
    if children are associative (addition, subtraction, etc) then try to find groups of
    constant children (that produce a value) and simplify them to a single term

    The purpose of this function is to simplify expressions like (1 + (1 + p)), which
    should be simplified to (2 + p). The former expression consists of an Addition, with
    a left child of Scalar type, and a right child of another Addition containing a
    Scalar and a Parameter. For this case, this function will first flatten the
    expression to a list of the bottom level children (i.e. [Scalar(1), Scalar(2),
    Parameter(p)]), and their operators (i.e. [None, Addition, Addition]), and then
    combine all the constant children (i.e. Scalar(1) and Scalar(1)) to a single child
    (i.e. Scalar(2))

    Note that this function will flatten the expression tree until a symbol is found
    that is not either an Addition or a Subtraction, so this function would simplify
    (3 - (2 + a*b*c)) to (1 + a*b*c)

    This function is useful if different children expressions contain non-constant terms
    that prevent them from being simplified, so for example (1 + a) + (b - 2) - (6 + c)
    will be simplified to (-7 + a + b - c)

    Parameters
    ----------

    myclass: class
        the binary operator class (pybamm.Addition or pybamm.Subtraction) operating on
        children left and right
    left: derived from pybamm.Symbol
        the left child of the binary operator
    right: derived from pybamm.Symbol
        the right child of the binary operator

    """
    numerator = []
    numerator_types = []

    def flatten(this_class, left_child, right_child, in_subtraction):
        """
        recursive function to flatten a term involving only additions or subtractions

        outputs to lists `numerator` and `numerator_types`

        Note that domains are all set to [] as we do not wish to consider domains once
        simplifications are applied

        e.g.

        (1 + 2) + 3       -> [1, 2, 3]    and [None, Addition, Addition]
        1 + (2 - 3)       -> [1, 2, 3]    and [None, Addition, Subtraction]
        1 - (2 + 3)       -> [1, 2, 3]    and [None, Subtraction, Subtraction]
        (1 + 2) - (2 + 3) -> [1, 2, 2, 3] and [None, Addition, Subtraction, Subtraction]
        """

        left_child.clear_domains()
        right_child.clear_domains()
        for side, child in [("left", left_child), ("right", right_child)]:
            if isinstance(child, (pybamm.Addition, pybamm.Subtraction)):
                left, right = child.orphans
                flatten(child.__class__, left, right, in_subtraction)

            else:
                numerator.append(child)
                if in_subtraction is None:
                    numerator_types.append(None)
                elif in_subtraction:
                    numerator_types.append(pybamm.Subtraction)
                else:
                    numerator_types.append(pybamm.Addition)

            if side == "left":
                if in_subtraction is None:
                    in_subtraction = this_class == pybamm.Subtraction
                elif this_class == pybamm.Subtraction:
                    in_subtraction = not in_subtraction

    flatten(myclass, left, right, None)

    def partition_by_constant(source, types):
        """
        function to partition a source list of symbols into those that return a constant
        value, and those that do not
        """
        constant = []
        nonconstant = []
        constant_types = []
        nonconstant_types = []

        for child, op_type in zip(source, types):
            if child.is_constant() and child.evaluate_ignoring_errors() is not None:
                constant.append(child)
                constant_types.append(op_type)
            else:
                nonconstant.append(child)
                nonconstant_types.append(op_type)
        return constant, nonconstant, constant_types, nonconstant_types

    def fold_add_subtract(array, types):
        """
        performs a fold operation on the children nodes in `array`, using the operator
        types given in `types`

        e.g. if the input was:
        array = [1, 2, 3, 4]
        types = [None, +, -, +]

        the result would be 1 + 2 - 3 + 4
        """
        ret = None
        if len(array) > 0:
            if types[0] in [None, pybamm.Addition]:
                ret = array[0]
            elif types[0] == pybamm.Subtraction:
                ret = -array[0]
            for child, typ in zip(array[1:], types[1:]):
                if typ == pybamm.Addition:
                    ret += child
                else:
                    ret -= child
        return ret

    # simplify identical terms
    i = 0
    while i < len(numerator) - 1:
        if isinstance(numerator[i], pybamm.Multiplication) and isinstance(
            numerator[i].children[0], pybamm.Scalar
        ):
            term_i = numerator[i].orphans[1]
            term_i_count = numerator[i].children[0].evaluate()
        else:
            term_i = numerator[i]
            term_i_count = 1

        # loop through rest of numerator counting up and deleting identical terms
        for j, (term_j, typ_j) in enumerate(
            zip(numerator[i + 1 :], numerator_types[i + 1 :])
        ):
            if isinstance(term_j, pybamm.Multiplication) and isinstance(
                term_j.left, pybamm.Scalar
            ):
                factor = term_j.left.evaluate()
                term_j = term_j.right
            else:
                factor = 1
            if term_i.id == term_j.id:
                if typ_j == pybamm.Addition:
                    term_i_count += factor
                elif typ_j == pybamm.Subtraction:
                    term_i_count -= factor
                del numerator[j + i + 1]
                del numerator_types[j + i + 1]

        # replace this term by count * term if count > 1
        if term_i_count != 1:
            # simplify the result just in case
            # (e.g. count == 0, or can fold constant into the term)
            numerator[i] = (term_i_count * term_i).simplify()

        i += 1

    # can reorder the numerator
    (constant, nonconstant, constant_types, nonconstant_types) = partition_by_constant(
        numerator, numerator_types
    )

    constant_expr = fold_add_subtract(constant, constant_types)
    nonconstant_expr = fold_add_subtract(nonconstant, nonconstant_types)

    if constant_expr is not None and nonconstant_expr is None:
        # might be no nonconstants
        new_expression = pybamm.simplify_if_constant(constant_expr)
    elif constant_expr is None and nonconstant_expr is not None:
        # might be no constants
        new_expression = nonconstant_expr
    else:
        # or mix of both
        constant_expr = pybamm.simplify_if_constant(constant_expr)
        new_expression = constant_expr + nonconstant_expr

    return new_expression


def simplify_multiplication_division(myclass, left, right):
    """
    if children are associative (multiply, division, etc) then try to find
    groups of constant children (that produce a value) and simplify them

    The purpose of this function is to simplify expressions of the type (1 * c / 2),
    which should simplify to (0.5 * c). The former expression consists of a Division,
    with a left child of a Multiplication containing a Scalar and a Parameter, and a
    right child consisting of a Scalar. For this case, this function will first flatten
    the expression to a list of the bottom level children on the numerator (i.e.
    [Scalar(1), Parameter(c)]) and their operators (i.e. [None, Multiplication]), as
    well as those children on the denominator (i.e. [Scalar(2)]. After this, all the
    constant children on the numerator and denominator (i.e. Scalar(1) and Scalar(2))
    will be combined appropriately, in this case to Scalar(0.5), and combined with the
    nonconstant children (i.e. Parameter(c))

    Note that this function will flatten the expression tree until a symbol is found
    that is not either an Multiplication, Division or MatrixMultiplication, so this
    function would simplify (3*(1 + d)*2) to (6 * (1 + d))

    As well as Multiplication and Division, this function can handle
    MatrixMultiplication. If any MatrixMultiplications are found on the
    numerator/denominator, no reordering of children is done to find groups of constant
    children. In this case only neighbouring constant children on the numerator are
    simplified

    Parameters
    ----------

    myclass: class
        the binary operator class (pybamm.Addition or pybamm.Subtraction) operating on
        children left and right
    left: derived from pybamm.Symbol
        the left child of the binary operator
    right: derived from pybamm.Symbol
        the right child of the binary operator

    """
    numerator = []
    denominator = []
    numerator_types = []
    denominator_types = []

    # recursive function to flatten a term involving only multiplications or divisions
    def flatten(
        previous_class,
        this_class,
        left_child,
        right_child,
        in_numerator,
        in_matrix_multiplication,
    ):
        """
        recursive function to flatten a term involving only Multiplication, Division or
        MatrixMultiplication. keeps track of wether a term is on the numerator or
        denominator. For those terms on the numerator, their operator type
        (Multiplication or MatrixMultiplication) is stored

        Note that multiplication *within* matrix multiplications, e.g. a@(b*c), are not
        flattened into a@b*c, as this would be incorrect (see #253)

        Note that the domains are all set to [] as we do not wish to consider domains
        once simplifications are applied

        outputs to lists `numerator`, `denominator` and `numerator_types`

        e.g.
        expression     numerator  denominator  numerator_types
        (1 * 2) / 3 ->  [1, 2]       [3]       [None, Multiplication]
        (1 @ 2) / 3 ->  [1, 2]       [3]       [None, MatrixMultiplication]
        1 / (c / 2) ->  [1, 2]       [c]       [None, Multiplication]
        """

        left_child.clear_domains()
        right_child.clear_domains()
        for side, child in [("left", left_child), ("right", right_child)]:

            if side == "left":
                other_child = right_child
            else:
                other_child = left_child

            # flatten if all matrix multiplications
            # flatten if one child is a matrix mult if the other term is a scalar or
            # vector
            if isinstance(child, pybamm.MatrixMultiplication) and (
                in_matrix_multiplication
                or isinstance(other_child, (pybamm.Scalar, pybamm.Vector))
            ):
                left, right = child.orphans
                if (
                    side == "left"
                    and this_class == pybamm.Multiplication
                    and isinstance(other_child, pybamm.Vector)
                ):
                    # change (m @ v1) * v2 -> v2 * m @ v so can simplify correctly
                    # (#341)
                    numerator.append(other_child)
                    numerator_types.append(previous_class)
                    flatten(
                        this_class, child.__class__, left, right, in_numerator, True
                    )
                    break
                if side == "left":
                    flatten(
                        previous_class, child.__class__, left, right, in_numerator, True
                    )
                else:
                    flatten(
                        this_class, child.__class__, left, right, in_numerator, True
                    )
            # flatten if all multiplies and divides
            elif (
                isinstance(child, (pybamm.Multiplication, pybamm.Division))
                and not in_matrix_multiplication
            ):
                left, right = child.orphans
                if side == "left":
                    flatten(
                        previous_class,
                        child.__class__,
                        left,
                        right,
                        in_numerator,
                        False,
                    )
                else:
                    flatten(
                        this_class, child.__class__, left, right, in_numerator, False
                    )
            # everything else don't flatten
            else:
                if in_numerator:
                    numerator.append(child)
                    if side == "left":
                        numerator_types.append(previous_class)
                    else:
                        numerator_types.append(this_class)
                else:
                    denominator.append(child)
                    if side == "left":
                        denominator_types.append(previous_class)
                    else:
                        denominator_types.append(this_class)

            if side == "left" and this_class == pybamm.Division:
                in_numerator = not in_numerator

    flatten(None, myclass, left, right, True, myclass == pybamm.MatrixMultiplication)

    # check if there is a matrix multiply in the numerator (if so we can't reorder it)
    numerator_has_mat_mul = any(
        [typ == pybamm.MatrixMultiplication for typ in numerator_types + [myclass]]
    )

    denominator_has_mat_mul = any(
        [typ == pybamm.MatrixMultiplication for typ in denominator_types]
    )

    def partition_by_constant(source, types=None):
        """
        function to partition a source list of symbols into those that return a constant
        value, and those that do not
        """
        constant = []
        nonconstant = []

        for child in source:
            if child.is_constant() and child.evaluate_ignoring_errors() is not None:
                constant.append(child)
            else:
                nonconstant.append(child)
        return constant, nonconstant

    def fold_multiply(array, types=None):
        """
        performs a fold operation on the children nodes in `array`, using the operator
        types given in `types`

        e.g. if the input was:
        array = [1, 2, 3, 4]
        types = [None, *, @, *]

        the result would be 1 * 2 @ 3 * 4
        """
        ret = None
        if len(array) > 0:
            if types is None:
                ret = array[0]
                for child in array[1:]:
                    ret *= child
            else:
                # work backwards through 'array' and 'types' so that multiplications
                # and matrix multiplications are performed in the most efficient order
                ret = array[-1]
                for child, typ in zip(reversed(array[:-1]), reversed(types[1:])):
                    if typ == pybamm.MatrixMultiplication:
                        ret = child @ ret
                    else:
                        ret = child * ret
        return ret

    def simplify_with_mat_mul(nodes, types):
        new_nodes = [nodes[0]]
        new_types = [types[0]]
        for child, typ in zip(nodes[1:], types[1:]):
            if (
                new_nodes[-1].is_constant()
                and child.is_constant()
                and new_nodes[-1].evaluate_ignoring_errors() is not None
                and child.evaluate_ignoring_errors() is not None
            ):
                if typ == pybamm.MatrixMultiplication:
                    new_nodes[-1] = new_nodes[-1] @ child
                else:
                    new_nodes[-1] *= child
                new_nodes[-1] = pybamm.simplify_if_constant(new_nodes[-1])
            else:
                new_nodes.append(child)
                new_types.append(typ)
        new_nodes = fold_multiply(new_nodes, new_types)
        return new_nodes

    if numerator_has_mat_mul and denominator_has_mat_mul:
        new_numerator = simplify_with_mat_mul(numerator, numerator_types)
        new_denominator = simplify_with_mat_mul(denominator, denominator_types)
        if new_denominator is None:
            result = new_numerator
        else:
            result = new_numerator / new_denominator

    elif numerator_has_mat_mul and not denominator_has_mat_mul:
        # can reorder the denominator since no matrix multiplies
        denominator_constant, denominator_nonconst = partition_by_constant(denominator)

        constant_denominator_expr = fold_multiply(denominator_constant)
        nonconst_denominator_expr = fold_multiply(denominator_nonconst)

        # fold constant denominator expr into numerator if possible
        if constant_denominator_expr is not None:
            for i, child in enumerate(numerator):
                if child.is_constant() and child.evaluate_ignoring_errors() is not None:
                    numerator[i] = child / constant_denominator_expr
                    numerator[i] = pybamm.simplify_if_constant(numerator[i])
                    constant_denominator_expr = None

        new_numerator = simplify_with_mat_mul(numerator, numerator_types)

        # result = constant_numerator_expr * new_numerator / nonconst_denominator_expr
        # need to take into accound that terms can be None
        if constant_denominator_expr is None:
            if nonconst_denominator_expr is None:
                result = new_numerator
            else:
                result = new_numerator / nonconst_denominator_expr
        else:
            # invert constant denominator terms for speed
            constant_numerator_expr = pybamm.simplify_if_constant(
                1 / constant_denominator_expr
            )

            if nonconst_denominator_expr is None:
                result = constant_numerator_expr * new_numerator
            else:
                result = (
                    constant_numerator_expr * new_numerator / nonconst_denominator_expr
                )

    elif not numerator_has_mat_mul and denominator_has_mat_mul:
        new_denominator = simplify_with_mat_mul(denominator, denominator_types)

        # can reorder the numerator since no matrix multiplies
        numerator_constant, numerator_nonconst = partition_by_constant(numerator)

        constant_numerator_expr = fold_multiply(numerator_constant)
        nonconst_numerator_expr = fold_multiply(numerator_nonconst)

        # result = constant_numerator_expr * nonconst_numerator_expr / new_denominator
        # need to take into account that terms can be None
        if constant_numerator_expr is None:
            result = nonconst_numerator_expr / new_denominator
        else:
            constant_numerator_expr = pybamm.simplify_if_constant(
                constant_numerator_expr
            )
            if nonconst_numerator_expr is None:
                result = constant_numerator_expr / new_denominator
            else:
                result = (
                    constant_numerator_expr * nonconst_numerator_expr / new_denominator
                )

    else:
        # can reorder the numerator since no matrix multiplies
        numerator_constant, numerator_nonconstant = partition_by_constant(numerator)

        constant_numerator_expr = fold_multiply(numerator_constant)
        nonconst_numerator_expr = fold_multiply(numerator_nonconstant)

        # can reorder the denominator since no matrix multiplies
        denominator_constant, denominator_nonconst = partition_by_constant(denominator)

        constant_denominator_expr = fold_multiply(denominator_constant)
        nonconst_denominator_expr = fold_multiply(denominator_nonconst)

        if constant_numerator_expr is not None:
            if constant_denominator_expr is not None:
                constant_numerator_expr = pybamm.simplify_if_constant(
                    constant_numerator_expr / constant_denominator_expr
                )
            else:
                constant_numerator_expr = pybamm.simplify_if_constant(
                    constant_numerator_expr
                )
        else:
            if constant_denominator_expr is not None:
                constant_numerator_expr = pybamm.simplify_if_constant(
                    1 / constant_denominator_expr
                )

        # result = constant_numerator_expr * nonconst_numerator_expr
        #    / nonconst_denominator_expr
        # need to take into account that terms can be None
        if constant_numerator_expr is None:
            result = nonconst_numerator_expr
        else:
            if nonconst_numerator_expr is None:
                result = constant_numerator_expr
            else:
                result = constant_numerator_expr * nonconst_numerator_expr

        if nonconst_denominator_expr is not None:
            result = result / nonconst_denominator_expr

    return result


class Simplification(object):
    def __init__(self, simplified_symbols=None):
        self._simplified_symbols = simplified_symbols or {}

    def simplify(self, symbol, clear_domains=True):
        """
        This function recurses down the tree, applying any simplifications defined in
        classes derived from pybamm.Symbol. E.g. any expression multiplied by a
        pybamm.Scalar(0) will be simplified to a pybamm.Scalar(0).
        If a symbol has already been simplified, the stored value is returned.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The symbol to simplify
        clear_domains : bool
            Whether to remove a symbol's domain when simplifying. Default is True.

        Returns
        -------
        :class:`pybamm.Symbol`
        Simplified symbol
        """

        try:
            return self._simplified_symbols[symbol.id]
        except KeyError:
            simplified_symbol = self._simplify(symbol, clear_domains)

            self._simplified_symbols[symbol.id] = simplified_symbol

            return simplified_symbol

    def _simplify(self, symbol, clear_domains=True):
        """ See :meth:`Simplification.simplify()`. """
        if clear_domains:
            symbol.clear_domains()

        if isinstance(symbol, pybamm.BinaryOperator):
            left, right = symbol.children
            # process children
            new_left = self.simplify(left)
            new_right = self.simplify(right)
            # _binary_simplify defined in derived classes for specific rules
            new_symbol = symbol._binary_simplify(new_left, new_right)

        elif isinstance(symbol, pybamm.UnaryOperator):
            # Reassign domain for gradient and divergence
            if isinstance(symbol, (pybamm.Gradient, pybamm.Divergence)):
                new_child = self.simplify(symbol.child, clear_domains=False)
            else:
                new_child = self.simplify(symbol.child)
            # _unary_simplify defined in derived classes for specific rules
            new_symbol = symbol._unary_simplify(new_child)

        elif isinstance(symbol, pybamm.Function):
            simplified_children = [None] * len(symbol.children)
            for i, child in enumerate(symbol.children):
                simplified_children[i] = self.simplify(child)
            # _function_simplify defined in function class
            new_symbol = symbol._function_simplify(simplified_children)

        elif isinstance(symbol, pybamm.Concatenation):
            new_children = [self.simplify(child) for child in symbol.children]
            new_symbol = symbol._concatenation_simplify(new_children)

        else:
            # Backup option: return new copy of the object
            try:
                new_symbol = symbol.new_copy()
                return new_symbol
            except NotImplementedError:
                raise NotImplementedError(
                    "Cannot simplify symbol of type '{}'".format(type(symbol))
                )

        return simplify_if_constant(new_symbol)
