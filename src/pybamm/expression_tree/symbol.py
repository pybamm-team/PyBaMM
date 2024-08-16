#
# Base Symbol Class for the expression tree
#
from __future__ import annotations
import numbers
import warnings

import numpy as np
import sympy
from scipy.sparse import csr_matrix, issparse
from functools import cached_property
from typing import TYPE_CHECKING, cast
from collections.abc import Sequence

import pybamm
from pybamm.util import import_optional_dependency
from pybamm.expression_tree.printing.print_name import prettify_print_name

if TYPE_CHECKING:  # pragma: no cover
    import casadi
    from pybamm.type_definitions import (
        ChildSymbol,
        ChildValue,
        DomainType,
        AuxiliaryDomainType,
        DomainsType,
    )

DOMAIN_LEVELS = ["primary", "secondary", "tertiary", "quaternary"]
EMPTY_DOMAINS: dict[str, list] = {k: [] for k in DOMAIN_LEVELS}


def domain_size(domain: list[str] | str):
    """
    Get the domain size.

    Empty domain has size 1.
    If the domain falls within the list of standard battery domains, the size is read
    from a dictionary of standard domain sizes. Otherwise, the hash of the domain string
    is used to generate a `random` domain size.
    """
    fixed_domain_sizes = {
        "current collector": 3,
        "negative particle": 5,
        "positive particle": 7,
        "negative electrode": 11,
        "separator": 13,
        "positive electrode": 17,
        "negative particle size": 19,
        "positive particle size": 23,
    }
    if domain in [[], None]:
        size = 1
    elif all(dom in fixed_domain_sizes for dom in domain):
        size = sum(fixed_domain_sizes[dom] for dom in domain)
    else:
        size = sum(hash(dom) % 100 for dom in domain)
    return size


def create_object_of_size(size: int, typ="vector"):
    """Return object, consisting of NaNs, of the right shape."""
    if typ == "vector":
        return np.nan * np.ones((size, 1))
    elif typ == "matrix":
        return np.nan * np.ones((size, size))


def evaluate_for_shape_using_domain(domains: dict[str, list[str] | str], typ="vector"):
    """
    Return a vector of the appropriate shape, based on the domains.
    Domain 'sizes' can clash, but are unlikely to, and won't cause failures if they do.
    """
    if isinstance(domains, dict):
        _domain_sizes = int(np.prod([domain_size(dom) for dom in domains.values()]))
    else:
        _domain_sizes = domain_size(domains)
    return create_object_of_size(_domain_sizes, typ)


def is_constant(symbol: Symbol):
    return isinstance(symbol, numbers.Number) or symbol.is_constant()


def is_scalar_x(expr: Symbol, x: int):
    """
    Utility function to test if an expression evaluates to a constant scalar value
    """
    if is_constant(expr):
        result = expr.evaluate_ignoring_errors(t=None)
        return isinstance(result, numbers.Number) and result == x
    else:
        return False


def is_scalar_zero(expr: Symbol):
    """
    Utility function to test if an expression evaluates to a constant scalar zero
    """
    return is_scalar_x(expr, 0)


def is_scalar_one(expr: Symbol):
    """
    Utility function to test if an expression evaluates to a constant scalar one
    """
    return is_scalar_x(expr, 1)


def is_scalar_minus_one(expr: Symbol):
    """
    Utility function to test if an expression evaluates to a constant scalar minus one
    """
    return is_scalar_x(expr, -1)


def is_matrix_x(expr: Symbol, x: int):
    """
    Utility function to test if an expression evaluates to a constant matrix value
    """
    if isinstance(expr, pybamm.Broadcast):
        return is_scalar_x(expr.child, x) or is_matrix_x(expr.child, x)

    if is_constant(expr):
        result = expr.evaluate_ignoring_errors(t=None)
        return (
            issparse(result)
            and (
                (x == 0 and np.prod(len(result.__dict__["data"])) == 0)
                or (
                    len(result.__dict__["data"]) == np.prod(result.shape)
                    and np.all(result.__dict__["data"] == x)
                )
            )
        ) or (isinstance(result, np.ndarray) and np.all(result == x))
    else:
        return False


def is_matrix_zero(expr: Symbol):
    """
    Utility function to test if an expression evaluates to a constant matrix zero
    """
    return is_matrix_x(expr, 0)


def is_matrix_one(expr: Symbol):
    """
    Utility function to test if an expression evaluates to a constant matrix one
    """
    return is_matrix_x(expr, 1)


def is_matrix_minus_one(expr: Symbol):
    """
    Utility function to test if an expression evaluates to a constant matrix minus one
    """
    return is_matrix_x(expr, -1)


def simplify_if_constant(symbol: pybamm.Symbol):
    """
    Utility function to simplify an expression tree if it evalutes to a constant
    scalar, vector or matrix
    """
    if symbol.is_constant():
        result = symbol.evaluate_ignoring_errors()
        if result is not None:
            if (
                isinstance(result, numbers.Number)
                or (isinstance(result, np.ndarray) and result.ndim == 0)
                or isinstance(result, np.bool_)
            ):
                # type-narrow for Scalar
                new_result = cast(float, result)
                return pybamm.Scalar(new_result)
            elif isinstance(result, np.ndarray) or issparse(result):
                if result.ndim == 1 or result.shape[1] == 1:
                    return pybamm.Vector(result, domains=symbol.domains)
                else:
                    # Turn matrix of zeros into sparse matrix
                    if isinstance(result, np.ndarray) and np.all(result == 0):
                        result = csr_matrix(result)
                    return pybamm.Matrix(result, domains=symbol.domains)

    return symbol


class Symbol:
    """
    Base node class for the expression tree.

    Parameters
    ----------

    name : str
        name for the node
    children : iterable :class:`Symbol`, optional
        children to attach to this node, default to an empty list
    domain : iterable of str, or str
        list of domains over which the node is valid (empty list indicates the symbol
        is valid over all domains)
    auxiliary_domains : dict of str
        dictionary of auxiliary domains over which the node is valid (empty dictionary
        indicates no auxiliary domains). Keys can be "secondary", "tertiary" or
        "quaternary". The symbol is broadcast over its auxiliary domains.
        For example, a symbol might have domain "negative particle", secondary domain
        "separator" and tertiary domain "current collector" (`domain="negative
        particle", auxiliary_domains={"secondary": "separator", "tertiary": "current
        collector"}`).
    domains : dict
        A dictionary equivalent to {'primary': domain, auxiliary_domains}. Either
        'domain' and 'auxiliary_domains', or just 'domains', should be provided
        (not both). In future, the 'domain' and 'auxiliary_domains' arguments may be
        deprecated.
    """

    def __init__(
        self,
        name: str,
        children: Sequence[Symbol] | None = None,
        domain: DomainType = None,
        auxiliary_domains: AuxiliaryDomainType = None,
        domains: DomainsType = None,
    ):
        super().__init__()
        self.name = name

        if children is None:
            children = []

        self._children = children
        # Keep a separate "orphans" attribute for backwards compatibility
        self._orphans = children

        # Set domains (and hence id)
        self.domains = self.read_domain_or_domains(domain, auxiliary_domains, domains)

        self._saved_evaluates_on_edges: dict = {}
        self._print_name = None

        # Test shape on everything but nodes that contain the base Symbol class or
        # the base BinaryOperator class
        if pybamm.settings.debug_mode is True:
            if not any(
                issubclass(pybamm.Symbol, type(x))
                or issubclass(pybamm.BinaryOperator, type(x))
                for x in self.pre_order()
            ):
                self.test_shape()

    @classmethod
    def _from_json(cls, snippet: dict):
        """
        Reconstructs a Symbol instance during deserialisation of a JSON file.

        Parameters
        ----------
        snippet: dict
            Contains the information needed to reconstruct a specific instance.
            At minimum, should contain "name", "children" and "domains".
        """

        return cls(
            snippet["name"], children=snippet["children"], domains=snippet["domains"]
        )

    @property
    def children(self):
        """
        returns the cached children of this node.

        Note: it is assumed that children of a node are not modified after initial
        creation
        """
        return self._children

    @property
    def name(self):
        """name of the node."""
        return self._name

    @name.setter
    def name(self, value: str):
        if not isinstance(value, str):
            raise TypeError(f"{value} must be of type str")
        self._name = value

    @property
    def domains(self):
        return self._domains

    @domains.setter
    def domains(self, domains):
        try:
            if (
                self._domains == domains
                # accounting for empty domains
                or {k: v for k, v in self._domains.items() if v != []} == domains
            ):
                return  # no change
        except AttributeError:
            # self._domains has not been set yet
            pass

        # Turn dictionary into appropriate form
        if domains == {"primary": []}:
            self._domains = EMPTY_DOMAINS
            self.set_id()
            return

        # Set default domains
        domains = {**EMPTY_DOMAINS, **domains}

        # Check domains don't clash
        for level, dom in domains.items():
            if level not in DOMAIN_LEVELS:
                raise pybamm.DomainError(
                    f"Domain keys must be one of '{DOMAIN_LEVELS}'"
                )
            if isinstance(dom, str):
                domains[level] = [dom]

        values = [tuple(val) for val in domains.values() if val != []]
        if len(set(values)) != len(values):
            raise pybamm.DomainError("All domains must be different")

        for i, level in enumerate(DOMAIN_LEVELS[:-1]):
            if domains[level] == []:
                if domains[DOMAIN_LEVELS[i + 1]] != []:
                    raise pybamm.DomainError("Domain levels must be filled in order")
                # don't test further if we have already found a missing domain
                break

        self._domains = domains
        self.set_id()

    @property
    def domain(self):
        """
        list of applicable domains.

        Returns
        -------
            iterable of str
        """
        return self._domains["primary"]

    @domain.setter
    def domain(self, domain):
        raise NotImplementedError(
            "Cannot set domain directly, use domains={'primary': domain} instead"
        )

    @property
    def auxiliary_domains(self):
        """Returns auxiliary domains."""
        raise NotImplementedError(
            "symbol.auxiliary_domains has been deprecated, use symbol.domains instead"
        )

    @property
    def secondary_domain(self):
        """Helper function to get the secondary domain of a symbol."""
        return self._domains["secondary"]

    @property
    def tertiary_domain(self):
        """Helper function to get the tertiary domain of a symbol."""
        return self._domains["tertiary"]

    @property
    def quaternary_domain(self):
        """Helper function to get the quaternary domain of a symbol."""
        return self._domains["quaternary"]

    def copy_domains(self, symbol: Symbol):
        """Copy the domains from a given symbol, bypassing checks."""
        if self._domains != symbol._domains:
            self._domains = symbol._domains
            self.set_id()

    def clear_domains(self):
        """Clear domains, bypassing checks."""
        if self._domains != EMPTY_DOMAINS:
            self._domains = EMPTY_DOMAINS
            self.set_id()

    def get_children_domains(self, children: Sequence[Symbol]):
        """Combine domains from children, at all levels."""
        domains: dict = {}
        for child in children:
            for level in child.domains.keys():
                if child.domains[level] == []:
                    pass
                elif (
                    level not in domains
                    or domains[level] == []
                    or child.domains[level] == domains[level]
                ):
                    domains[level] = child.domains[level]
                else:
                    raise pybamm.DomainError(
                        "children must have same or empty domains, "
                        f"not {domains[level]} and {child.domains[level]}"
                    )

        return domains

    def read_domain_or_domains(
        self,
        domain: DomainType,
        auxiliary_domains: AuxiliaryDomainType,
        domains: DomainsType,
    ):
        if domains is None:
            if isinstance(domain, str):
                domain = [domain]
            elif domain is None:
                domain = []
            auxiliary_domains = auxiliary_domains or {}

            domains = {"primary": domain, **auxiliary_domains}
        else:
            if domain is not None:
                raise ValueError("Only one of 'domain' or 'domains' should be provided")
            if auxiliary_domains is not None:
                raise ValueError(
                    "Only one of 'auxiliary_domains' or 'domains' should be provided"
                )
        return domains

    @property
    def id(self):
        return self._id

    def set_id(self):
        """
        Set the immutable "identity" of a variable (e.g. for identifying y_slices).

        Hashing can be slow, so we set the id when we create the node, and hence only
        need to hash once.
        """
        self._id = hash(
            (
                self.__class__,
                self.name,
                *tuple([child.id for child in self.children]),
                *tuple([(k, tuple(v)) for k, v in self.domains.items() if v != []]),
            )
        )

    @property
    def scale(self):
        return self._scale

    @property
    def reference(self):
        return self._reference

    def __eq__(self, other):
        try:
            return self._id == other._id
        except AttributeError:
            if isinstance(other, numbers.Number):
                return self._id == pybamm.Scalar(other)._id
            else:
                return False

    def __hash__(self):
        return self._id

    @property
    def orphans(self):
        """
        Returning new copies of the children, with parents removed to avoid corrupting
        the expression tree internal data
        """
        return self._orphans

    def render(self):  # pragma: no cover
        """
        Print out a visual representation of the tree (this node and its children)
        """
        anytree = import_optional_dependency("anytree")
        for pre, _, node in anytree.RenderTree(self):
            if isinstance(node, pybamm.Scalar) and node.name != str(node.value):
                print(f"{pre}{node.name} = {node.value}")
            else:
                print(f"{pre}{node.name}")

    def visualise(self, filename: str):
        """
        Produces a .png file of the tree (this node and its children) with the
        name filename

        Parameters
        ----------

        filename : str
            filename to output, must end in ".png"
        """

        DotExporter = import_optional_dependency("anytree.exporter", "DotExporter")
        # check that filename ends in .png.
        if filename[-4:] != ".png":
            raise ValueError("filename should end in .png")

        new_node, counter = self.relabel_tree(self, 0)

        try:
            DotExporter(
                new_node, nodeattrfunc=lambda node: f'label="{node.label}"'
            ).to_picture(filename)
        except FileNotFoundError:  # pragma: no cover
            # raise error but only through logger so that test passes
            pybamm.logger.error("Please install graphviz>=2.42.2 to use dot exporter")

    def relabel_tree(self, symbol: Symbol, counter: int):
        """
        Finds all children of a symbol and assigns them a new id so that they can be
        visualised properly using the graphviz output
        """
        anytree = import_optional_dependency("anytree")
        name = symbol.name
        if name == "div":
            name = "&nabla;&sdot;"
        elif name == "grad":
            name = "&nabla;"
        elif name == "/":
            name = "&divide;"
        elif name == "*":
            name = "&times;"
        elif name == "-":
            name = "&minus;"
        elif name == "+":
            name = "&#43;"
        elif name == "**":
            name = "^"

        new_node = anytree.Node(str(counter), label=name)
        counter += 1

        new_children = []
        for child in symbol.children:
            new_child, counter = self.relabel_tree(child, counter)
            new_children.append(new_child)
        new_node.children = new_children

        return new_node, counter

    def pre_order(self):
        """
        returns an iterable that steps through the tree in pre-order fashion.

        Examples
        --------

        >>> a = pybamm.Symbol('a')
        >>> b = pybamm.Symbol('b')
        >>> for node in (a*b).pre_order():
        ...     print(node.name)
        *
        a
        b
        """
        anytree = import_optional_dependency("anytree")
        return anytree.PreOrderIter(self)

    def __str__(self):
        """return a string representation of the node and its children."""
        return self._name

    def __repr__(self):
        """returns the string `__class__(id, name, children, domain)`"""
        return f"{self.__class__.__name__!s}({hex(self.id)}, {self._name!s}, children={[str(child) for child in self.children]!s}, domains={({k: v for k, v in self.domains.items() if v != []})!s})"

    def __add__(self, other: ChildSymbol) -> pybamm.Addition:
        """return an :class:`Addition` object."""
        return pybamm.add(self, other)

    def __radd__(self, other: ChildSymbol) -> pybamm.Addition:
        """return an :class:`Addition` object."""
        return pybamm.add(other, self)

    def __sub__(self, other: ChildSymbol) -> pybamm.Subtraction:
        """return a :class:`Subtraction` object."""
        return pybamm.subtract(self, other)

    def __rsub__(self, other: ChildSymbol) -> pybamm.Subtraction:
        """return a :class:`Subtraction` object."""
        return pybamm.subtract(other, self)

    def __mul__(self, other: ChildSymbol) -> pybamm.Multiplication:
        """return a :class:`Multiplication` object."""
        return pybamm.multiply(self, other)

    def __rmul__(self, other: ChildSymbol) -> pybamm.Multiplication:
        """return a :class:`Multiplication` object."""
        return pybamm.multiply(other, self)

    def __matmul__(self, other: ChildSymbol) -> pybamm.MatrixMultiplication:
        """return a :class:`MatrixMultiplication` object."""
        return pybamm.matmul(self, other)

    def __rmatmul__(self, other: ChildSymbol) -> pybamm.MatrixMultiplication:
        """return a :class:`MatrixMultiplication` object."""
        return pybamm.matmul(other, self)

    def __truediv__(self, other: ChildSymbol) -> pybamm.Division:
        """return a :class:`Division` object."""
        return pybamm.divide(self, other)

    def __rtruediv__(self, other: ChildSymbol) -> pybamm.Division:
        """return a :class:`Division` object."""
        return pybamm.divide(other, self)

    def __pow__(self, other: ChildSymbol) -> pybamm.Power:
        """return a :class:`Power` object."""
        return pybamm.simplified_power(self, other)

    def __rpow__(self, other: Symbol) -> pybamm.Power:
        """return a :class:`Power` object."""
        return pybamm.simplified_power(other, self)

    def __lt__(self, other: Symbol | float) -> pybamm.NotEqualHeaviside:
        """return a :class:`NotEqualHeaviside` object, or a smooth approximation."""
        return pybamm.expression_tree.binary_operators._heaviside(self, other, False)

    def __le__(self, other: Symbol) -> pybamm.EqualHeaviside:
        """return a :class:`EqualHeaviside` object, or a smooth approximation."""
        return pybamm.expression_tree.binary_operators._heaviside(self, other, True)

    def __gt__(self, other: Symbol) -> pybamm.NotEqualHeaviside:
        """return a :class:`NotEqualHeaviside` object, or a smooth approximation."""
        return pybamm.expression_tree.binary_operators._heaviside(other, self, False)

    def __ge__(self, other: Symbol) -> pybamm.EqualHeaviside:
        """return a :class:`EqualHeaviside` object, or a smooth approximation."""
        return pybamm.expression_tree.binary_operators._heaviside(other, self, True)

    def __neg__(self) -> pybamm.Negate:
        """return a :class:`Negate` object."""
        if isinstance(self, pybamm.Negate):
            # Double negative is a positive
            return self.orphans[0]
        elif isinstance(self, pybamm.Broadcast):
            # Move negation inside the broadcast
            # Apply recursively
            return self.create_copy([-self.orphans[0]])
        elif isinstance(self, pybamm.Subtraction):
            # negation flips the subtraction
            return self.right - self.left
        elif isinstance(self, pybamm.Concatenation) and all(
            child.is_constant() for child in self.children
        ):
            return pybamm.concatenation(*[-child for child in self.orphans])
        else:
            return pybamm.simplify_if_constant(pybamm.Negate(self))

    def __abs__(self) -> pybamm.AbsoluteValue:
        """return an :class:`AbsoluteValue` object, or a smooth approximation."""
        if isinstance(self, pybamm.AbsoluteValue):
            # No need to apply abs a second time
            return self
        elif isinstance(self, pybamm.Broadcast):
            # Move absolute value inside the broadcast
            # Apply recursively
            abs_self_not_broad = abs(self.orphans[0])
            return self.create_copy([abs_self_not_broad])
        else:
            k = pybamm.settings.abs_smoothing
            # Return exact approximation if that is the setting or the outcome is a
            # constant (i.e. no need for smoothing)
            if k == "exact" or is_constant(self):
                out = pybamm.AbsoluteValue(self)
            else:
                out = pybamm.smooth_absolute_value(self, k)
            return pybamm.simplify_if_constant(out)

    def __mod__(self, other: Symbol) -> pybamm.Modulo:
        """return an :class:`Modulo` object."""
        return pybamm.simplify_if_constant(pybamm.Modulo(self, other))

    def __bool__(self):
        raise NotImplementedError(
            "Boolean operator not defined for Symbols. You might be seeing this message because you are trying to "
            "specify an if statement based on the value of a symbol, e.g."
            "\nif x < 0:\n"
            "\ty = 1\n"
            "else:\n"
            "\ty = 2\n"
            "In this case, use heaviside functions instead:"
            "\ny = 1 * (x < 0) + 2 * (x >= 0)"
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        If a numpy ufunc is applied to a symbol, call the corresponding pybamm function
        instead.
        """
        return getattr(pybamm, ufunc.__name__)(*inputs, **kwargs)

    def diff(self, variable: Symbol):
        """
        Differentiate a symbol with respect to a variable. For any symbol that can be
        differentiated, return `1` if differentiating with respect to yourself,
        `self._diff(variable)` if `variable` is in the expression tree of the symbol,
        and zero otherwise.

        Parameters
        ----------
        variable : :class:`pybamm.Symbol`
            The variable with respect to which to differentiate
        """
        if variable == self:
            return pybamm.Scalar(1)
        elif any(variable == x for x in self.pre_order()):
            return self._diff(variable)
        elif variable == pybamm.t and self.has_symbol_of_classes(
            (pybamm.VariableBase, pybamm.StateVectorBase)
        ):
            return self._diff(variable)
        else:
            return pybamm.Scalar(0)

    def _diff(self, variable):
        """
        Default behaviour for differentiation, overriden by Binary and Unary Operators
        """
        raise NotImplementedError

    def jac(
        self,
        variable: pybamm.Symbol,
        known_jacs: dict[pybamm.Symbol, pybamm.Symbol] | None = None,
        clear_domain=True,
    ):
        """
        Differentiate a symbol with respect to a (slice of) a StateVector
        or StateVectorDot.
        See :class:`pybamm.Jacobian`.
        """
        jac = pybamm.Jacobian(known_jacs, clear_domain=clear_domain)
        if not isinstance(variable, (pybamm.StateVector, pybamm.StateVectorDot)):
            raise TypeError(
                "Jacobian can only be taken with respect to a 'StateVector' "
                f"or 'StateVectorDot', but {variable} is a {type(variable)}"
            )
        return jac.jac(self, variable)

    def _jac(self, variable):
        """
        Default behaviour for jacobian, will raise a ``NotImplementedError``
        if this member function has not been defined for the node.
        """
        raise NotImplementedError

    def _base_evaluate(
        self,
        t: float | None = None,
        y: np.ndarray | None = None,
        y_dot: np.ndarray | None = None,
        inputs: dict | str | None = None,
    ):
        """
        evaluate expression tree.

        will raise a ``NotImplementedError`` if this member function has not
        been defined for the node. For example, :class:`Scalar` returns its
        scalar value, but :class:`Variable` will raise ``NotImplementedError``

        Parameters
        ----------

        t : float or numeric type, optional
            time at which to evaluate (default None)

        y : numpy.array, optional
            array with state values to evaluate when solving (default None)

        y_dot : numpy.array, optional
            array with time derivatives of state values to evaluate when solving
            (default None)
        """
        raise NotImplementedError(
            "method self.evaluate() not implemented for symbol "
            f"{self!s} of type {type(self)}"
        )

    def evaluate(
        self,
        t: float | None = None,
        y: np.ndarray | None = None,
        y_dot: np.ndarray | None = None,
        inputs: dict | str | None = None,
    ) -> ChildValue:
        """Evaluate expression tree (wrapper to allow using dict of known values).

        Parameters
        ----------
        t : float or numeric type, optional
            time at which to evaluate (default None)
        y : numpy.array, optional
            array with state values to evaluate when solving (default None)
        y_dot : numpy.array, optional
            array with time derivatives of state values to evaluate when solving
            (default None)
        inputs : dict, optional
            dictionary of inputs to use when solving (default None)

        Returns
        -------
        number or array
            the node evaluated at (t,y)
        """
        return self._base_evaluate(t, y, y_dot, inputs)

    def evaluate_for_shape(self):
        """
        Evaluate expression tree to find its shape.

        For symbols that cannot be evaluated directly (e.g. `Variable` or `Parameter`),
        a vector of the appropriate shape is returned instead, using the symbol's
        domain. See :meth:`pybamm.Symbol.evaluate()`
        """
        try:
            return self._saved_evaluate_for_shape
        except AttributeError:
            self._saved_evaluate_for_shape = self._evaluate_for_shape()
            return self._saved_evaluate_for_shape

    def _evaluate_for_shape(self):
        """See :meth:`Symbol.evaluate_for_shape`"""
        return self.evaluate()

    def is_constant(self):
        """
        returns true if evaluating the expression is not dependent on `t` or `y`
        or `inputs`

        See Also
        --------
        evaluate : evaluate the expression
        """
        # Default behaviour is False
        return False

    def evaluate_ignoring_errors(self, t: float | None = 0):
        """
        Evaluates the expression. If a node exists in the tree that cannot be evaluated
        as a scalar or vector (e.g. Time, Parameter, Variable, StateVector), then None
        is returned. If there is an InputParameter in the tree then a 1 is returned.
        Otherwise the result of the evaluation is given.

        See Also
        --------
        evaluate : evaluate the expression
        """
        try:
            result = self.evaluate(t=t, inputs="shape test")
        except NotImplementedError:
            # return None if NotImplementedError is raised
            # (there is a e.g. Parameter, Variable, ... in the tree)
            return None
        except TypeError as error:
            # return None if specific TypeError is raised
            # (there is a e.g. StateVector in the tree)
            if error.args[0] == "StateVector cannot evaluate input 'y=None'":
                return None
            elif error.args[0] == "StateVectorDot cannot evaluate input 'y_dot=None'":
                return None
            else:  # pragma: no cover
                raise error
        except ValueError as error:
            # return None if specific ValueError is raised
            # (there is a e.g. Time in the tree)
            if error.args[0] == "t must be provided":
                return None
            raise pybamm.ShapeError(
                f"Cannot find shape (original error: {error})"
            ) from error  # pragma: no cover
        return result

    def evaluates_to_number(self):
        """
        Returns True if evaluating the expression returns a number.
        Returns False otherwise, including if NotImplementedError or TyperError
        is raised.
        !Not to be confused with isinstance(self, pybamm.Scalar)!

        See Also
        --------
        evaluate : evaluate the expression
        """
        return self.shape_for_testing == ()

    def evaluates_to_constant_number(self):
        return self.evaluates_to_number() and self.is_constant()

    def evaluates_on_edges(self, dimension: str) -> bool:
        """
        Returns True if a symbol evaluates on an edge, i.e. symbol contains a gradient
        operator, but not a divergence operator, and is not an IndefiniteIntegral.
        Caches the solution for faster results.

        Parameters
        ----------
        dimension : str
            The dimension (primary, secondary, etc) in which to query evaluation on
            edges

        Returns
        -------
        bool
            Whether the symbol evaluates on edges (in the finite volume discretisation
            sense)
        """
        if dimension not in self._saved_evaluates_on_edges:
            self._saved_evaluates_on_edges[dimension] = self._evaluates_on_edges(
                dimension
            )

        return self._saved_evaluates_on_edges[dimension]

    def _evaluates_on_edges(self, dimension):
        # Default behaviour: return False
        return False

    def has_symbol_of_classes(
        self, symbol_classes: tuple[type[Symbol], ...] | type[Symbol]
    ):
        """
        Returns True if equation has a term of the class(es) `symbol_class`.

        Parameters
        ----------
        symbol_classes : pybamm class or iterable of classes
            The classes to test the symbol against
        """
        return any(isinstance(symbol, symbol_classes) for symbol in self.pre_order())

    def to_casadi(
        self,
        t: casadi.MX | None = None,
        y: casadi.MX | None = None,
        y_dot: casadi.MX | None = None,
        inputs: dict | None = None,
        casadi_symbols: Symbol | None = None,
    ):
        """
        Convert the expression tree to a CasADi expression tree.
        See :class:`pybamm.CasadiConverter`.
        """
        return pybamm.CasadiConverter(casadi_symbols).convert(self, t, y, y_dot, inputs)

    def _children_for_copying(self, children: list[Symbol] | None = None) -> Symbol:
        """
        Gets existing children for a symbol being copied if they aren't provided.
        """
        if children is None:
            children = [child.create_copy() for child in self.children]
        return children

    def create_copy(
        self,
        new_children: list[pybamm.Symbol] | None = None,
        perform_simplifications: bool = True,
    ):
        """
        Make a new copy of a symbol, to avoid Tree corruption errors while bypassing
        copy.deepcopy(), which is slow.

        If new_children are provided, they are used instead of the existing children.

        If `perform_simplifications` = True, some classes (e.g. `BinaryOperator`,
        `UnaryOperator`, `Concatenation`) will perform simplifications and error checks
        based on the new children before copying the symbol. This may result in a
        different symbol being returned than the one copied.

        Turning off this behaviour to ensure the symbol remains unchanged is
        discouraged.
        """
        children = self._children_for_copying(new_children)
        return self.__class__(self.name, children, domains=self.domains)

    def new_copy(
        self,
        new_children: list[Symbol] | None = None,
        perform_simplifications: bool = True,
    ):
        """ """
        warnings.warn(
            "The 'new_copy' function for expression tree symbols is deprecated, use "
            "'create_copy' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create_copy(new_children, perform_simplifications)

    @cached_property
    def size(self):
        """
        Size of an object, found by evaluating it with appropriate t and y
        """
        return np.prod(self.shape)

    @cached_property
    def shape(self):
        """
        Shape of an object, found by evaluating it with appropriate t and y.
        """
        # Default behaviour is to try to evaluate the object directly
        # Try with some large y, to avoid having to unpack (slow)
        try:
            y = np.nan * np.ones((1000, 1))
            evaluated_self = self.evaluate(0, y, y, inputs="shape test")
        # If that fails, fall back to calculating how big y should really be
        except ValueError:
            unpacker = pybamm.SymbolUnpacker(pybamm.StateVector)
            state_vectors_in_node = unpacker.unpack_symbol(self)
            min_y_size = max(
                max(len(x._evaluation_array) for x in state_vectors_in_node), 1
            )
            # Pick a y that won't cause RuntimeWarnings
            y = np.nan * np.ones((min_y_size, 1))
            evaluated_self = self.evaluate(0, y, y, inputs="shape test")

        # Return shape of evaluated object
        if isinstance(evaluated_self, numbers.Number):
            return ()
        else:
            return evaluated_self.shape

    @property
    def size_for_testing(self):
        """Size of an object, based on shape for testing."""
        return np.prod(self.shape_for_testing)

    @property
    def shape_for_testing(self):
        """
        Shape of an object for cases where it cannot be evaluated directly. If a symbol
        cannot be evaluated directly (e.g. it is a `Variable` or `Parameter`), it is
        instead given an arbitrary domain-dependent shape.
        """
        evaluated_self = self.evaluate_for_shape()
        if isinstance(evaluated_self, numbers.Number):
            return ()
        else:
            return evaluated_self.shape

    @property
    def ndim_for_testing(self):
        """
        Number of dimensions of an object,
        found by evaluating it with appropriate t and y
        """
        return len(self.shape_for_testing)

    def test_shape(self):
        """
        Check that the discretised self has a pybamm `shape`, i.e. can be evaluated.

        Raises
        ------
        pybamm.ShapeError
            If the shape of the object cannot be found
        """
        try:
            self.shape_for_testing
        except ValueError as e:
            raise pybamm.ShapeError(f"Cannot find shape (original error: {e})") from e

    @property
    def print_name(self):
        return self._print_name

    @print_name.setter
    def print_name(self, name):
        self._raw_print_name = name
        self._print_name = prettify_print_name(name)

    def to_equation(self):
        return sympy.Symbol(str(self.name))

    def to_json(self):
        """
        Method to serialise a Symbol object into JSON.
        """

        json_dict = {
            "name": self.name,
            "id": self.id,
            "domains": self.domains,
        }

        return json_dict
