#
# Unary operator classes and methods
#
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, issparse
import sympy
import pybamm
from pybamm.util import import_optional_dependency
from pybamm.type_definitions import DomainsType


class UnaryOperator(pybamm.Symbol):
    """
    A node in the expression tree representing a unary operator
    (e.g. '-', grad, div)

    Derived classes will specify the particular operator

    Parameters
    ----------
    name : str
        name of the node
    child : :class:`Symbol`
        child node
    domains : dict
        A dictionary equivalent to {'primary': domain, auxiliary_domains}.
    """

    def __init__(
        self,
        name: str,
        child: pybamm.Symbol,
        domains: DomainsType = None,
    ):
        if isinstance(child, (float, int, np.number)):
            child = pybamm.Scalar(child)
        domains = domains or child.domains

        super().__init__(name, children=[child], domains=domains)
        self.child = self.children[0]

    @classmethod
    def _from_json(cls, snippet: dict):
        """Use to instantiate when deserialising"""

        instance = cls.__new__(cls)

        super(UnaryOperator, instance).__init__(
            snippet["name"],
            snippet["children"],
            domains=snippet["domains"],
        )
        instance.child = instance.children[0]

        return instance

    def __str__(self):
        """See :meth:`pybamm.Symbol.__str__()`."""
        return f"{self.name}({self.child!s})"

    def create_copy(
        self,
        new_children: list[pybamm.Symbol] | None = None,
        perform_simplifications: bool = True,
    ):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        if new_children and len(new_children) > 1:
            raise ValueError(
                f"Unary operator of type {type(self)} must have exactly one child."
            )
        child = self._children_for_copying(new_children)[0]

        new_symbol = self._unary_new_copy(child, perform_simplifications)
        new_symbol.copy_domains(self)
        return new_symbol

    def _unary_new_copy(self, child, perform_simplifications=True):
        """Make a new copy of the unary operator, with child `child`"""
        return self.__class__(child)

    def _unary_jac(self, child_jac):
        """Calculate the jacobian of a unary operator."""
        raise NotImplementedError

    def _unary_evaluate(self, child):
        """Perform unary operation on a child."""
        raise NotImplementedError(
            f"{self.__class__} does not implement _unary_evaluate."
        )

    def evaluate(
        self,
        t: float | None = None,
        y: np.ndarray | None = None,
        y_dot: np.ndarray | None = None,
        inputs: dict | str | None = None,
    ):
        """See :meth:`pybamm.Symbol.evaluate()`."""
        child = self.child.evaluate(t, y, y_dot, inputs)
        return self._unary_evaluate(child)

    def _evaluate_for_shape(self):
        """
        Default behaviour: unary operator has same shape as child
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        return self.children[0].evaluate_for_shape()

    def _evaluates_on_edges(self, dimension: str) -> bool:
        """See :meth:`pybamm.Symbol._evaluates_on_edges()`."""
        return self.child.evaluates_on_edges(dimension)

    def is_constant(self):
        """See :meth:`pybamm.Symbol.is_constant()`."""
        return self.child.is_constant()

    def _sympy_operator(self, child):
        """Apply appropriate SymPy operators."""
        return self._unary_evaluate(child)

    def to_equation(self):
        """Convert the node and its subtree into a SymPy equation."""
        if self.print_name is not None:
            return sympy.Symbol(self.print_name)
        else:
            eq1 = self.child.to_equation()
            return self._sympy_operator(eq1)


class Negate(UnaryOperator):
    """
    A node in the expression tree representing a `-` negation operator.
    """

    def __init__(self, child):
        """See :meth:`pybamm.UnaryOperator.__init__()`."""
        super().__init__("-", child)

    def __str__(self):
        """See :meth:`pybamm.Symbol.__str__()`."""
        return f"{self.name}{self.child!s}"

    def _diff(self, variable: pybamm.Symbol):
        """See :meth:`pybamm.Symbol._diff()`."""
        return -self.child.diff(variable)

    def _unary_jac(self, child_jac):
        """See :meth:`pybamm.UnaryOperator._unary_jac()`."""
        return -child_jac

    def _unary_evaluate(self, child):
        """See :meth:`UnaryOperator._unary_evaluate()`."""
        return -child

    def _unary_new_copy(self, child, perform_simplifications: bool = True):
        """
        Creates a new copy of the operator with the child `child`.

        Uses the overridden :meth:`__neg__` to cover scenarios where the child
        is some specific symbol types.
        """
        if perform_simplifications:
            return -child
        else:
            return Negate(child)


class AbsoluteValue(UnaryOperator):
    """
    A node in the expression tree representing an `abs` operator.
    """

    def __init__(self, child):
        """See :meth:`pybamm.UnaryOperator.__init__()`."""
        super().__init__("abs", child)

    def diff(self, variable):
        """See :meth:`pybamm.Symbol.diff()`."""
        return sign(self.child) * self.child.diff(variable)

    def _unary_jac(self, child_jac):
        """See :meth:`pybamm.UnaryOperator._unary_jac()`."""
        return sign(self.child) * child_jac

    def _unary_evaluate(self, child):
        """See :meth:`UnaryOperator._unary_evaluate()`."""
        return np.abs(child)

    def _unary_new_copy(self, child, perform_simplifications: bool = True):
        """
        Creates a new copy of the operator with the child `child`.

        Uses the overridden :meth:`__abs__` to cover scenarios where the child
        is some specific symbol types.
        """
        if perform_simplifications:
            return abs(child)
        else:
            return AbsoluteValue(child)


class Sign(UnaryOperator):
    """
    A node in the expression tree representing a `sign` operator.
    """

    def __init__(self, child):
        """See :meth:`pybamm.UnaryOperator.__init__()`."""
        super().__init__("sign", child)

    @classmethod
    def _from_json(cls, snippet: dict):
        raise NotImplementedError()

    def diff(self, variable):
        """See :meth:`pybamm.Symbol.diff()`."""
        return pybamm.Scalar(0)

    def _unary_jac(self, child_jac):
        """See :meth:`pybamm.UnaryOperator._unary_jac()`."""
        return pybamm.Scalar(0)

    def _unary_evaluate(self, child):
        """See :meth:`UnaryOperator._unary_evaluate()`."""
        if issparse(child):
            return csr_matrix.sign(child)
        else:
            with np.errstate(invalid="ignore"):
                return np.sign(child)

    def _unary_new_copy(self, child, perform_simplifications: bool = True):
        """
        Creates a new copy of the operator with the child `child`.

        Uses the convenience function :meth:`sign` to cover scenarios where the child is
        a concatenation or broadcast, and simplifies the symbol.
        """
        if perform_simplifications:
            return sign(child)
        else:
            return Sign(child)


class Floor(UnaryOperator):
    """
    A node in the expression tree representing an `floor` operator.
    """

    def __init__(self, child):
        """See :meth:`pybamm.UnaryOperator.__init__()`."""
        super().__init__("floor", child)

    def diff(self, variable):
        """See :meth:`pybamm.Symbol.diff()`."""
        return pybamm.Scalar(0)

    def _unary_jac(self, child_jac):
        """See :meth:`pybamm.UnaryOperator._unary_jac()`."""
        return pybamm.Scalar(0)

    def _unary_evaluate(self, child):
        """See :meth:`UnaryOperator._unary_evaluate()`."""
        return np.floor(child)


class Ceiling(UnaryOperator):
    """
    A node in the expression tree representing a `ceil` operator.
    """

    def __init__(self, child):
        """See :meth:`pybamm.UnaryOperator.__init__()`."""
        super().__init__("ceil", child)

    def diff(self, variable):
        """See :meth:`pybamm.Symbol.diff()`."""
        return pybamm.Scalar(0)

    def _unary_jac(self, child_jac):
        """See :meth:`pybamm.UnaryOperator._unary_jac()`."""
        return pybamm.Scalar(0)

    def _unary_evaluate(self, child):
        """See :meth:`UnaryOperator._unary_evaluate()`."""
        return np.ceil(child)


class Index(UnaryOperator):
    """
    A node in the expression tree, which stores the index that should be
    extracted from its child after the child has been evaluated.

    Parameters
    ----------
    child : :class:`pybamm.Symbol`
        The symbol of which to take the index
    index : int or slice
        The index (if int) or indices (if slice) to extract from the symbol
    name : str, optional
        The name of the symbol
    check_size : bool, optional
        Whether to check if the slice size exceeds the child size. Default is True.
        This should always be True when creating a new symbol so that the appropriate
        check is performed, but should be False for creating a new copy to avoid
        unnecessarily repeating the check.
    """

    def __init__(self, child, index, name=None, check_size=True):
        self.index = index
        if index == -1:
            self.slice = slice(-1, None)
            if name is None:
                name = "Index[-1]"
        elif isinstance(index, int):
            self.slice = slice(index, index + 1)
            if name is None:
                name = "Index[" + str(index) + "]"
        elif isinstance(index, slice):
            self.slice = index
            if name is None:
                if index.start is None:
                    name = f"Index[:{index.stop:d}]"
                else:
                    name = f"Index[{index.start:d}:{index.stop:d}]"
        else:
            raise TypeError("index must be integer or slice")

        if check_size:
            if self.slice in (slice(0, 1), slice(-1, None)):
                pass
            elif self.slice.stop > child.size:
                raise ValueError("slice size exceeds child size")

        super().__init__(name, child)

        # no domain for integer value key
        if isinstance(index, int):
            self.clear_domains()

    @classmethod
    def _from_json(cls, snippet: dict):
        """See :meth:`pybamm.UnaryOperator._from_json()`."""
        index = slice(
            snippet["index"]["start"],
            snippet["index"]["stop"],
            snippet["index"]["step"],
        )

        return cls(
            snippet["children"][0],
            index,
            name=snippet["name"],
            check_size=snippet["check_size"],
        )

    def _unary_jac(self, child_jac):
        """See :meth:`pybamm.UnaryOperator._unary_jac()`."""

        # if child.jac returns a matrix of zeros, this subsequently gives a bug
        # when trying to simplify the node Index(child_jac). Instead, search the
        # tree for StateVectors and return a matrix of zeros of the correct size
        # if none are found.
        if not self.has_symbol_of_classes(pybamm.StateVector):
            jac = csr_matrix((1, child_jac.shape[1]))
            return pybamm.Matrix(jac)
        else:
            return Index(child_jac, self.index)

    def set_id(self):
        """See :meth:`pybamm.Symbol.set_id()`"""
        self._id = hash(
            (
                self.__class__,
                self.name,
                self.slice.start,
                self.slice.stop,
                self.children[0].id,
                *tuple(self.domain),
            )
        )

    def _unary_evaluate(self, child):
        """See :meth:`UnaryOperator._unary_evaluate()`."""
        return child[self.slice]

    def _unary_new_copy(self, child, perform_simplifications=True):
        """See :meth:`UnaryOperator._unary_new_copy()`."""
        # this
        new_index = self.__class__(child, self.index, check_size=False)
        # Keep same domains
        new_index.copy_domains(self)
        return new_index

    def _evaluate_for_shape(self):
        return self._unary_evaluate(self.children[0].evaluate_for_shape())

    def _evaluates_on_edges(self, dimension: str) -> bool:
        """See :meth:`pybamm.Symbol._evaluates_on_edges()`."""
        return False

    def to_json(self):
        """
        Method to serialise an Index object into JSON.
        """

        json_dict = {
            "name": self.name,
            "id": self.id,
            "index": {
                "start": self.slice.start,
                "stop": self.slice.stop,
                "step": self.slice.step,
            },
            "check_size": False,
        }

        return json_dict


class SpatialOperator(UnaryOperator):
    """
    A node in the expression tree representing a unary spatial operator
    (e.g. grad, div)

    Derived classes will specify the particular operator

    This type of node will be replaced by the :class:`Discretisation`
    class with a :class:`Matrix`

    Parameters
    ----------

    name : str
        name of the node
    child : :class:`Symbol`
        child node
    domains : dict
        A dictionary equivalent to {'primary': domain, auxiliary_domains}.
    """

    def __init__(
        self,
        name: str,
        child: pybamm.Symbol,
        domains: dict[str, list[str] | str] | None = None,
    ):
        super().__init__(name, child, domains)

    def to_json(self):
        raise NotImplementedError(
            "pybamm.SpatialOperator:"
            "Serialisation is only implemented for discretised models."
        )

    @classmethod
    def _from_json(cls, snippet):
        raise NotImplementedError(
            "pybamm.SpatialOperator:"
            "Please use a discretised model when reading in from JSON."
        )


class Gradient(SpatialOperator):
    """
    A node in the expression tree representing a grad operator.
    """

    def __init__(self, child):
        if child.domain == []:
            raise pybamm.DomainError(
                f"Cannot take gradient of '{child}' since its domain is empty. "
                + "Try broadcasting the object first, e.g.\n\n"
                "\tpybamm.grad(pybamm.PrimaryBroadcast(symbol, 'domain'))"
            )
        if child.evaluates_on_edges("primary") is True:
            raise TypeError(
                f"Cannot take gradient of '{child}' since it evaluates on edges"
            )
        super().__init__("grad", child)

    def _evaluates_on_edges(self, dimension: str) -> bool:
        """See :meth:`pybamm.Symbol._evaluates_on_edges()`."""
        return True

    def _unary_new_copy(self, child, perform_simplifications: bool = True):
        """
        Creates a new copy of the operator with the child `child`.

        Uses the convenience function :meth:`grad` to cover scenarios where the gradient
        is zero, or the child is a broadcast object.
        """
        if perform_simplifications:
            return grad(child)
        else:
            return Gradient(child)

    def _sympy_operator(self, child):
        """Override :meth:`pybamm.UnaryOperator._sympy_operator`"""
        sympy_Gradient = import_optional_dependency(
            "sympy.vector.operators", "Gradient"
        )
        return sympy_Gradient(child)


class Divergence(SpatialOperator):
    """
    A node in the expression tree representing a div operator.
    """

    def __init__(self, child):
        if child.domain == []:
            raise pybamm.DomainError(
                f"Cannot take divergence of '{child}' since its domain is empty. "
                + "Try broadcasting the object first, e.g.\n\n"
                "\tpybamm.div(pybamm.PrimaryBroadcast(symbol, 'domain'))"
            )
        if child.evaluates_on_edges("primary") is False:
            raise TypeError(
                f"Cannot take divergence of '{child}' since it does not "
                + "evaluate on edges. Usually, a gradient should be taken before the "
                "divergence."
            )
        super().__init__("div", child)

    def _evaluates_on_edges(self, dimension: str) -> bool:
        """See :meth:`pybamm.Symbol._evaluates_on_edges()`."""
        return False

    def _unary_new_copy(self, child, perform_simplifications: bool = True):
        """
        Creates a new copy of the operator with the child `child`.

        Uses the convenience function :meth:`div` to cover scenarios where divergence is
        0 or interacts with other functions.
        """
        if perform_simplifications:
            return div(child)
        else:
            return Divergence(child)

    def _sympy_operator(self, child):
        """Override :meth:`pybamm.UnaryOperator._sympy_operator`"""
        sympy_Divergence = import_optional_dependency(
            "sympy.vector.operators", "Divergence"
        )
        return sympy_Divergence(child)


class Laplacian(SpatialOperator):
    """
    A node in the expression tree representing a Laplacian operator. This is
    currently only implemeted in the weak form for finite element formulations.
    """

    def __init__(self, child):
        super().__init__("laplacian", child)

    def _evaluates_on_edges(self, dimension: str) -> bool:
        """See :meth:`pybamm.Symbol._evaluates_on_edges()`."""
        return False


class GradientSquared(SpatialOperator):
    """
    A node in the expression tree representing a the inner product of the grad
    operator with itself. In particular, this is useful in the finite element
    formualtion where we only require the (sclar valued) square of the gradient,
    and not the gradient itself.
    """

    def __init__(self, child):
        super().__init__("grad squared", child)

    def _evaluates_on_edges(self, dimension: str) -> bool:
        """See :meth:`pybamm.Symbol._evaluates_on_edges()`."""
        return False


class Mass(SpatialOperator):
    """
    Returns the mass matrix for a given symbol, accounting for Dirchlet boundary
    conditions where necessary (e.g. in the finite element formualtion)
    """

    def __init__(self, child):
        super().__init__("mass", child)

    def _evaluate_for_shape(self):
        return pybamm.evaluate_for_shape_using_domain(self.domains, typ="matrix")


class BoundaryMass(SpatialOperator):
    """
    Returns the mass matrix for a given symbol assembled over the boundary of
    the domain, accounting for Dirchlet boundary conditions where necessary
    (e.g. in the finite element formualtion)
    """

    def __init__(self, child):
        super().__init__("boundary mass", child)

    def _evaluate_for_shape(self):
        return pybamm.evaluate_for_shape_using_domain(self.domains, typ="matrix")


class Integral(SpatialOperator):
    """
    A node in the expression tree representing an integral operator.

    .. math::
        I = \\int_{a}^{b}\\!f(u)\\,du,

    where :math:`a` and :math:`b` are the left-hand and right-hand boundaries of
    the domain respectively, and :math:`u\\in\\text{domain}`.

    Parameters
    ----------
    function : :class:`pybamm.Symbol`
        The function to be integrated (will become self.children[0])
    integration_variable : :class:`pybamm.IndependentVariable`
        The variable over which to integrate
    """

    def __init__(
        self,
        child,
        integration_variable: (
            list[pybamm.IndependentVariable] | pybamm.IndependentVariable
        ),
    ):
        if not isinstance(integration_variable, list):
            integration_variable = [integration_variable]

        name = "integral"
        for var in integration_variable:
            if isinstance(var, pybamm.SpatialVariable):
                # Check that child and integration_variable domains agree
                if var.domain == child.domain:
                    self._integration_dimension = "primary"
                elif var.domain == child.domains["secondary"]:
                    self._integration_dimension = "secondary"
                elif var.domain == child.domains["tertiary"]:
                    self._integration_dimension = "tertiary"
                elif var.domain == child.domains["quaternary"]:
                    self._integration_dimension = "quaternary"
                else:
                    raise pybamm.DomainError(
                        "integration_variable must be the same as child domain or "
                        "an auxiliary domain"
                    )
            else:
                raise TypeError(
                    "integration_variable must be of type pybamm.SpatialVariable, "
                    f"not {type(var)}"
                )
            name += f" d{var.name}"

        if self._integration_dimension == "primary":
            # integral of a child takes the domain from auxiliary domain of the child
            domains = {
                "primary": child.domains["secondary"],
                "secondary": child.domains["tertiary"],
                "tertiary": child.domains["quaternary"],
            }
        elif self._integration_dimension == "secondary":
            # integral in the secondary dimension keeps the same domain, moves
            # quaternary to tertiary and tertiary to secondary domain
            domains = {
                "primary": child.domains["primary"],
                "secondary": child.domains["tertiary"],
                "tertiary": child.domains["quaternary"],
            }
        elif self._integration_dimension == "tertiary":
            # integral in the tertiary dimension keeps the domain and secondary domain,
            # moves quaternary to tertiary
            domains = {
                "primary": child.domains["primary"],
                "secondary": child.domains["secondary"],
                "tertiary": child.domains["quaternary"],
            }
        elif self._integration_dimension == "quaternary":
            # integral in the quaternary dimension keeps the domain, secondary and
            # tertiary domains
            domains = {
                "primary": child.domains["primary"],
                "secondary": child.domains["secondary"],
                "tertiary": child.domains["tertiary"],
            }
        if any(isinstance(var, pybamm.SpatialVariable) for var in integration_variable):
            name += f" {child.domain}"

        self._integration_variable = integration_variable
        super().__init__(name, child, domains)

    @property
    def integration_variable(self):
        return self._integration_variable

    def set_id(self):
        """See :meth:`pybamm.Symbol.set_id()`"""
        self._id = hash(
            (
                self.__class__,
                self.name,
                *tuple(
                    [
                        integration_variable.id
                        for integration_variable in self.integration_variable
                    ]
                ),
                self.children[0].id,
                *tuple(self.domain),
            )
        )

    def _unary_new_copy(self, child, perform_simplifications=True):
        """See :meth:`UnaryOperator._unary_new_copy()`."""
        return self.__class__(child, self.integration_variable)

    def _evaluate_for_shape(self):
        """See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`"""
        return pybamm.evaluate_for_shape_using_domain(self.domains)

    def _evaluates_on_edges(self, dimension: str) -> bool:
        """See :meth:`pybamm.Symbol._evaluates_on_edges()`."""
        return False

    def _sympy_operator(self, child):
        """Override :meth:`pybamm.UnaryOperator._sympy_operator`"""
        return sympy.Integral(child, sympy.Symbol("xn"))


class BaseIndefiniteIntegral(Integral):
    """
    Base class for indefinite integrals (forward or backward).

    Parameters
    ----------
    function : :class:`pybamm.Symbol`
        The function to be integrated (will become self.children[0])
    integration_variable : :class:`pybamm.IndependentVariable`
        The variable over which to integrate
    """

    def __init__(self, child, integration_variable):
        if isinstance(integration_variable, list):
            if len(integration_variable) > 1:
                raise NotImplementedError(
                    "Indefinite integral only implemented w.r.t. one variable"
                )
            else:
                integration_variable = integration_variable[0]
        super().__init__(child, integration_variable)
        # overwrite domains with child domains
        self.copy_domains(child)

    def _evaluate_for_shape(self):
        return self.children[0].evaluate_for_shape()

    def _evaluates_on_edges(self, dimension):
        # If child evaluates on edges, indefinite integral doesn't
        # If child doesn't evaluate on edges, indefinite integral does
        return not self.child.evaluates_on_edges(dimension)


class IndefiniteIntegral(BaseIndefiniteIntegral):
    """
    A node in the expression tree representing an indefinite integral operator.

    .. math::
        I = \\int_{x_\text{min}}^{x}\\!f(u)\\,du

    where :math:`u\\in\\text{domain}` which can represent either a
    spatial or temporal variable.

    Parameters
    ----------
    function : :class:`pybamm.Symbol`
        The function to be integrated (will become self.children[0])
    integration_variable : :class:`pybamm.IndependentVariable`
        The variable over which to integrate
    """

    def __init__(self, child, integration_variable):
        super().__init__(child, integration_variable)
        # Overwrite the name
        self.name = f"{child.name} integrated w.r.t {self.integration_variable[0].name}"
        if isinstance(integration_variable, pybamm.SpatialVariable):
            self.name += f" on {self.integration_variable[0].domain}"


class BackwardIndefiniteIntegral(BaseIndefiniteIntegral):
    """
    A node in the expression tree representing a backward indefinite integral operator.

    .. math::
        I = \\int_{x}^{x_\text{max}}\\!f(u)\\,du

    where :math:`u\\in\\text{domain}` which can represent either a
    spatial or temporal variable.

    Parameters
    ----------
    function : :class:`pybamm.Symbol`
        The function to be integrated (will become self.children[0])
    integration_variable : :class:`pybamm.IndependentVariable`
        The variable over which to integrate
    """

    def __init__(self, child, integration_variable):
        super().__init__(child, integration_variable)
        # Overwrite the name
        self.name = f"{child.name} integrated backward w.r.t {self.integration_variable[0].name}"
        if isinstance(integration_variable, pybamm.SpatialVariable):
            self.name += f" on {self.integration_variable[0].domain}"


class DefiniteIntegralVector(SpatialOperator):
    """
    A node in the expression tree representing an integral of the basis used
    for discretisation

    .. math::
        I = \\int_{a}^{b}\\!\\psi(x)\\,dx,

    where :math:`a` and :math:`b` are the left-hand and right-hand boundaries of
    the domain respectively and :math:`\\psi` is the basis function.

    Parameters
    ----------
    variable : :class:`pybamm.Symbol`
        The variable whose basis will be integrated over the entire domain (will
        become self.children[0])
    vector_type : str, optional
        Whether to return a row or column vector (default is row)
    """

    def __init__(self, child, vector_type="row"):
        name = "basis integral"
        self.vector_type = vector_type
        super().__init__(name, child)
        # integrating removes the domain
        self.clear_domains()

    def set_id(self):
        """See :meth:`pybamm.Symbol.set_id()`"""
        self._id = hash(
            (
                self.__class__,
                self.name,
                self.vector_type,
                self.children[0].id,
                *tuple(self.domain),
            )
        )

    def _unary_new_copy(self, child, perform_simplifications=True):
        """See :meth:`UnaryOperator._unary_new_copy()`."""
        return self.__class__(child, vector_type=self.vector_type)

    def _evaluate_for_shape(self):
        """See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`"""
        return pybamm.evaluate_for_shape_using_domain(self.domains)


class BoundaryIntegral(SpatialOperator):
    """
    A node in the expression tree representing an integral operator over the
    boundary of a domain

    .. math::
        I = \\int_{\\partial a}\\!f(u)\\,du,

    where :math:`\\partial a` is the boundary of the domain, and
    :math:`u\\in\\text{domain boundary}`.

    Parameters
    ----------
    function : :class:`pybamm.Symbol`
        The function to be integrated (will become self.children[0])
    region : str, optional
        The region of the boundary over which to integrate. If region is `entire`
        (default) the integration is carried out over the entire boundary. If
        region is `negative tab` or `positive tab` then the integration is only
        carried out over the appropriate part of the boundary corresponding to
        the tab.
    """

    def __init__(self, child, region="entire"):
        # boundary integral removes domains
        domains = {}

        name = "boundary integral over "
        if region == "entire":
            name += "entire boundary"
        elif region == "negative tab":
            name += "negative tab"
        elif region == "positive tab":
            name += "positive tab"
        self.region = region
        super().__init__(name, child, domains)

    def set_id(self):
        """See :meth:`pybamm.Symbol.set_id()`"""
        self._id = hash(
            (self.__class__, self.name, self.children[0].id, *tuple(self.domain))
        )

    def _unary_new_copy(self, child, perform_simplifications=True):
        """See :meth:`UnaryOperator._unary_new_copy()`."""
        return self.__class__(child, region=self.region)

    def _evaluate_for_shape(self):
        """See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`"""
        return pybamm.evaluate_for_shape_using_domain(self.domains)

    def _evaluates_on_edges(self, dimension: str) -> bool:
        """See :meth:`pybamm.Symbol._evaluates_on_edges()`."""
        return False


class DeltaFunction(SpatialOperator):
    """
    Delta function. Currently can only be implemented at the edge of a domain.

    Parameters
    ----------
    child : :class:`pybamm.Symbol`
        The variable that sets the strength of the delta function
    side : str
        Which side of the domain to implement the delta function on
    """

    def __init__(self, child, side, domain):
        self.side = side
        if domain is None:
            raise pybamm.DomainError("Delta function domain cannot be None")
        domains = {"primary": domain}
        if child.domain != []:
            domains["secondary"] = child.domain
        super().__init__("delta_function", child, domains)

    def set_id(self):
        """See :meth:`pybamm.Symbol.set_id()`"""
        self._id = hash(
            (
                self.__class__,
                self.name,
                self.side,
                self.children[0].id,
                *tuple([(k, tuple(v)) for k, v in self.domains.items()]),
            )
        )

    def _evaluates_on_edges(self, dimension: str) -> bool:
        """See :meth:`pybamm.Symbol._evaluates_on_edges()`."""
        return False

    def _unary_new_copy(self, child, perform_simplifications=True):
        """See :meth:`UnaryOperator._unary_new_copy()`."""
        return self.__class__(child, self.side, self.domain)

    def evaluate_for_shape(self):
        """See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`"""
        child_eval = self.children[0].evaluate_for_shape()
        vec = pybamm.evaluate_for_shape_using_domain(self.domains)

        return np.outer(child_eval, vec).reshape(-1, 1)


class BoundaryOperator(SpatialOperator):
    """
    A node in the expression tree which gets the boundary value of a variable on its
    primary domain.

    Parameters
    ----------
    name : str
        The name of the symbol
    child : :class:`pybamm.Symbol`
        The variable whose boundary value to take
    side : str
        Which side to take the boundary value on ("left" or "right")
    """

    def __init__(self, name, child, side):
        # side can only be "negative tab" or "positive tab" if domain is
        # "current collector"
        if side in ["negative tab", "positive tab"]:
            if child.domain[0] != "current collector":
                raise pybamm.ModelError(
                    f"""Can only take boundary value on the tabs in the domain
                'current collector', but {child} has domain {child.domain[0]}"""
                )
        self.side = side
        # boundary value of a child takes the primary domain from secondary domain
        # of the child
        # tertiary auxiliary domain shift down to secondary, quarternary to tertiary
        domains = {
            "primary": child.domains["secondary"],
            "secondary": child.domains["tertiary"],
            "tertiary": child.domains["quaternary"],
        }
        super().__init__(name, child, domains)

    def set_id(self):
        """See :meth:`pybamm.Symbol.set_id()`"""
        self._id = hash(
            (
                self.__class__,
                self.name,
                self.side,
                self.children[0].id,
                *tuple([(k, tuple(v)) for k, v in self.domains.items()]),
            )
        )

    def _unary_new_copy(self, child, perform_simplifications=True):
        """See :meth:`UnaryOperator._unary_new_copy()`."""
        return self.__class__(child, self.side)

    def _evaluate_for_shape(self):
        """See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`"""
        return pybamm.evaluate_for_shape_using_domain(self.domains)


class BoundaryValue(BoundaryOperator):
    """
    A node in the expression tree which gets the boundary value of a variable on its
    primary domain.

    Parameters
    ----------
    child : :class:`pybamm.Symbol`
        The variable whose boundary value to take
    side : str
        Which side to take the boundary value on ("left" or "right")
    """

    def __init__(self, child, side):
        super().__init__("boundary value", child, side)

    def _unary_new_copy(self, child, perform_simplifications: bool = True):
        """
        Creates a new copy of the operator with the child `child`.

        Uses the convenience function :meth:`boundary_value` to perform checks before
        creating a BoundaryValue object.
        """
        if perform_simplifications:
            return boundary_value(child, self.side)
        else:
            return BoundaryValue(child, self.side)

    def _sympy_operator(self, child):
        """Override :meth:`pybamm.UnaryOperator._sympy_operator`"""
        if (
            self.child.domain[0] in ["negative particle", "positive particle"]
            and self.side == "right"
        ):
            # value on the surface of the particle
            if str(child) == "1":
                return child
            else:
                latex_child = sympy.latex(child) + r"^{surf}"
                return sympy.Symbol(latex_child)

        elif self.side == "positive tab":
            return child

        else:
            latex_child = sympy.latex(child) + r"^{" + sympy.latex(self.side) + r"}"
            return sympy.Symbol(latex_child)


class ExplicitTimeIntegral(UnaryOperator):
    def __init__(self, children, initial_condition):
        super().__init__("explicit time integral", children)
        self.initial_condition = initial_condition

    @classmethod
    def _from_json(cls, snippet: dict):
        return cls(snippet["children"][0], snippet["initial_condition"])

    def _unary_new_copy(self, child, perform_simplifications=True):
        return self.__class__(child, self.initial_condition)

    def is_constant(self):
        return False

    def to_json(self):
        """
        Convert ExplicitTimeIntegral to json for serialisation.

        Both `children` and `initial_condition` contain Symbols, and are therefore
        dealt with by `pybamm.Serialise._SymbolEncoder.default()` directly.
        """
        json_dict = {
            "name": self.name,
            "id": self.id,
        }

        return json_dict


class BoundaryGradient(BoundaryOperator):
    """
    A node in the expression tree which gets the boundary flux of a variable on its
    primary domain.

    Parameters
    ----------
    child : :class:`pybamm.Symbol`
        The variable whose boundary flux to take
    side : str
        Which side to take the boundary flux on ("left" or "right")
    """

    def __init__(self, child, side):
        super().__init__("boundary flux", child, side)


class EvaluateAt(SpatialOperator):
    """
    A node in the expression tree which evaluates a symbol at a given position in space
    in its primary domain. Currently this is only implemented for 1D primary domains.

    Parameters
    ----------
    child : :class:`pybamm.Symbol`
        The variable to evaluate
    position : :class:`pybamm.Symbol`
        The position in space on the symbol's primary domain at which to evaluate
        the symbol.
    """

    def __init__(self, child, position):
        self.position = position

        # "evaluate at" of a child takes the primary domain from secondary domain
        # of the child
        # tertiary auxiliary domain shift down to secondary, quarternary to tertiary
        domains = {
            "primary": child.domains["secondary"],
            "secondary": child.domains["tertiary"],
            "tertiary": child.domains["quaternary"],
        }

        super().__init__("evaluate", child, domains)

    def set_id(self):
        """See :meth:`pybamm.Symbol.set_id()`"""
        self._id = hash(
            (
                self.__class__,
                self.name,
                self.position,
                self.children[0].id,
                *tuple([(k, tuple(v)) for k, v in self.domains.items()]),
            )
        )

    def _unary_jac(self, child_jac):
        """See :meth:`pybamm.UnaryOperator._unary_jac()`."""
        return pybamm.Scalar(0)

    def _unary_new_copy(self, child, perform_simplifications=True):
        """See :meth:`UnaryOperator._unary_new_copy()`."""
        return self.__class__(child, self.position)

    def _evaluate_for_shape(self):
        """See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`"""
        return pybamm.evaluate_for_shape_using_domain(self.domains)


class UpwindDownwind(SpatialOperator):
    """
    A node in the expression tree representing an upwinding or downwinding operator.
    Usually to be used for better stability in convection-dominated equations.
    """

    def __init__(self, name, child):
        if child.domain == []:
            raise pybamm.DomainError(
                f"Cannot upwind '{child}' since its domain is empty. "
                + "Try broadcasting the object first, e.g.\n\n"
                "\tpybamm.div(pybamm.PrimaryBroadcast(symbol, 'domain'))"
            )
        if child.evaluates_on_edges("primary") is True:
            raise TypeError(
                f"Cannot upwind '{child}' since it does not " + "evaluate on nodes."
            )
        super().__init__(name, child)

    def _evaluates_on_edges(self, dimension: str) -> bool:
        """See :meth:`pybamm.Symbol._evaluates_on_edges()`."""
        return True


class Upwind(UpwindDownwind):
    """
    Upwinding operator. To be used if flow velocity is positive (left to right).
    """

    def __init__(self, child):
        super().__init__("upwind", child)


class Downwind(UpwindDownwind):
    """
    Downwinding operator. To be used if flow velocity is negative (right to left).
    """

    def __init__(self, child):
        super().__init__("downwind", child)


class NotConstant(UnaryOperator):
    """Special class to wrap a symbol that should not be treated as a constant."""

    def __init__(self, child):
        super().__init__("not_constant", child)

    def _diff(self, variable: pybamm.Symbol):
        """See :meth:`pybamm.Symbol._diff()`."""
        return self.child.diff(variable)

    def _unary_jac(self, child_jac):
        """See :meth:`pybamm.UnaryOperator._unary_jac()`."""
        return child_jac

    def _unary_evaluate(self, child):
        """See :meth:`UnaryOperator._unary_evaluate()`."""
        return child

    def is_constant(self):
        """See :meth:`pybamm.Symbol.is_constant()`."""
        # This symbol is not constant
        return False


#
# Methods to call Gradient, Divergence, Laplacian and GradientSquared
#


def grad(symbol):
    """
    convenience function for creating a :class:`Gradient`

    Parameters
    ----------

    symbol : :class:`Symbol`
        the gradient will be performed on this sub-symbol

    Returns
    -------

    :class:`Gradient`
        the gradient of ``symbol``
    """
    # Gradient of a broadcast is zero
    if isinstance(symbol, pybamm.PrimaryBroadcast):
        if symbol.child.domain == []:
            new_child = pybamm.Scalar(0)
        else:
            new_child = pybamm.PrimaryBroadcast(0, symbol.child.domain)
        return pybamm.PrimaryBroadcastToEdges(new_child, symbol.domain)
    elif isinstance(symbol, pybamm.SecondaryBroadcast):
        # Take gradient of the child
        # then broadcast back to the originalsymbol's secondary domain
        # We can do this because gradient only acts on the primary domain
        return pybamm.SecondaryBroadcast(grad(symbol.child), symbol.secondary_domain)
    elif isinstance(symbol, pybamm.FullBroadcast):
        return pybamm.FullBroadcastToEdges(0, broadcast_domains=symbol.domains)
    else:
        return Gradient(symbol)


def div(symbol):
    """
    convenience function for creating a :class:`Divergence`

    Parameters
    ----------

    symbol : :class:`Symbol`
        the divergence will be performed on this sub-symbol

    Returns
    -------

    :class:`Divergence`
        the divergence of ``symbol``
    """
    # Divergence of a broadcast is zero
    if isinstance(symbol, pybamm.PrimaryBroadcastToEdges):
        if symbol.child.domain == []:
            new_child = pybamm.Scalar(0)
        else:
            new_child = pybamm.PrimaryBroadcast(0, symbol.child.domain)
        return pybamm.PrimaryBroadcast(new_child, symbol.domain)
    # Divergence commutes with Negate operator
    if isinstance(symbol, pybamm.Negate):
        return -div(symbol.orphans[0])
    elif isinstance(symbol, (pybamm.Multiplication, pybamm.Division)):
        left, right = symbol.orphans
        if isinstance(left, pybamm.Negate):
            return -div(symbol._binary_new_copy(left.orphans[0], right))

    # Last resort
    return Divergence(symbol)


def laplacian(symbol):
    """
    convenience function for creating a :class:`Laplacian`

    Parameters
    ----------

    symbol : :class:`Symbol`
        the Laplacian will be performed on this sub-symbol

    Returns
    -------

    :class:`Laplacian`
        the Laplacian of ``symbol``
    """

    return Laplacian(symbol)


def grad_squared(symbol):
    """
    convenience function for creating a :class:`GradientSquared`

    Parameters
    ----------

    symbol : :class:`Symbol`
        the inner product of the gradient with itself will be performed on this
        sub-symbol

    Returns
    -------

    :class:`GradientSquared`
        inner product of the gradient of ``symbol`` with itself
    """

    return GradientSquared(symbol)


def upwind(symbol):
    """convenience function for creating a :class:`Upwind`"""
    return Upwind(symbol)


def downwind(symbol):
    """convenience function for creating a :class:`Downwind`"""
    return Downwind(symbol)


#
# Method to call SurfaceValue
#


def surf(symbol):
    """
    convenience function for creating a right :class:`BoundaryValue`, usually in the
    spherical geometry.

    Parameters
    ----------

    symbol : :class:`pybamm.Symbol`
        the surface value of this symbol will be returned

    Returns
    -------
    :class:`pybamm.BoundaryValue`
        the surface value of ``symbol``
    """
    return boundary_value(symbol, "right")


def boundary_value(symbol, side):
    """
    convenience function for creating a :class:`pybamm.BoundaryValue`

    Parameters
    ----------
    symbol : `pybamm.Symbol`
        The symbol whose boundary value to take
    side : str
        Which side to take the boundary value on ("left" or "right")

    Returns
    -------
    :class:`BoundaryValue`
        the new integrated expression tree
    """
    # Can't take boundary value if the symbol evaluates on edges
    if symbol.evaluates_on_edges("primary"):
        raise ValueError(
            "Can't take the boundary value of a symbol that evaluates on edges"
        )

    # If symbol doesn't have a domain, its boundary value is itself
    if symbol.domain == []:
        return symbol
    # If symbol is a primary or full broadcast, reduce by one dimension
    if isinstance(symbol, (pybamm.PrimaryBroadcast, pybamm.FullBroadcast)):
        return symbol.reduce_one_dimension()
    # If symbol is a secondary broadcast, its boundary value is a primary broadcast of
    # the boundary value of its child
    if isinstance(symbol, pybamm.SecondaryBroadcast):
        # Read child (making copy)
        child = symbol.orphans[0]
        # Take boundary value
        boundary_child = boundary_value(child, side)
        # Broadcast back to the original symbol's secondary domain
        return pybamm.PrimaryBroadcast(boundary_child, symbol.secondary_domain)
    # Otherwise, calculate boundary value
    else:
        return BoundaryValue(symbol, side)


def boundary_gradient(symbol, side):
    # Gradient of a broadcast is zero
    if isinstance(symbol, pybamm.Broadcast):
        return 0 * symbol.reduce_one_dimension()
    else:
        return BoundaryGradient(symbol, side)


def sign(symbol):
    """Returns a :class:`Sign` object."""
    if isinstance(symbol, pybamm.Broadcast):
        # Move sign inside the broadcast
        # Apply recursively
        return symbol._unary_new_copy(sign(symbol.orphans[0]))
    elif isinstance(symbol, pybamm.Concatenation) and not isinstance(
        symbol, pybamm.ConcatenationVariable
    ):
        return pybamm.concatenation(*[sign(child) for child in symbol.orphans])
    return pybamm.simplify_if_constant(Sign(symbol))


def smooth_absolute_value(symbol, k):
    """
    Smooth approximation to the absolute value function. k is the smoothing parameter,
    set by `pybamm.settings.abs_smoothing`. The recommended value is k=10.
    """
    x = symbol
    exp = pybamm.exp
    kx = k * symbol
    return x * (exp(kx) - exp(-kx)) / (exp(kx) + exp(-kx))
