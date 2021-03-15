#
# Unary operator classes and methods
#
import numbers
import numpy as np
import pybamm
from scipy.sparse import issparse, csr_matrix


class UnaryOperator(pybamm.Symbol):
    """A node in the expression tree representing a unary operator
    (e.g. '-', grad, div)

    Derived classes will specify the particular operator

    **Extends:** :class:`Symbol`

    Parameters
    ----------
    name : str
        name of the node
    child : :class:`Symbol`
        child node

    """

    def __init__(self, name, child, domain=None, auxiliary_domains=None):
        if isinstance(child, numbers.Number):
            child = pybamm.Scalar(child)
        if domain is None:
            domain = child.domain
        if auxiliary_domains is None:
            auxiliary_domains = child.auxiliary_domains
        super().__init__(
            name, children=[child], domain=domain, auxiliary_domains=auxiliary_domains
        )
        self.child = self.children[0]

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        return "{}({!s})".format(self.name, self.child)

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        new_child = self.child.new_copy()
        return self._unary_new_copy(new_child)

    def _unary_new_copy(self, child):
        """Make a new copy of the unary operator, with child `child`"""

        return self.__class__(child)

    def _unary_jac(self, child_jac):
        """Calculate the jacobian of a unary operator."""
        raise NotImplementedError

    def _unary_evaluate(self, child):
        """Perform unary operation on a child. """
        raise NotImplementedError

    def evaluate(self, t=None, y=None, y_dot=None, inputs=None, known_evals=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        if known_evals is not None:
            if self.id not in known_evals:
                child, known_evals = self.child.evaluate(
                    t, y, y_dot, inputs, known_evals
                )
                known_evals[self.id] = self._unary_evaluate(child)
            return known_evals[self.id], known_evals
        else:
            child = self.child.evaluate(t, y, y_dot, inputs)
            return self._unary_evaluate(child)

    def _evaluate_for_shape(self):
        """
        Default behaviour: unary operator has same shape as child
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        return self.children[0].evaluate_for_shape()

    def _evaluates_on_edges(self, dimension):
        """ See :meth:`pybamm.Symbol._evaluates_on_edges()`. """
        return self.child.evaluates_on_edges(dimension)

    def is_constant(self):
        """ See :meth:`pybamm.Symbol.is_constant()`. """
        return self.child.is_constant()


class Negate(UnaryOperator):
    """A node in the expression tree representing a `-` negation operator

    **Extends:** :class:`UnaryOperator`
    """

    def __init__(self, child):
        """ See :meth:`pybamm.UnaryOperator.__init__()`. """
        super().__init__("-", child)

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        return "{}{!s}".format(self.name, self.child)

    def _diff(self, variable):
        """ See :meth:`pybamm.Symbol._diff()`. """
        return -self.child.diff(variable)

    def _unary_jac(self, child_jac):
        """ See :meth:`pybamm.UnaryOperator._unary_jac()`. """
        return -child_jac

    def _unary_evaluate(self, child):
        """ See :meth:`UnaryOperator._unary_evaluate()`. """
        return -child


class AbsoluteValue(UnaryOperator):
    """A node in the expression tree representing an `abs` operator

    **Extends:** :class:`UnaryOperator`
    """

    def __init__(self, child):
        """ See :meth:`pybamm.UnaryOperator.__init__()`. """
        super().__init__("abs", child)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        child = self.child.new_copy()
        return Sign(child) * child.diff(variable)

    def _unary_jac(self, child_jac):
        """ See :meth:`pybamm.UnaryOperator._unary_jac()`. """
        child = self.child.new_copy()
        return Sign(child) * child_jac

    def _unary_evaluate(self, child):
        """ See :meth:`UnaryOperator._unary_evaluate()`. """
        return np.abs(child)


class Sign(UnaryOperator):
    """A node in the expression tree representing a `sign` operator

    **Extends:** :class:`UnaryOperator`
    """

    def __init__(self, child):
        """ See :meth:`pybamm.UnaryOperator.__init__()`. """
        super().__init__("sign", child)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        return pybamm.Scalar(0)

    def _unary_jac(self, child_jac):
        """ See :meth:`pybamm.UnaryOperator._unary_jac()`. """
        return pybamm.Scalar(0)

    def _unary_evaluate(self, child):
        """ See :meth:`UnaryOperator._unary_evaluate()`. """
        if issparse(child):
            return csr_matrix.sign(child)
        else:
            return np.sign(child)


class Floor(UnaryOperator):
    """A node in the expression tree representing an `floor` operator

    **Extends:** :class:`UnaryOperator`
    """

    def __init__(self, child):
        """ See :meth:`pybamm.UnaryOperator.__init__()`. """
        super().__init__("floor", child)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        return pybamm.Scalar(0)

    def _unary_jac(self, child_jac):
        """ See :meth:`pybamm.UnaryOperator._unary_jac()`. """
        return pybamm.Scalar(0)

    def _unary_evaluate(self, child):
        """ See :meth:`UnaryOperator._unary_evaluate()`. """
        return np.floor(child)


class Ceiling(UnaryOperator):
    """A node in the expression tree representing a `ceil` operator

    **Extends:** :class:`UnaryOperator`
    """

    def __init__(self, child):
        """ See :meth:`pybamm.UnaryOperator.__init__()`. """
        super().__init__("ceil", child)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        return pybamm.Scalar(0)

    def _unary_jac(self, child_jac):
        """ See :meth:`pybamm.UnaryOperator._unary_jac()`. """
        return pybamm.Scalar(0)

    def _unary_evaluate(self, child):
        """ See :meth:`UnaryOperator._unary_evaluate()`. """
        return np.ceil(child)


class Index(UnaryOperator):
    """A node in the expression tree, which stores the index that should be
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
                    name = "Index[:{:d}]".format(index.stop)
                else:
                    name = "Index[{:d}:{:d}]".format(index.start, index.stop)
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

    def _unary_jac(self, child_jac):
        """ See :meth:`pybamm.UnaryOperator._unary_jac()`. """

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
        """ See :meth:`pybamm.Symbol.set_id()` """
        self._id = hash(
            (
                self.__class__,
                self.name,
                self.slice.start,
                self.slice.stop,
                self.children[0].id,
            )
            + tuple(self.domain)
        )

    def _unary_evaluate(self, child):
        """ See :meth:`UnaryOperator._unary_evaluate()`. """
        return child[self.slice]

    def _unary_new_copy(self, child):
        """ See :meth:`UnaryOperator._unary_new_copy()`. """
        new_index = self.__class__(child, self.index, check_size=False)
        # Keep same domains
        new_index.copy_domains(self)
        return new_index

    def _evaluate_for_shape(self):
        return self._unary_evaluate(self.children[0].evaluate_for_shape())

    def _evaluates_on_edges(self, dimension):
        """ See :meth:`pybamm.Symbol._evaluates_on_edges()`. """
        return False


class SpatialOperator(UnaryOperator):
    """A node in the expression tree representing a unary spatial operator
    (e.g. grad, div)

    Derived classes will specify the particular operator

    This type of node will be replaced by the :class:`Discretisation`
    class with a :class:`Matrix`

    **Extends:** :class:`UnaryOperator`

    Parameters
    ----------

    name : str
        name of the node
    child : :class:`Symbol`
        child node

    """

    def __init__(self, name, child, domain=None, auxiliary_domains=None):
        super().__init__(name, child, domain, auxiliary_domains)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        # We shouldn't need this
        raise NotImplementedError


class Gradient(SpatialOperator):
    """A node in the expression tree representing a grad operator

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child):
        if child.domain == []:
            raise pybamm.DomainError(
                "Cannot take gradient of '{}' since its domain is empty. ".format(child)
                + "Try broadcasting the object first, e.g.\n\n"
                "\tpybamm.grad(pybamm.PrimaryBroadcast(symbol, 'domain'))"
            )
        if child.evaluates_on_edges("primary") is True:
            raise TypeError(
                "Cannot take gradient of '{}' since it evaluates on edges".format(child)
            )
        super().__init__("grad", child)

    def _evaluates_on_edges(self, dimension):
        """ See :meth:`pybamm.Symbol._evaluates_on_edges()`. """
        return True


class Divergence(SpatialOperator):
    """A node in the expression tree representing a div operator

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child):
        if child.domain == []:
            raise pybamm.DomainError(
                "Cannot take divergence of '{}' since its domain is empty. ".format(
                    child
                )
                + "Try broadcasting the object first, e.g.\n\n"
                "\tpybamm.div(pybamm.PrimaryBroadcast(symbol, 'domain'))"
            )
        if child.evaluates_on_edges("primary") is False:
            raise TypeError(
                "Cannot take divergence of '{}' since it does not ".format(child)
                + "evaluate on edges. Usually, a gradient should be taken before the "
                "divergence."
            )
        super().__init__("div", child)

    def _evaluates_on_edges(self, dimension):
        """ See :meth:`pybamm.Symbol._evaluates_on_edges()`. """
        return False


class Laplacian(SpatialOperator):
    """A node in the expression tree representing a laplacian operator. This is
    currently only implemeted in the weak form for finite element formulations.

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child):
        super().__init__("laplacian", child)

    def _evaluates_on_edges(self, dimension):
        """ See :meth:`pybamm.Symbol._evaluates_on_edges()`. """
        return False


class Gradient_Squared(SpatialOperator):
    """A node in the expression tree representing a the inner product of the grad
    operator with itself. In particular, this is useful in the finite element
    formualtion where we only require the (sclar valued) square of the gradient,
    and  not the gradient itself.
    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child):
        super().__init__("grad squared", child)

    def _evaluates_on_edges(self, dimension):
        """ See :meth:`pybamm.Symbol._evaluates_on_edges()`. """
        return True


class Mass(SpatialOperator):
    """Returns the mass matrix for a given symbol, accounting for Dirchlet boundary
    conditions where necessary (e.g. in the finite element formualtion)
    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child):
        super().__init__("mass", child)

    def _evaluate_for_shape(self):
        return pybamm.evaluate_for_shape_using_domain(self.domain, typ="matrix")


class BoundaryMass(SpatialOperator):
    """Returns the mass matrix for a given symbol assembled over the boundary of
    the domain, accounting for Dirchlet boundary conditions where necessary
    (e.g. in the finite element formualtion)
    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child):
        super().__init__("boundary mass", child)

    def _evaluate_for_shape(self):
        return pybamm.evaluate_for_shape_using_domain(self.domain, typ="matrix")


class Integral(SpatialOperator):
    """A node in the expression tree representing an integral operator

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

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child, integration_variable):
        if not isinstance(integration_variable, list):
            integration_variable = [integration_variable]

        name = "integral"
        for var in integration_variable:
            if isinstance(var, pybamm.SpatialVariable):
                # Check that child and integration_variable domains agree
                if var.domain == child.domain:
                    self._integration_dimension = "primary"
                elif (
                    "secondary" in child.auxiliary_domains
                    and var.domain == child.auxiliary_domains["secondary"]
                ):
                    self._integration_dimension = "secondary"
                elif (
                    "tertiary" in child.auxiliary_domains
                    and var.domain == child.auxiliary_domains["tertiary"]
                ):
                    self._integration_dimension = "tertiary"
                else:
                    raise pybamm.DomainError(
                        "integration_variable must be the same as child domain or "
                        "an auxiliary domain"
                    )
            else:
                raise TypeError(
                    "integration_variable must be of type pybamm.SpatialVariable, "
                    "not {}".format(type(var))
                )
            name += " d{}".format(var.name)

        if self._integration_dimension == "primary":
            # integral of a child takes the domain from auxiliary domain of the child
            if child.auxiliary_domains != {}:
                domain = child.auxiliary_domains["secondary"]
                if "tertiary" in child.auxiliary_domains:
                    auxiliary_domains = {
                        "secondary": child.auxiliary_domains["tertiary"]
                    }
                else:
                    auxiliary_domains = {}
            # if child has no auxiliary domain, integral removes domain
            else:
                domain = []
                auxiliary_domains = {}
        elif self._integration_dimension == "secondary":
            # integral in the secondary dimension keeps the same domain, moves tertiary
            # domain to secondary domain
            domain = child.domain
            if "tertiary" in child.auxiliary_domains:
                auxiliary_domains = {"secondary": child.auxiliary_domains["tertiary"]}
            else:
                auxiliary_domains = {}
        elif self._integration_dimension == "tertiary":
            # integral in the tertiary dimension keeps the domain and secondary domain
            domain = child.domain
            auxiliary_domains = {"secondary": child.auxiliary_domains["secondary"]}

        if any(isinstance(var, pybamm.SpatialVariable) for var in integration_variable):
            name += " {}".format(child.domain)

        self._integration_variable = integration_variable
        super().__init__(
            name, child, domain=domain, auxiliary_domains=auxiliary_domains
        )

    @property
    def integration_variable(self):
        return self._integration_variable

    def set_id(self):
        """ See :meth:`pybamm.Symbol.set_id()` """
        self._id = hash(
            (self.__class__, self.name)
            + tuple(
                [
                    integration_variable.id
                    for integration_variable in self.integration_variable
                ]
            )
            + (self.children[0].id,)
            + tuple(self.domain)
        )

    def _unary_new_copy(self, child):
        """ See :meth:`UnaryOperator._unary_new_copy()`. """

        return self.__class__(child, self.integration_variable)

    def _evaluate_for_shape(self):
        """ See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()` """
        return pybamm.evaluate_for_shape_using_domain(
            self.domain, self.auxiliary_domains
        )

    def _evaluates_on_edges(self, dimension):
        """ See :meth:`pybamm.Symbol._evaluates_on_edges()`. """
        return False


class BaseIndefiniteIntegral(Integral):
    """Base class for indefinite integrals (forward or backward).

    Parameters
    ----------
    function : :class:`pybamm.Symbol`
        The function to be integrated (will become self.children[0])
    integration_variable : :class:`pybamm.IndependentVariable`
        The variable over which to integrate

    **Extends:** :class:`Integral`
    """

    def __init__(self, child, integration_variable):
        if isinstance(integration_variable, list):
            if len(integration_variable) > 1:
                raise NotImplementedError(
                    "Indefinite integral only implemeted w.r.t. one variable"
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
    """A node in the expression tree representing an indefinite integral operator

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

    **Extends:** :class:`BaseIndefiniteIntegral`
    """

    def __init__(self, child, integration_variable):
        super().__init__(child, integration_variable)
        # Overwrite the name
        self.name = "{} integrated w.r.t {}".format(
            child.name, self.integration_variable[0].name
        )
        if isinstance(integration_variable, pybamm.SpatialVariable):
            self.name += " on {}".format(self.integration_variable[0].domain)


class BackwardIndefiniteIntegral(BaseIndefiniteIntegral):
    """A node in the expression tree representing a backward indefinite integral
    operator

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

    **Extends:** :class:`BaseIndefiniteIntegral`
    """

    def __init__(self, child, integration_variable):
        super().__init__(child, integration_variable)
        # Overwrite the name
        self.name = "{} integrated backward w.r.t {}".format(
            child.name, self.integration_variable[0].name
        )
        if isinstance(integration_variable, pybamm.SpatialVariable):
            self.name += " on {}".format(self.integration_variable[0].domain)


class DefiniteIntegralVector(SpatialOperator):
    """A node in the expression tree representing an integral of the basis used
    for discretisation

    .. math::
        I = \\int_{a}^{b}\\!\\psi(x)\\,dx,

    where :math:`a` and :math:`b` are the left-hand and right-hand boundaries of
    the domain respectively and :math:`\\psi` is the basis function.

    Parameters
    ----------
    variable : :class:`pybamm.Symbol`
        The variable whose basis will be integrated over the entire domain
    vector_type : str, optional
        Whether to return a row or column vector (default is row)

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child, vector_type="row"):
        name = "basis integral"
        self.vector_type = vector_type
        super().__init__(name, child)
        # integrating removes the domain
        self.clear_domains()

    def set_id(self):
        """ See :meth:`pybamm.Symbol.set_id()` """
        self._id = hash(
            (self.__class__, self.name, self.vector_type)
            + (self.children[0].id,)
            + tuple(self.domain)
        )

    def _unary_new_copy(self, child):
        """ See :meth:`UnaryOperator._unary_new_copy()`. """

        return self.__class__(child, vector_type=self.vector_type)

    def _evaluate_for_shape(self):
        """ See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()` """
        return pybamm.evaluate_for_shape_using_domain(self.domain)


class BoundaryIntegral(SpatialOperator):
    """A node in the expression tree representing an integral operator over the
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
    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child, region="entire"):
        # boundary integral removes domain
        domain = []
        auxiliary_domains = {}

        name = "boundary integral over "
        if region == "entire":
            name += "entire boundary"
        elif region == "negative tab":
            name += "negative tab"
        elif region == "positive tab":
            name += "positive tab"
        self.region = region
        super().__init__(
            name, child, domain=domain, auxiliary_domains=auxiliary_domains
        )

    def set_id(self):
        """ See :meth:`pybamm.Symbol.set_id()` """
        self._id = hash(
            (self.__class__, self.name) + (self.children[0].id,) + tuple(self.domain)
        )

    def _unary_new_copy(self, child):
        """ See :meth:`UnaryOperator._unary_new_copy()`. """

        return self.__class__(child, region=self.region)

    def _evaluate_for_shape(self):
        """ See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()` """
        return pybamm.evaluate_for_shape_using_domain(self.domain)

    def _evaluates_on_edges(self, dimension):
        """ See :meth:`pybamm.Symbol._evaluates_on_edges()`. """
        return False


class DeltaFunction(SpatialOperator):
    """Delta function. Currently can only be implemented at the edge of a domain

    Parameters
    ----------
    child : :class:`pybamm.Symbol`
        The variable that sets the strength of the delta function
    side : str
        Which side of the domain to implement the delta function on

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child, side, domain):
        self.side = side
        if domain is None:
            raise pybamm.DomainError("Delta function domain cannot be None")
        if child.domain != []:
            auxiliary_domains = {"secondary": child.domain}
        else:
            auxiliary_domains = {}
        super().__init__("delta_function", child, domain, auxiliary_domains)

    def set_id(self):
        """ See :meth:`pybamm.Symbol.set_id()` """
        self._id = hash(
            (self.__class__, self.name, self.side, self.children[0].id)
            + tuple(self.domain)
            + tuple([(k, tuple(v)) for k, v in self.auxiliary_domains.items()])
        )

    def _evaluates_on_edges(self, dimension):
        """ See :meth:`pybamm.Symbol._evaluates_on_edges()`. """
        return False

    def _unary_new_copy(self, child):
        """ See :meth:`UnaryOperator._unary_new_copy()`. """
        return self.__class__(child, self.side, self.domain)

    def evaluate_for_shape(self):
        """
        See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`
        """
        child_eval = self.children[0].evaluate_for_shape()
        vec = pybamm.evaluate_for_shape_using_domain(self.domain)

        return np.outer(child_eval, vec).reshape(-1, 1)


class BoundaryOperator(SpatialOperator):
    """A node in the expression tree which gets the boundary value of a variable.

    Parameters
    ----------
    name : str
        The name of the symbol
    child : :class:`pybamm.Symbol`
        The variable whose boundary value to take
    side : str
        Which side to take the boundary value on ("left" or "right")

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, name, child, side):
        # side can only be "negative tab" or "positive tab" if domain is
        # "current collector"
        if side in ["negative tab", "positive tab"]:
            if child.domain[0] != "current collector":
                raise pybamm.ModelError(
                    """Can only take boundary value on the tabs in the domain
                'current collector', but {} has domain {}""".format(
                        child, child.domain[0]
                    )
                )
        self.side = side
        # boundary value of a child takes the domain from auxiliary domain of the child
        if child.auxiliary_domains != {}:
            domain = child.auxiliary_domains["secondary"]
        # if child has no auxiliary domain, boundary operator removes domain
        else:
            domain = []
        # tertiary auxiliary domain shift down to secondary
        try:
            auxiliary_domains = {"secondary": child.auxiliary_domains["tertiary"]}
        except KeyError:
            auxiliary_domains = {}
        super().__init__(
            name, child, domain=domain, auxiliary_domains=auxiliary_domains
        )

    def set_id(self):
        """ See :meth:`pybamm.Symbol.set_id()` """
        self._id = hash(
            (self.__class__, self.name, self.side, self.children[0].id)
            + tuple(self.domain)
            + tuple([(k, tuple(v)) for k, v in self.auxiliary_domains.items()])
        )

    def _unary_new_copy(self, child):
        """ See :meth:`UnaryOperator._unary_new_copy()`. """
        return self.__class__(child, self.side)

    def _evaluate_for_shape(self):
        """ See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()` """
        return pybamm.evaluate_for_shape_using_domain(
            self.domain, self.auxiliary_domains
        )


class BoundaryValue(BoundaryOperator):
    """A node in the expression tree which gets the boundary value of a variable.

    Parameters
    ----------
    child : :class:`pybamm.Symbol`
        The variable whose boundary value to take
    side : str
        Which side to take the boundary value on ("left" or "right")

    **Extends:** :class:`BoundaryOperator`
    """

    def __init__(self, child, side):
        super().__init__("boundary value", child, side)

    def _unary_new_copy(self, child):
        """ See :meth:`UnaryOperator._unary_new_copy()`. """
        return boundary_value(child, self.side)


class BoundaryGradient(BoundaryOperator):
    """A node in the expression tree which gets the boundary flux of a variable.

    Parameters
    ----------
    child : :class:`pybamm.Symbol`
        The variable whose boundary flux to take
    side : str
        Which side to take the boundary flux on ("left" or "right")

    **Extends:** :class:`BoundaryOperator`
    """

    def __init__(self, child, side):
        super().__init__("boundary flux", child, side)


class UpwindDownwind(SpatialOperator):
    """A node in the expression tree representing an upwinding or downwinding operator.
    Usually to be used for better stability in convection-dominated equations.

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, name, child):
        if child.domain == []:
            raise pybamm.DomainError(
                "Cannot upwind '{}' since its domain is empty. ".format(child)
                + "Try broadcasting the object first, e.g.\n\n"
                "\tpybamm.div(pybamm.PrimaryBroadcast(symbol, 'domain'))"
            )
        if child.evaluates_on_edges("primary") is True:
            raise TypeError(
                "Cannot upwind '{}' since it does not ".format(child)
                + "evaluate on nodes."
            )
        super().__init__(name, child)

    def _evaluates_on_edges(self, dimension):
        """ See :meth:`pybamm.Symbol._evaluates_on_edges()`. """
        return True


class Upwind(UpwindDownwind):
    """
    Upwinding operator. To be used if flow velocity is positive (left to right).

    **Extends:** :class:`UpwindDownwind`
    """

    def __init__(self, child):
        super().__init__("upwind", child)


class Downwind(UpwindDownwind):
    """
    Downwinding operator. To be used if flow velocity is negative (right to left).

    **Extends:** :class:`UpwindDownwind`
    """

    def __init__(self, child):
        super().__init__("downwind", child)


class NotConstant(UnaryOperator):
    """Special class to wrap a symbol that should not be treated as a constant"""

    def __init__(self, child):
        super().__init__("not_constant", child)

    def _unary_new_copy(self, child):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return NotConstant(child)

    def _diff(self, variable):
        """ See :meth:`pybamm.Symbol._diff()`. """
        return self.child.diff(variable)

    def _unary_jac(self, child_jac):
        """ See :meth:`pybamm.UnaryOperator._unary_jac()`. """
        return child_jac

    def _unary_evaluate(self, child):
        """ See :meth:`UnaryOperator._unary_evaluate()`. """
        return child

    def is_constant(self):
        """ See :meth:`pybamm.Symbol.is_constant()`. """
        # This symbol is not constant
        return False


#
# Methods to call Gradient, Divergence, Laplacian and Gradient_Squared
#


def grad(symbol):
    """convenience function for creating a :class:`Gradient`

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
        new_child = pybamm.PrimaryBroadcast(0, symbol.child.domain)
        return pybamm.PrimaryBroadcastToEdges(new_child, symbol.domain)
    else:
        return Gradient(symbol)


def div(symbol):
    """convenience function for creating a :class:`Divergence`

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
        new_child = pybamm.PrimaryBroadcast(0, symbol.child.domain)
        return pybamm.PrimaryBroadcast(new_child, symbol.domain)
    else:
        return Divergence(symbol)


def laplacian(symbol):
    """convenience function for creating a :class:`Laplacian`

    Parameters
    ----------

    symbol : :class:`Symbol`
        the laplacian will be performed on this sub-symbol

    Returns
    -------

    :class:`Laplacian`
        the laplacian of ``symbol``
    """

    return Laplacian(symbol)


def grad_squared(symbol):
    """convenience function for creating a :class:`Gradient_Squared`

    Parameters
    ----------

    symbol : :class:`Symbol`
        the inner product of the gradient with itself will be performed on this
        sub-symbol

    Returns
    -------

    :class:`Gradient_Squared`
        inner product of the gradient of ``symbol`` with itself
    """

    return Gradient_Squared(symbol)


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
    """convenience function for creating a right :class:`BoundaryValue`, usually in the
    spherical geometry

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


#
# Methods for averaging
#


def x_average(symbol):
    """
    convenience function for creating an average in the x-direction

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The function to be averaged

    Returns
    -------
    :class:`Symbol`
        the new averaged symbol
    """
    # Can't take average if the symbol evaluates on edges
    if symbol.evaluates_on_edges("primary"):
        raise ValueError("Can't take the x-average of a symbol that evaluates on edges")
    # If symbol doesn't have a domain, its average value is itself
    if symbol.domain in [[], ["current collector"]]:
        new_symbol = symbol.new_copy()
        new_symbol.parent = None
        return new_symbol
    # If symbol is a Broadcast, its average value is its child
    elif isinstance(symbol, pybamm.Broadcast):
        return symbol.orphans[0]
    # If symbol is a concatenation of Broadcasts, its average value is its child
    elif (
        isinstance(symbol, pybamm.Concatenation)
        and all(isinstance(child, pybamm.Broadcast) for child in symbol.children)
        and symbol.domain == ["negative electrode", "separator", "positive electrode"]
    ):
        a, b, c = [orp.orphans[0] for orp in symbol.orphans]
        if a.id == b.id == c.id:
            out = a
        else:
            geo = pybamm.geometric_parameters
            l_n = geo.l_n
            l_s = geo.l_s
            l_p = geo.l_p
            out = (l_n * a + l_s * b + l_p * c) / (l_n + l_s + l_p)
        # To respect domains we may need to broadcast the child back out
        child = symbol.children[0]
        # If symbol being returned doesn't have empty domain, return it
        if out.domain != []:
            return out
        # Otherwise we may need to broadcast it
        elif child.auxiliary_domains == {}:
            return out
        else:
            domain = child.auxiliary_domains["secondary"]
            if "tertiary" not in child.auxiliary_domains:
                return pybamm.PrimaryBroadcast(out, domain)
            else:
                auxiliary_domains = {"secondary": child.auxiliary_domains["tertiary"]}
                return pybamm.FullBroadcast(out, domain, auxiliary_domains)
    # Otherwise, use Integral to calculate average value
    else:
        geo = pybamm.geometric_parameters
        # Even if domain is "negative electrode", "separator", or
        # "positive electrode", and we know l, we still compute it as Integral(1, x)
        # as this will be easier to identify for simplifications later on
        if symbol.domain == ["negative particle"]:
            x = pybamm.standard_spatial_vars.x_n
            l = geo.l_n
        elif symbol.domain == ["positive particle"]:
            x = pybamm.standard_spatial_vars.x_p
            l = geo.l_p
        else:
            x = pybamm.SpatialVariable("x", domain=symbol.domain)
            v = pybamm.ones_like(symbol)
            l = pybamm.Integral(v, x)
        return Integral(symbol, x) / l


def z_average(symbol):
    """convenience function for creating an average in the z-direction

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The function to be averaged

    Returns
    -------
    :class:`Symbol`
        the new averaged symbol
    """
    # Can't take average if the symbol evaluates on edges
    if symbol.evaluates_on_edges("primary"):
        raise ValueError("Can't take the z-average of a symbol that evaluates on edges")
    # Symbol must have domain [] or ["current collector"]
    if symbol.domain not in [[], ["current collector"]]:
        raise pybamm.DomainError(
            """z-average only implemented in the 'current collector' domain,
            but symbol has domains {}""".format(
                symbol.domain
            )
        )
    # If symbol doesn't have a domain, its average value is itself
    if symbol.domain == []:
        new_symbol = symbol.new_copy()
        new_symbol.parent = None
        return new_symbol
    # If symbol is a Broadcast, its average value is its child
    elif isinstance(symbol, pybamm.Broadcast):
        return symbol.orphans[0]
    # Otherwise, use Integral to calculate average value
    else:
        # We compute the length as Integral(1, z) as this will be easier to identify
        # for simplifications later on and it gives the correct behaviour when using
        # ZeroDimensionalSpatialMethod
        z = pybamm.standard_spatial_vars.z
        v = pybamm.ones_like(symbol)
        l = pybamm.Integral(v, z)
        return Integral(symbol, z) / l


def yz_average(symbol):
    """convenience function for creating an average in the y-z-direction

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The function to be averaged

    Returns
    -------
    :class:`Symbol`
        the new averaged symbol
    """
    # Symbol must have domain [] or ["current collector"]
    if symbol.domain not in [[], ["current collector"]]:
        raise pybamm.DomainError(
            """y-z-average only implemented in the 'current collector' domain,
            but symbol has domains {}""".format(
                symbol.domain
            )
        )
    # If symbol doesn't have a domain, its average value is itself
    if symbol.domain == []:
        new_symbol = symbol.new_copy()
        new_symbol.parent = None
        return new_symbol
    # If symbol is a Broadcast, its average value is its child
    elif isinstance(symbol, pybamm.Broadcast):
        return symbol.orphans[0]
    # Otherwise, use Integral to calculate average value
    else:
        # We compute the area as Integral(1, [y,z]) as this will be easier to identify
        # for simplifications later on and it gives the correct behaviour when using
        # ZeroDimensionalSpatialMethod
        y = pybamm.standard_spatial_vars.y
        z = pybamm.standard_spatial_vars.z
        v = pybamm.ones_like(symbol)
        A = pybamm.Integral(v, [y, z])
        return Integral(symbol, [y, z]) / A


def r_average(symbol):
    """convenience function for creating an average in the r-direction

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The function to be averaged

    Returns
    -------
    :class:`Symbol`
        the new averaged symbol
    """
    # Can't take average if the symbol evaluates on edges
    if symbol.evaluates_on_edges("primary"):
        raise ValueError("Can't take the r-average of a symbol that evaluates on edges")
    # Otherwise, if symbol doesn't have a particle domain,
    # its r-averaged value is itself
    elif symbol.domain not in [
        ["positive particle"],
        ["negative particle"],
        ["working particle"],
    ]:
        new_symbol = symbol.new_copy()
        new_symbol.parent = None
        return new_symbol
    # If symbol is a secondary broadcast onto "negative electrode" or
    # "positive electrode", take the r-average of the child then broadcast back
    elif isinstance(symbol, pybamm.SecondaryBroadcast) and symbol.domains[
        "secondary"
    ] in [["positive electrode"], ["negative electrode"], ["working electrode"]]:
        child = symbol.orphans[0]
        child_av = pybamm.r_average(child)
        return pybamm.PrimaryBroadcast(child_av, symbol.domains["secondary"])
    # If symbol is a Broadcast onto a particle domain, its average value is its child
    elif isinstance(symbol, pybamm.PrimaryBroadcast) and symbol.domain in [
        ["positive particle"],
        ["negative particle"],
        ["working particle"],
    ]:
        return symbol.orphans[0]
    else:
        r = pybamm.SpatialVariable("r", symbol.domain)
        v = pybamm.FullBroadcast(
            pybamm.Scalar(1), symbol.domain, symbol.auxiliary_domains
        )
        return Integral(symbol, r) / Integral(v, r)


def boundary_value(symbol, side):
    """convenience function for creating a :class:`pybamm.BoundaryValue`

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
    # If symbol doesn't have a domain, its boundary value is itself
    if symbol.domain == []:
        new_symbol = symbol.new_copy()
        new_symbol.parent = None
        return new_symbol
    # If symbol is a primary or full broadcast, its boundary value is its child
    if isinstance(symbol, (pybamm.PrimaryBroadcast, pybamm.FullBroadcast)):
        return symbol.orphans[0]
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


def sign(symbol):
    """ Returns a :class:`Sign` object. """
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
