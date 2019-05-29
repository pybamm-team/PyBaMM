#
# Unary operator classes and methods
#
import autograd
import numpy as np
import pybamm
from inspect import signature
from scipy.sparse import csr_matrix


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

    def __init__(self, name, child):
        super().__init__(name, children=[child], domain=child.domain)
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

    def _unary_simplify(self, simplified_child):
        """
        Simplify a unary operator. Default behaviour is to make a new copy, with
        simplified child.
        """

        return self._unary_new_copy(simplified_child)

    def _unary_evaluate(self, child):
        """Perform unary operation on a child. """
        raise NotImplementedError

    def evaluate(self, t=None, y=None, known_evals=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        if known_evals is not None:
            if self.id not in known_evals:
                child, known_evals = self.child.evaluate(t, y, known_evals)
                known_evals[self.id] = self._unary_evaluate(child)
            return known_evals[self.id], known_evals
        else:
            child = self.child.evaluate(t, y)
            return self._unary_evaluate(child)


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

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            return -self.child.diff(variable)

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        return -self.child.jac(variable)

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
        # Derivative is not well-defined
        raise NotImplementedError("Derivative of absolute function is not defined")

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        # Derivative is not well-defined
        raise NotImplementedError("Derivative of absolute function is not defined")

    def _unary_evaluate(self, child):
        """ See :meth:`UnaryOperator._unary_evaluate()`. """
        return np.abs(child)


class Function(UnaryOperator):
    """A node in the expression tree representing an arbitrary function

    Parameters
    ----------
    func : method
        A function that takes either 0 or 1 parameters. If func takes no parameters,
        self.evaluate() return func(). Otherwise, self.evaluate(t,y) returns
        func(child.evaluate(t,y))
    child : :class:`pybamm.Symbol`
        The child node to apply the function to

    **Extends:** :class:`UnaryOperator`
    """

    def __init__(self, func, child):
        """ See :meth:`pybamm.UnaryOperator.__init__()`. """
        self.func = func
        super().__init__("function ({})".format(func.__name__), child)
        # hack to work out whether function takes any params
        # (signature doesn't work for numpy)
        if isinstance(func, np.ufunc):
            self.takes_no_params = False
        else:
            self.takes_no_params = len(signature(func).parameters) == 0

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            child = self.orphans[0]
            if variable.id in [symbol.id for symbol in child.pre_order()]:
                # if variable appears in the function,use autograd to differentiate
                # function, and apply chain rule
                return child.diff(variable) * Function(autograd.grad(self.func), child)
            else:
                # otherwise the derivative of the function is zero
                return pybamm.Scalar(0)

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        child = self.orphans[0]
        if child.evaluates_to_number():
            # return zeros of right size
            variable_y_indices = np.arange(
                variable.y_slice.start, variable.y_slice.stop
            )
            jac = csr_matrix((1, np.size(variable_y_indices)))
            return pybamm.Matrix(jac)
        else:
            jac_fun = Function(autograd.elementwise_grad(self.func), child) * child.jac(
                variable
            )
            jac_fun.domain = self.domain
            return jac_fun

    def _unary_evaluate(self, child):
        """ See :meth:`UnaryOperator._unary_evaluate()`. """
        if self.takes_no_params:
            return self.func()
        else:
            return self.func(child)

    def _unary_new_copy(self, child):
        """ See :meth:`UnaryOperator._unary_new_copy()`. """
        return pybamm.Function(self.func, child)

    def _unary_simplify(self, simplified_child):
        """ See :meth:`UnaryOperator._unary_simplify()`. """
        if self.takes_no_params:
            # If self.func() takes no parameters then we can always simplify it
            return pybamm.Scalar(self.func())
        else:
            return pybamm.Function(self.func, simplified_child)


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
    """

    def __init__(self, child, index, name=None):
        self.index = index
        if isinstance(index, int):
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

        if self.slice.stop > child.size:
            raise ValueError("slice size exceeds child size")

        super().__init__(name, child)

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        child_jac = self.child.jac(variable)
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

        return self.__class__(child, self.index)


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

    def __init__(self, name, child):
        super().__init__(name, child)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        # We shouldn't need this
        raise NotImplementedError

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        raise NotImplementedError

    def _unary_simplify(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """

        # if there are none of these nodes in the child tree, then this expression
        # does not depend on space, and therefore the spatial operator result is zero
        search_types = (pybamm.Variable, pybamm.StateVector, pybamm.SpatialVariable)

        # do the search, return a scalar zero node if no relevent nodes are found
        if all([not (isinstance(n, search_types)) for n in self.pre_order()]):
            return pybamm.Scalar(0)
        else:
            return self.__class__(child)


class Gradient(SpatialOperator):
    """A node in the expression tree representing a grad operator

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child):
        super().__init__("grad", child)


class Divergence(SpatialOperator):
    """A node in the expression tree representing a div operator

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child):
        super().__init__("div", child)


class Laplacian(SpatialOperator):
    """A node in the expression tree representing a laplacian operator. This is
    currently only implemeted in the weak form for finite element formulations.

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child):
        super().__init__("laplacian", child)


class Source(SpatialOperator):
    """A node in the expression tree representing a source term in the (weak)
    finite element formulation

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child):
        super().__init__("source", child)


class Integral(SpatialOperator):
    """A node in the expression tree representing an integral operator

    .. math::
        I = \\int_{a}^{b}\\!f(u)\\,du,

    where :math:`a` and :math:`b` are the left-hand and right-hand boundaries of
    the domain respectively, and :math:`u\\in\\text{domain}`.
    Can be integration with respect to time or space.

    Parameters
    ----------
    function : :class:`pybamm.Symbol`
        The function to be integrated (will become self.children[0])
    integration_variable : :class:`pybamm.IndependentVariable`
        The variable over which to integrate

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child, integration_variable):
        name = "integral"
        for var in integration_variable:
            if isinstance(var, pybamm.SpatialVariable):
                # Check that child and integration_variable domains agree
                if child.domain != var.domain:
                    raise pybamm.DomainError(
                        """child and integration_variable must have the same domain"""
                    )
            elif not isinstance(var, pybamm.IndependentVariable):
                raise ValueError(
                    """integration_variable must be of type pybamm.IndependentVariable,
                       not {}""".format(
                        type(var)
                    )
                )
            name += " d{}".format(var.name)
            if isinstance(var, pybamm.SpatialVariable):
                name += " {}".format(var.domain)

        self._integration_variable = integration_variable
        super().__init__(name, child)
        # integrating removes the domain
        self.domain = []

    @property
    def integration_variable(self):
        return self._integration_variable

    def set_id(self):
        """ See :meth:`pybamm.Symbol.set_id()` """
        self._id = hash(
            (self.__class__, self.name)
            + tuple([integration_variable.id for integration_variable in self.integration_variable])
            + (self.child.id,)
            + tuple(self.domain)
        )

    def _unary_simplify(self, simplified_child):
        """ See :meth:`UnaryOperator._unary_simplify()`. """

        return self.__class__(simplified_child, self.integration_variable)

    def _unary_new_copy(self, child):
        """ See :meth:`UnaryOperator._unary_new_copy()`. """

        return self.__class__(child, self.integration_variable)


class IndefiniteIntegral(Integral):
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

    **Extends:** :class:`Integral`
    """

    def __init__(self, child, integration_variable):
        super().__init__(child, integration_variable)
        # Overwrite the name
        self.name = "{} integrated w.r.t {}".format(
            child.name, integration_variable.name
        )
        if isinstance(integration_variable, pybamm.SpatialVariable):
            self.name += " on {}".format(integration_variable.domain)
        # the integrated variable has the same domain as the child
        self.domain = child.domain


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
        self.side = side
        # Domain of Boundary must be ([]) so that expressions can be formed
        # of boundary values of variables in different domains
        super().__init__(name, child)
        self.domain = []

    def set_id(self):
        """ See :meth:`pybamm.Symbol.set_id()` """
        self._id = hash(
            (self.__class__, self.name, self.side, self.children[0].id)
            + tuple(self.domain)
        )

    def _unary_simplify(self, simplified_child):
        """ See :meth:`UnaryOperator._unary_simplify()`. """
        return self.__class__(simplified_child, self.side)

    def _unary_new_copy(self, child):
        """ See :meth:`UnaryOperator._unary_new_copy()`. """
        return self.__class__(child, self.side)


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


class BoundaryFlux(BoundaryOperator):
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


#
# Methods to call Gradient and Divergence
#


def grad(expression):
    """convenience function for creating a :class:`Gradient`

    Parameters
    ----------

    expression : :class:`Symbol`
        the gradient will be performed on this sub-expression

    Returns
    -------

    :class:`Gradient`
        the gradient of ``expression``
    """

    return Gradient(expression)


def div(expression):
    """convenience function for creating a :class:`Divergence`

    Parameters
    ----------

    expression : :class:`Symbol`
        the divergence will be performed on this sub-expression

    Returns
    -------

    :class:`Divergence`
        the divergence of ``expression``
    """

    return Divergence(expression)


def laplacian(expression):
    """convenience function for creating a :class:`Laplacian`

    Parameters
    ----------

    expression : :class:`Symbol`
        the laplacian will be performed on this sub-expression

    Returns
    -------

    :class:`Laplacian`
        the laplacian of ``expression``
    """

    return Laplacian(expression)


#
# Method to call SurfaceValue
#

def surf(variable, set_domain=False):
    """convenience function for creating a right :class:`BoundaryValue`, usually in the
    spherical geometry

    Parameters
    ----------

    variable : :class:`Symbol`
        the surface value of this variable will be returned

    Returns
    -------

    :class:`GetSurfaceValue`
        the surface value of ``variable``
    """
    out = boundary_value(variable, "right")
    if set_domain:
        if variable.domain == ["negative particle"]:
            out.domain = ["negative electrode"]
        elif variable.domain == ["positive particle"]:
            out.domain = ["positive electrode"]
    return out


def average(symbol):
    """convenience function for creating an average

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The function to be averaged

    Returns
    -------
    :class:`Symbol`
        the new averaged symbol
    """
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
        l_n = pybamm.geometric_parameters.l_n
        l_s = pybamm.geometric_parameters.l_s
        l_p = pybamm.geometric_parameters.l_p
        a, b, c = [orp.orphans[0] for orp in symbol.orphans]
        return (l_n * a + l_s * b + l_p * c) / (l_n + l_s + l_p)
    # Otherwise, use Integral to calculate average value
    else:
        if symbol.domain == ["negative electrode"]:
            x = pybamm.standard_spatial_vars.x_n
            l = pybamm.geometric_parameters.l_n
        elif symbol.domain == ["separator"]:
            x = pybamm.standard_spatial_vars.x_s
            l = pybamm.geometric_parameters.l_s
        elif symbol.domain == ["positive electrode"]:
            x = pybamm.standard_spatial_vars.x_p
            l = pybamm.geometric_parameters.l_p
        elif symbol.domain == ["negative electrode", "separator", "positive electrode"]:
            x = pybamm.standard_spatial_vars.x
            l = pybamm.Scalar(1)
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(symbol.domain))

        return Integral(symbol, x) / l


def boundary_value(symbol, side):
    """convenience function for creating a :class:`Integral`

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
    # If symbol is a Broadcast, its boundary value is its child
    if isinstance(symbol, pybamm.Broadcast):
        return symbol.orphans[0]
    # Otherwise, calculate boundary value
    else:
        return BoundaryValue(symbol, side)


#
# Method to call Source
#

def source(expression):
    """convenience function for creating a :class:`Source`

    Parameters
    ----------

    expression : :class:`Symbol`
        this sub-expression will be converted into weak form by multiplication
        with the mass matrix

    Returns
    -------

    :class:`Source`
        the weak form of the source ``expression``
    """

    return Source(expression)
