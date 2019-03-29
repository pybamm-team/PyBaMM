#
# Unary operator classes and methods
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import autograd
import numpy as np
from scipy.sparse import diags


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

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        return "{}({!s})".format(self.name, self.children[0])


class Negate(UnaryOperator):
    """A node in the expression tree representing a `-` negation operator

    **Extends:** :class:`UnaryOperator`
    """

    def __init__(self, child):
        """ See :meth:`pybamm.UnaryOperator.__init__()`. """
        super().__init__("-", child)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return -self.children[0].evaluate(t, y)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            return -self.children[0].diff(variable)

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            return -self.children[0].jac(variable)

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        return "{}{!s}".format(self.name, self.children[0])


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

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return np.abs(self.children[0].evaluate(t, y))


class Function(UnaryOperator):
    """A node in the expression tree representing an arbitrary function

    **Extends:** :class:`UnaryOperator`
    """

    def __init__(self, func, child):
        """ See :meth:`pybamm.UnaryOperator.__init__()`. """
        super().__init__("function ({})".format(func.__name__), child)
        self.func = func

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
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            child = self.orphans[0]
            if variable.id in [symbol.id for symbol in child.pre_order()]:
                # if variable appears in the function,use autograd to differentiate
                # function, and apply chain rule
                return child.jac(variable) * Function(autograd.jacobian(self.func), child)
            else:
                # otherwise the derivative of the function is zero
                return pybamm.Scalar(0)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.func(self.children[0].evaluate(t, y))


class Diagonal(UnaryOperator):
    """A node in the expression tree representing an operator which creates a
    diagonal matrix from a vector

    **Extends:** :class:`UnaryOperator`
    """

    def __init__(self, child):
        """ See :meth:`pybamm.UnaryOperator.__init__()`. """
        super().__init__("diag", child)

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        # We shouldn't need this
        raise NotImplementedError

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        # We shouldn't need this
        raise NotImplementedError

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return diags(self.children[0].evaluate(t, y), 0)


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
        # We shouldn't need this
        raise NotImplementedError


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
        if isinstance(integration_variable, pybamm.SpatialVariable):
            # Check that child and integration_variable domains agree
            if child.domain != integration_variable.domain:
                raise pybamm.DomainError(
                    """child and integration_variable must have the same domain"""
                )
        elif not isinstance(integration_variable, pybamm.IndependentVariable):
            raise ValueError(
                """integration_variable must be of type pybamm.IndependentVariable,
                   not {}""".format(
                    type(integration_variable)
                )
            )
        name = "integral d{}".format(integration_variable.name)
        if isinstance(integration_variable, pybamm.SpatialVariable):
            name += " {}".format(integration_variable.domain)
        super().__init__(name, child)
        self._integration_variable = integration_variable
        # integrating removes the domain
        self.domain = []

    @property
    def integration_variable(self):
        return self._integration_variable


class BoundaryValue(SpatialOperator):
    """A node in the expression tree which gets the boundary value of a variable.

    Parameters
    ----------
    child : `pybamm.Symbol`
        The variable whose boundary value to take
    side : string
        Which side to take the boundary value on ("left" or "right")

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child, side):
        super().__init__("boundary", child)
        self.side = side
        # Domain of BoundaryValue must be ([]) so that expressions can be formed
        # of boundary values of variables in different domains
        self.domain = []


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


#
# Method to call SurfaceValue
#


def surf(variable):
    """convenience function for creating a :class:`SurfaceValue`

    Parameters
    ----------

    variable : :class:`Symbol`
        the surface value of this variable will be returned

    Returns
    -------

    :class:`GetSurfaceValue`
        the surface value of ``variable``
    """

    return BoundaryValue(variable, "right")
