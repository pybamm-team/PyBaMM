#
# Unary operator classes and methods
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import numpy as np


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

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.func(self.children[0].evaluate(t, y))


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
    """A node in the expression tree representing an integral operator (definite or
    indefinite)

    .. math::
        \\text{definite}: \\quad I = \\int_{a}^{b}\\!f(u)\\,du,

        \\text{indefinite}: \\quad I(s) = \\int_{a}^{s}\\!f(u)\\,du,

    where :math:`a` and :math:`b` are the left-hand and right-hand boundaries of
    the domain respectively, and :math:`s\\in\\text{domain}`.
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

    @property
    def integration_variable(self):
        return self._integration_variable


class SurfaceValue(SpatialOperator):
    """A node in the expression tree which gets the surface value of a variable.

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child):
        super().__init__("surf", child)

        # Domain of SurfaceValue must be ([]) so that expressions can be formed
        # of surface values of variables in different domains
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

    return SurfaceValue(variable)
