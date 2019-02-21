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

    @property
    def shape(self):
        """inherit shape from child"""
        return self.child.shape

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

    @property
    def shape(self):
        """
        lose shape inheritance from child, as Spatial Operator will change the shape
        """
        raise AttributeError(
            """'{}' object has no attribute 'shape'""".format(type(self))
        )


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
