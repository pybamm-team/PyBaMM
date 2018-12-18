#
# Unary operator classes and methods
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class UnaryOperator(pybamm.Symbol):
    """A node in the expression tree representing a unary operator
    (e.g. '-', grad, div)

    Derived classes will specify the particular operator

    Arguments:

    ``name`` (str)
        name of the node
    ``child`` (:class:`Symbol`)
        child node

    *Extends:* :class:`Symbol`
    """

    def __init__(self, name, child):
        super().__init__(name, children=[child])


class SpatialOperator(UnaryOperator):
    """A node in the expression tree representing a unary spatial operator
    (e.g. grad, div)

    Derived classes will specify the particular operator

    This type of node will be replaced by the :class:`BaseDiscretisation`
    class with a :class:`Matrix`

    Arguments:

    ``name`` (str)
        name of the node
    ``child`` (:class:`Symbol`)
        child node

    *Extends:* :class:`UnaryOperator`
    """

    def __init__(self, name, child):
        super().__init__(name, child)
        # self.domain = child.domain

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        return "{}({!s})".format(self.name, self.children[0])


class Gradient(SpatialOperator):
    """A node in the expression tree representing an grad operator

    *Extends:* :class:`SpatialOperator`
    """

    def __init__(self, child):
        super().__init__("grad", child)


class Divergence(SpatialOperator):
    """A node in the expression tree representing an div operator

    *Extends:* :class:`SpatialOperator`
    """

    def __init__(self, child):
        super().__init__("div", child)


def grad(expression):
    """convenience function for creating a :class:`Gradient`

    Arguments:

    ``expression`` (:class:`Symbol`)
        the gradient will be performed on this sub-expression
    """

    return Gradient(variable)


def div(expression):
    """convenience function for creating a :class:`Divergence`

    Arguments:

    ``expression`` (:class:`Symbol`)
        the gradient will be performed on this sub-expression
    """

    return Divergence(variable)
