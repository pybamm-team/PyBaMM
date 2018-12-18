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

    @property
    def id(self):
        """
        The immutable "identity" of a variable (for identifying y_slices).

        This is identical to what we'd put in a __hash__ function
        However, implementing __hash__ requires also implementing __eq__,
        which would then mess with loop-checking in the anytree module
        """

        return hash((self.__class__, self.name, self.children[0].id))


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

    return Gradient(expression)


def div(expression):
    """convenience function for creating a :class:`Divergence`

    Arguments:

    ``expression`` (:class:`Symbol`)
        the gradient will be performed on this sub-expression
    """

    return Divergence(expression)
