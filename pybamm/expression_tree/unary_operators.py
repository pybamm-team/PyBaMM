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


class SpatialOperator(UnaryOperator):
    """A node in the expression tree representing a unary spatial operator
    (e.g. grad, div)

    Derived classes will specify the particular operator

    This type of node will be replaced by the :class:`BaseDiscretisation`
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
    """A node in the expression tree representing an intgral operator
    Can be integration with respect to time or space

    Parameters
    ----------
    function : :class:`pybamm.Symbol`
        The function to be integrated (will become self.children[0])
    integration_variable : :class:`pybamm.IndependentVariable`
        The variable over which to integrate

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, function, integration_variable):
        if isinstance(integration_variable, pybamm.Space):
            # Check that function and integration_variable domains agree
            if function.domain != integration_variable.domain:
                raise pybamm.DomainError(
                    """function and integration_variable must have the same domain"""
                )
        elif not isinstance(integration_variable, pybamm.IndependentVariable):
            raise ValueError(
                """integration_variable must be of type pybamm.IndependentVariable,
                   not {}""".format(
                    type(integration_variable)
                )
            )
        super().__init__("integral d{}".format(integration_variable), function)
        self._integration_variable = integration_variable

    @property
    def function(self):
        return self.children[0]

    @property
    def integration_variable(self):
        return self._integration_variable


class Broadcast(SpatialOperator):
    """A node in the expression tree representing a broadcasting operator.
    Broadcasts a child (which *must* have empty domain) to a specified domain. After
    discretisation, this will evaluate to an array of the right shape for the specified
    domain.

    Parameters
    ----------
    child : :class:`Symbol`
        child node
    domain : iterable of string
        the domain to broadcast the child to
    name : string
        name of the node

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child, domain, name=None):
        if child.domain != []:
            raise pybamm.DomainError(
                """Domain of a broadcasted child must be [] but is '{}'""".format(
                    child.domain
                )
            )
        if name is None:
            name = "broadcast"
        super().__init__(name, child)
        # overwrite child domain ([]) with specified broadcasting domain
        self.domain = domain


class NumpyBroadcast(Broadcast):
    """A node in the expression tree implementing a broadcasting operator using numpy.
    Broadcasts a child (which *must* have empty domain) to a specified domain. To do
    this, creates a np array of ones of the same shape as the submesh domain, and then
    multiplies the child by that array upon evaluation

    Parameters
    ----------
    child : :class:`Symbol`
        child node
    domain : iterable of string
        the domain to broadcast the child to
    mesh : mesh class
        the mesh used for discretisation

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child, domain, mesh):
        super().__init__(child, domain, name="numpy broadcast")
        # create broadcasting vector (vector of ones with shape determined by the
        # domain)
        broadcasting_vector_size = sum([mesh[dom].npts for dom in domain])
        self.broadcasting_vector = np.ones(broadcasting_vector_size)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        # if child is a vector, add a dimension for broadcasting
        if isinstance(self.children[0], pybamm.Vector):
            return (
                self.children[0].evaluate(t, y)[:, np.newaxis]
                * self.broadcasting_vector
            )
        # otherwise just do normal multiplication
        else:
            return self.children[0].evaluate(t, y) * self.broadcasting_vector


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
