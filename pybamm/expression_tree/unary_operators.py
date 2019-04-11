#
# Unary operator classes and methods
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import autograd
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

    def simplify(self):
        """ See :meth:`pybamm.Symbol.simplify()`. """
        child = self.children[0].simplify()

        # _binary_simplify defined in derived classes for specific rules
        new_node = self._unary_simplify(child)

        return pybamm.simplify_if_constant(new_node)

    def _unary_simplify(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """

        return self.__class__(child)

    def _unary_evaluate(self, child):
        """Perform unary operation on a child. """
        raise NotImplementedError

    def evaluate(self, t=None, y=None, known_evals=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        if known_evals is not None:
            if self.id not in known_evals:
                child, known_evals = self.children[0].evaluate(t, y, known_evals)
                known_evals[self.id] = self._unary_evaluate(child)
            return known_evals[self.id], known_evals
        else:
            child = self.children[0].evaluate(t, y)
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
        return "{}{!s}".format(self.name, self.children[0])

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            return -self.children[0].diff(variable)

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

    def _unary_evaluate(self, child):
        """ See :meth:`UnaryOperator._unary_evaluate()`. """
        return np.abs(child)


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

    def _unary_evaluate(self, child):
        """ See :meth:`UnaryOperator._unary_evaluate()`. """
        return self.func(child)

    # Function needs its own simplify as it has a different __init__ signature
    def simplify(self):
        """ See :meth:`pybamm.Symbol.simplify()`. """
        child = self.children[0].simplify()

        new_node = pybamm.Function(self.func, child)

        return pybamm.simplify_if_constant(new_node)


class Index(UnaryOperator):
    """A node in the expression tree, which stores the index that should be
    extracted from its child after the child has been evaluated.
    """

    def __init__(self, child, index, name=None):
        if name is None:
            name = child.name + "[" + str(index) + "]"
        super().__init__(name, child)
        self.index = index

    def _unary_evaluate(self, child):
        """ See :meth:`UnaryOperator._unary_evaluate()`. """
        return child[self.index]

    def _unary_simplify(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """

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

    def _unary_simplify(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """

        return self.__class__(child, self.integration_variable)


class IndefiniteIntegral(SpatialOperator):
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
        name = "{} integrated w.r.t {}".format(child.name, integration_variable.name)
        if isinstance(integration_variable, pybamm.SpatialVariable):
            name += "on {}".format(integration_variable.domain)
        super().__init__(name, child)
        self._integration_variable = integration_variable
        # the integrated variable has the same domain as the child
        self.domain = child.domain

    @property
    def integration_variable(self):
        return self._integration_variable

    def _unary_simplify(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """

        return self.__class__(child, self.integration_variable)


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

    def _unary_simplify(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """

        return self.__class__(child, self.side)


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


def integrate(expression, variable):
    """convenience function for creating a :class:`Integral`

    Parameters
    ----------

    expression : :class:`pybamm.Symbol`
        The function to be integrated
    integration_variable : :class:`pybamm.IndependentVariable`
        The variable over which to integrate

    Returns
    -------

    :class:`Integral`
        the new integrated expression tree
    """

    return Integral(expression, variable)
