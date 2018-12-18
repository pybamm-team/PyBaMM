#
# Binary operator classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class BinaryOperator(pybamm.Symbol):
    """A node in the expression tree representing a binary operator (e.g. `+`, `*`)

    Derived classes will specify the particular operator

    Arguments:

    ``name`` (str)
        name of the node
    ``left`` (:class:`Symbol`)
        lhs child node
    ``right`` (:class:`Symbol`)
        rhs child node

    *Extends:* :class:`Symbol`
    """

    def __init__(self, name, left, right):
        super().__init__(name, children=[left, right])

    @property
    def id(self):
        """
        The immutable "identity" of a variable (for identifying y_slices).

        This is identical to what we'd put in a __hash__ function
        However, implementing __hash__ requires also implementing __eq__,
        which would then mess with loop-checking in the anytree module
        """
        return hash(
            (self.__class__, self.name, self.children[0].id, self.children[1].id)
        )

    def __str__(self):
        """ See :meth:`pybamm.Symbol.__str__()`. """
        return "{!s} {} {!s}".format(self.children[0], self.name, self.children[1])


class Addition(BinaryOperator):
    """A node in the expression tree representing an addition operator

    *Extends:* :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__("+", left, right)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(t, y) + self.children[1].evaluate(t, y)


class Subtraction(BinaryOperator):
    """A node in the expression tree representing a subtraction operator

    *Extends:* :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """

        super().__init__("-", left, right)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(t, y) - self.children[1].evaluate(t, y)


class Multiplication(BinaryOperator):
    """A node in the expression tree representing a multiplication operator

    *Extends:* :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """

        super().__init__("*", left, right)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        if isinstance(self.children[0], pybamm.Matrix):
            return self.children[0].evaluate(t, y) @ self.children[1].evaluate(t, y)
        else:
            return self.children[0].evaluate(t, y) * self.children[1].evaluate(t, y)


class Division(BinaryOperator):
    """A node in the expression tree representing a division operator

    *Extends:* :class:`BinaryOperator`
    """

    def __init__(self, left, right):
        """ See :meth:`pybamm.BinaryOperator.__init__()`. """
        super().__init__("/", left, right)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self.children[0].evaluate(t, y) / self.children[1].evaluate(t, y)
