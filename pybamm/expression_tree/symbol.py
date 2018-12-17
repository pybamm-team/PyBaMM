#
# Base Symbol Class for the expression tree
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import anytree
import copy


class Symbol(anytree.NodeMixin):
    """Base node class for the expression tree

    Arguements:

    ``name`` (str)
        name for the node

    ``children`` (iterable of :class:`Symbol`)
        children to attach to this node

    """

    def __init__(self, name, children=[]):
        super(Symbol, self).__init__()
        self._name = name

        for child in children:
            # copy child before adding
            copy.copy(child).parent = self

    @property
    def name(self):
        """name of the node"""
        return self._name

    def render(self):
        """print out a visual representation of the tree (this node and its
        children)
        """
        for pre, _, node in anytree.RenderTree(self):
            print("%s%s" % (pre, str(node)))

    def pre_order(self):
        """returns an iterable that steps through the tree in pre-order
        fashion

        Example:

        .. code-block:: python

            for node in tree.pre_order():
                print node.name

        """
        return anytree.PreOrderIter(self)

    def __str__(self):
        """returns the name of the node"""
        return self._name

    def __repr__(self):
        """returns the string `Symbol(name, parent)`"""
        return "Symbol({!s}, {!s})".format(self._name, self.parent)

    def __add__(self, other):
        """return an :class:`Addition` object"""
        if isinstance(other, Symbol):
            return pybamm.Addition(self, other)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        """return an :class:`Subtraction` object"""
        if isinstance(other, Symbol):
            return pybamm.Subtraction(self, other)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        """return an :class:`Multiplication` object"""
        if isinstance(other, Symbol):
            return pybamm.Multiplication(self, other)
        else:
            raise NotImplementedError

    def __truediv__(self, other):
        """return an :class:`Division` object"""
        if isinstance(other, Symbol):
            return pybamm.Division(self, other)
        else:
            raise NotImplementedError

    def evaluate(self, y):
        """evaluate expression tree

        Arguments:

        ``y`` (numpy.array)
            an expression can depend on a vector

        """
        raise NotImplementedError(
            """method self.evaluate(y) not implemented
               for symbol {!s} of type {}""".format(
                self, type(self)
            )
        )
