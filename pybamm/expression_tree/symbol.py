#
# Base Symbol Class for the expression tree
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import anytree
import copy


class Symbol(anytree.NodeMixin):
    def __init__(self, name, children=[], parent=None):
        super(Symbol, self).__init__()
        self._name = name

        for child in children:
            # copy child before adding
            copy.copy(child).parent = self

    @property
    def name(self):
        return self._name

    def render(self):
        for pre, _, node in anytree.RenderTree(self):
            print("%s%s" % (pre, node.name))

    def pre_order(self):
        return anytree.PreOrderIter(self)

    def __str__(self):
        return self._name

    def __repr__(self):
        return "{!s}({!s}, {!s}, {!s})".format(
            self.__class__, self.name, self.children, self.parent
        )

    @property
    def id(self):
        return hash((self.__class__, self.name))

    def __add__(self, other):
        if isinstance(other, Symbol):
            return pybamm.Addition(self, other)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, Symbol):
            return pybamm.Subtraction(self, other)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, Symbol):
            return pybamm.Multiplication(self, other)
        else:
            raise NotImplementedError

    def __truediv__(self, other):
        if isinstance(other, Symbol):
            return pybamm.Division(self, other)
        else:
            raise NotImplementedError

    def evaluate(self, y):
        raise NotImplementedError(
            """method self.evaluate(y) not implemented
               for symbol {!s} of type {}""".format(
                self, type(self)
            )
        )
