#
# Base Symbol Class for the expression tree
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import anytree


class Symbol(anytree.AnyNode):
    def __init__(self, name, parent=None):
        super(Symbol, self).__init__(id=name, parent=parent)
        self._name = name

    def __str__(self):
        return self._name

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
