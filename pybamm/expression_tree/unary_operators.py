#
# Unary operator classes and methods
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class UnaryOperator(pybamm.Symbol):
    def __init__(self, name, child, parent=None):
        super().__init__(name, parent)
        self.child = child


class SpatialOperator(UnaryOperator):
    def __init__(self, name, child, parent=None):
        super().__init__(name, child, parent)
        self.domain = child.domain

    def __str__(self):
        return "{}({!s})".format(self.name, self.child)


class Gradient(SpatialOperator):
    def __init__(self, child, parent=None):
        super().__init__("grad", child, parent)


class Divergence(SpatialOperator):
    def __init__(self, child, parent=None):
        super().__init__("div", child, parent)


def grad(variable):
    return Gradient(variable)


def div(variable):
    return Divergence(variable)
