#
# Unary operator classes and methods
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class UnaryOperator(pybamm.Symbol):
    def __init__(self, name, child):
        super().__init__(name, children=[child])


class SpatialOperator(UnaryOperator):
    def __init__(self, name, child):
        super().__init__(name, child)
        # self.domain = child.domain

    def __str__(self):
        return "{}({!s})".format(self.name, self.children[0])


class Gradient(SpatialOperator):
    def __init__(self, child):
        super().__init__("grad", child)


class Divergence(SpatialOperator):
    def __init__(self, child):
        super().__init__("div", child)


def grad(variable):
    return Gradient(variable)


def div(variable):
    return Divergence(variable)
