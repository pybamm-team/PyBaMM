#
# Vector classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Vector(pybamm.Array):
    def __init__(self, entries, name=None, parent=None):
        super().__init__(entries, name=name, parent=parent)


class VariableVector(pybamm.Symbol):
    """A vector that depends on an input y."""

    def __init__(self, y_slice, name=None, parent=None):
        if name is None:
            name = str(y_slice)
        super().__init__(name=name, parent=parent)
        self._y_slice = y_slice

    @property
    def y_slice(self):
        return self._y_slice

    def evaluate(self, y):
        return y[self._y_slice]
