#
# Scalar class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Scalar(pybamm.Symbol):
    def __init__(self, value, name=None, parent=None):
        # set default name if not provided
        if name is None:
            name = str(value)

        super().__init__(name, parent=parent)
        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def evaluate(self, y):
        return self._value
