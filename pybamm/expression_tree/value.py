#
# Value class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Value(pybamm.Symbol):
    def __init__(self, value):
        super().__init__(None)
        self._value = value

    def evaluate(self, y=None):
        return self._value
