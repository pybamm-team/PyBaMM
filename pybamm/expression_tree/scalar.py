#
# Scalar class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Scalar(pybamm.Symbol):
    """A node in the expression tree representing a scalar value"""

    def __init__(self, value, name=None, parent=None):
        """
        Args:
            value (numeric type): the value returned by the node when evaluated
            name (str): the name of the node. Optional, defaulted to
                        `str(value)` if not provided
        """
        # set default name if not provided
        if name is None:
            name = str(value)

        super().__init__(name, parent=parent)
        self.value = value

    @property
    def value(self):
        """numeric: the value returned by the node when evaluated"""
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def evaluate(self, y):
        """return the value of the Scalar node"""
        return self._value
