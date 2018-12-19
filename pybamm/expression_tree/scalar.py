#
# Scalar class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Scalar(pybamm.Symbol):
    """A node in the expression tree representing a scalar value

    **Extends:** :class:`Symbol`

    Parameters
    ----------

    value : numeric
        the value returned by the node when evaluated
    name : str, optional
        the name of the node. Defaulted to ``str(value)``
        if not provided

    """

    def __init__(self, value, name=None):
        """

        """
        # set default name if not provided
        if name is None:
            name = str(value)

        super().__init__(name)
        self.value = value

    @property
    def value(self):
        """the value returned by the node when evaluated"""
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self._value
