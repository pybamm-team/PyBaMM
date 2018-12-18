#
# Scalar class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Scalar(pybamm.Symbol):
    """A node in the expression tree representing a scalar value

    Arguments:

    ``value`` (numeric type)
        the value returned by the node when evaluated
    ``name`` (str)
        the name of the node. Optional, defaulted to ``str(value)``
        if not provided

    *Extends:* :class:`Symbol`
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
    def id(self):
        """
        The immutable "identity" of a variable (for identifying y_slices).

        This is identical to what we'd put in a __hash__ function
        However, implementing __hash__ requires also implementing __eq__,
        which would then mess with loop-checking in the anytree module
        """

        return hash((self.__class__, self.name, self.value))

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
