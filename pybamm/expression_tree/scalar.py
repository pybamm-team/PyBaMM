#
# Scalar class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Scalar(pybamm.Domain, pybamm.Symbol):
    """A node in the expression tree representing a scalar value

    **Extends:** :class:`Symbol`

    Parameters
    ----------

    value : numeric
        the value returned by the node when evaluated
    name : str, optional
        the name of the node. Defaulted to ``str(value)``
        if not provided
    domain : iterable of str, optional
        list of domains the parameter is valid over, defaults to empty list

    """

    def __init__(self, value, name=None, domain=[]):
        """

        """
        # set default name if not provided
        if name is None:
            name = str(value)

        super().__init__(name, domain=domain)
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
