#
# Vector classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Vector(pybamm.Array):
    """node in the expression tree that holds a vector type (e.g. :class:`numpy.array`)

    **Extends:** :class:`Array`

    Parameters
    ----------

    entries : numpy.array
        the array associated with the node
    name : str, optional
        the name of the node

    """

    def __init__(self, entries, name=None):
        super().__init__(entries, name=name)


class StateVector(pybamm.Symbol):
    """
    node in the expression tree that holds a slice to read from an external vector type

    Arguments:

    ``y_slice``
        the slice of an external y to read
    ``name``
        the name of the node

    *Extends:* :class:`Array`
    """

    def __init__(self, y_slice, name=None):
        if name is None:
            name = str(y_slice)
        super().__init__(name=name)
        self._y_slice = y_slice

    @property
    def y_slice(self):
        """Slice of an external y to read"""
        return self._y_slice

    def evaluate(self, t, y):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return y[self._y_slice]
