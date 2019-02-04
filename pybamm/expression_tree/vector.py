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
    domain : iterable of str, optional
        list of domains the parameter is valid over, defaults to empty list

    """

    def __init__(self, entries, name=None, domain=[]):
        if name is None:
            name = "Vector of shape {!s}".format(entries.shape)
        super().__init__(entries, name=name, domain=domain)


class StateVector(pybamm.Symbol):
    """
    node in the expression tree that holds a slice to read from an external vector type

    Parameters
    ----------

    y_slice: slice
        the slice of an external y to read
    name: str, optional
        the name of the node
    domain : iterable of str, optional
        list of domains the parameter is valid over, defaults to empty list

    *Extends:* :class:`Array`
    """

    def __init__(self, y_slice, name=None, domain=[]):
        if name is None:
            name = "StateVector with slice '{!s}'".format(y_slice)
        super().__init__(name=name, domain=domain)
        self._y_slice = y_slice

    @property
    def y_slice(self):
        """Slice of an external y to read"""
        return self._y_slice

    def evaluate(self, t, y):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return y[self._y_slice]
