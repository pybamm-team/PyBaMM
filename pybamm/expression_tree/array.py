#
# NumpyArray class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Array(pybamm.Symbol):
    """node in the expression tree that holds an tensor type variable
    (e.g. :class:`numpy.array`)

    Parameters
    ----------

    entries : numpy.array
        the array associated with the node
    name : str, optional
        the name of the node
    domain : iterable of str, optional
        list of domains the parameter is valid over, defaults to empty list

    *Extends:* :class:`Symbol`
    """

    def __init__(self, entries, name=None, domain=[]):
        if name is None:
            name = str(entries)
        super().__init__(name, domain=domain)
        self._entries = entries

    @property
    def entries(self):
        return self._entries

    @property
    def ndim(self):
        """ returns the number of dimensions of the tensor"""
        return self._entries.ndim

    @property
    def shape(self):
        """ returns the number of entries along each dimension"""
        return self._entries.shape

    @property
    def size(self):
        """ returns the total number of entries in the tensor"""
        return self._entries.size

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self._entries
