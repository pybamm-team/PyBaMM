#
# NumpyArray class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Array(pybamm.Symbol):
    """node in the expression tree that holds an tensor type variable (e.g. :class:`numpy.array`)

    Arguements:
    ``entries``
        the array associated with the node
    ``name``
        the name of the node

    *Extends:* :class:`Symbol`
    """

    def __init__(self, entries, name=None):
        if name is None:
            name = str(entries)
        super().__init__(name)
        self._entries = entries

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

    def evaluate(self, y):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return self._entries
