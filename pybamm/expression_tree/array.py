#
# NumpyArray class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Array(pybamm.Symbol):
    """
    Parameters
    ----------
    entries : :class:`numpy.array`
        Entries of the matrix
    """

    def __init__(self, entries, name=None, parent=None):
        if name is None:
            name = str(entries)
        super().__init__(name, parent=parent)
        self._entries = entries

    @property
    def ndim(self):
        return self._entries.ndim

    @property
    def shape(self):
        return self._entries.shape

    @property
    def size(self):
        return self._entries.size

    def evaluate(self, y):
        return self._entries
