#
# NumpyArray class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

from scipy.sparse import issparse


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
            name = "Array of shape {!s}".format(entries.shape)
        self._entries = entries
        super().__init__(name, domain=domain)

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

    def set_id(self):
        """ See :meth:`pybamm.Symbol.set_id()`. """
        # We must include the entries in the hash, since different arrays can be
        # indistinguishable by class, name and domain alone
        # Slightly different syntax for sparse and non-sparse matrices
        entries = self._entries
        if issparse(entries):
            entries_str = entries.data.tostring()
        else:
            entries_str = entries.tostring()

        self._id = hash(
            (self.__class__, self.name) + tuple(self.domain) + tuple(entries_str)
        )

    def _base_evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol._base_evaluate()`. """
        return self._entries
