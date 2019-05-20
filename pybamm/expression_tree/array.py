#
# NumpyArray class
#
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

    def __init__(self, entries, name=None, domain=[], entries_string=None):
        if name is None:
            name = "Array of shape {!s}".format(entries.shape)
        self._entries = entries
        # Use known entries string to avoid re-hashing, where possible
        self.entries_string = entries_string
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

    @property
    def entries_string(self):
        return self._entries_string

    @entries_string.setter
    def entries_string(self, value):
        # We must include the entries in the hash, since different arrays can be
        # indistinguishable by class, name and domain alone
        # Slightly different syntax for sparse and non-sparse matrices
        if value is not None:
            self._entries_string = value
        else:
            entries = self._entries
            if issparse(entries):
                self._entries_string = str(entries.__dict__)
            else:
                self._entries_string = entries.tostring()

    def set_id(self):
        """ See :meth:`pybamm.Symbol.set_id()`. """
        self._id = hash(
            (self.__class__, self.name, self.entries_string) + tuple(self.domain)
        )

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return self.__class__(self.entries, self.name, self.domain, self.entries_string)

    def _base_evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol._base_evaluate()`. """
        return self._entries
