#
# NumpyArray class
#
import numpy as np
from scipy.sparse import csr_matrix, issparse

import pybamm
from pybamm.util import have_optional_dependency


class Array(pybamm.Symbol):
    """
    Node in the expression tree that holds an tensor type variable
    (e.g. :class:`numpy.array`)

    Parameters
    ----------

    entries : numpy.array or list
        the array associated with the node. If a list is provided, it is converted to a
        numpy array
    name : str, optional
        the name of the node
    domain : iterable of str, optional
        list of domains the parameter is valid over, defaults to empty list
    auxiliary_domains : dict, optional
        dictionary of auxiliary domains, defaults to empty dict
    domains : dict
        A dictionary equivalent to {'primary': domain, auxiliary_domains}. Either
        'domain' and 'auxiliary_domains', or just 'domains', should be provided
        (not both). In future, the 'domain' and 'auxiliary_domains' arguments may be
        deprecated.
    entries_string : str
        String representing the entries (slow to recalculate when copying)
    """

    def __init__(
        self,
        entries,
        name=None,
        domain=None,
        auxiliary_domains=None,
        domains=None,
        entries_string=None,
    ):
        # if
        if isinstance(entries, list):
            entries = np.array(entries)
        if entries.ndim == 1:
            entries = entries[:, np.newaxis]
        if name is None:
            name = f"Array of shape {entries.shape!s}"
        self._entries = entries.astype(float)
        # Use known entries string to avoid re-hashing, where possible
        self.entries_string = entries_string
        super().__init__(
            name, domain=domain, auxiliary_domains=auxiliary_domains, domains=domains
        )

    @classmethod
    def _from_json(cls, snippet: dict):
        instance = cls.__new__(cls)

        if isinstance(snippet["entries"], dict):
            matrix = csr_matrix(
                (
                    snippet["entries"]["data"],
                    snippet["entries"]["row_indices"],
                    snippet["entries"]["column_pointers"],
                ),
                shape=snippet["entries"]["shape"],
            )
        else:
            matrix = snippet["entries"]

        instance.__init__(
            matrix,
            name=snippet["name"],
            domains=snippet["domains"],
        )

        return instance

    @property
    def entries(self):
        return self._entries

    @property
    def ndim(self):
        """returns the number of dimensions of the tensor."""
        return self._entries.ndim

    @property
    def shape(self):
        """returns the number of entries along each dimension."""
        return self._entries.shape

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
                dct = entries.__dict__
                self._entries_string = ["shape", str(dct["_shape"])]
                for key in ["data", "indices", "indptr"]:
                    self._entries_string += [key, dct[key].tobytes()]
                self._entries_string = tuple(self._entries_string)
                # self._entries_string = str(entries.__dict__)
            else:
                self._entries_string = (entries.tobytes(),)

    def set_id(self):
        """See :meth:`pybamm.Symbol.set_id()`."""
        self._id = hash(
            (self.__class__, self.name, *self.entries_string, *tuple(self.domain))
        )

    def _jac(self, variable):
        """See :meth:`pybamm.Symbol._jac()`."""
        # Return zeros of correct size
        jac = csr_matrix((self.size, variable.evaluation_array.count(True)))
        return pybamm.Matrix(jac)

    def create_copy(self):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        return self.__class__(
            self.entries,
            self.name,
            domains=self.domains,
            entries_string=self.entries_string,
        )

    def _base_evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        """See :meth:`pybamm.Symbol._base_evaluate()`."""
        return self._entries

    def is_constant(self):
        """See :meth:`pybamm.Symbol.is_constant()`."""
        return True

    def to_equation(self):
        """Returns the value returned by the node when evaluated."""
        sympy = have_optional_dependency("sympy")
        entries_list = self.entries.tolist()
        return sympy.Array(entries_list)

    def to_json(self):
        """
        Method to serialise an Array object into JSON.
        """

        if isinstance(self.entries, np.ndarray):
            matrix = self.entries.tolist()
        elif isinstance(self.entries, csr_matrix):
            matrix = {
                "shape": self.entries.shape,
                "data": self.entries.data.tolist(),
                "row_indices": self.entries.indices.tolist(),
                "column_pointers": self.entries.indptr.tolist(),
            }

        json_dict = {
            "name": self.name,
            "id": self.id,
            "domains": self.domains,
            "entries": matrix,
        }

        return json_dict


def linspace(start, stop, num=50, **kwargs):
    """
    Creates a linearly spaced array by calling `numpy.linspace` with keyword
    arguments 'kwargs'. For a list of 'kwargs' see the
    `numpy linspace documentation <https://tinyurl.com/yc4ne47x>`_
    """
    return pybamm.Array(np.linspace(start, stop, num, **kwargs))


def meshgrid(x, y, **kwargs):
    """
    Return coordinate matrices as from coordinate vectors by calling
    `numpy.meshgrid` with keyword arguments 'kwargs'. For a list of 'kwargs'
    see the `numpy meshgrid documentation <https://tinyurl.com/y8azewrj>`_
    """
    [X, Y] = np.meshgrid(x.entries, y.entries)
    X = pybamm.Array(X)
    Y = pybamm.Array(Y)
    return X, Y
