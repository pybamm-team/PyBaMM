#
# NumpyArray class
#
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, issparse

import pybamm
from pybamm.type_definitions import DomainType, AuxiliaryDomainType, DomainsType
import sympy


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
        entries: np.ndarray | list[float] | csr_matrix,
        name: str | None = None,
        domain: DomainType = None,
        auxiliary_domains: AuxiliaryDomainType = None,
        domains: DomainsType = None,
        entries_string: str | None = None,
    ) -> None:
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

        return cls(
            matrix,
            name=snippet["name"],
            domains=snippet["domains"],
        )

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
    def entries_string(self, value: None | tuple):
        # We must include the entries in the hash, since different arrays can be
        # indistinguishable by class, name and domain alone
        # Slightly different syntax for sparse and non-sparse matrices
        if value is not None:
            self._entries_string = value
        else:
            entries = self._entries
            if issparse(entries):
                dct = entries.__dict__
                entries_string = ["shape", str(dct["_shape"])]
                for key in ["data", "indices", "indptr"]:
                    entries_string += [key, dct[key].tobytes()]
                self._entries_string = tuple(entries_string)
                # self._entries_string = str(entries.__dict__)
            else:
                self._entries_string = (entries.tobytes(),)

    def set_id(self):
        """See :meth:`pybamm.Symbol.set_id()`."""
        self._id = hash(
            (self.__class__, self.name, *self.entries_string, *tuple(self.domain))
        )

    def _jac(self, variable) -> pybamm.Matrix:
        """See :meth:`pybamm.Symbol._jac()`."""
        # Return zeros of correct size
        jac = csr_matrix((self.size, variable.evaluation_array.count(True)))
        return pybamm.Matrix(jac)

    def create_copy(
        self,
        new_children=None,
        perform_simplifications: bool = True,
    ):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        return self.__class__(
            self.entries,
            self.name,
            domains=self.domains,
            entries_string=self.entries_string,
        )

    def _base_evaluate(
        self,
        t: float | None = None,
        y: np.ndarray | None = None,
        y_dot: np.ndarray | None = None,
        inputs: dict | str | None = None,
    ):
        """See :meth:`pybamm.Symbol._base_evaluate()`."""
        return self._entries

    def is_constant(self):
        """See :meth:`pybamm.Symbol.is_constant()`."""
        return True

    def to_equation(self) -> sympy.Array:
        """Returns the value returned by the node when evaluated."""
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


def linspace(start: float, stop: float, num: int = 50, **kwargs) -> pybamm.Array:
    """
    Creates a linearly spaced array by calling `numpy.linspace` with keyword
    arguments 'kwargs'. For a list of 'kwargs' see the
    `numpy linspace documentation
    <https://numpy.org/doc/stable/reference/generated/numpy.linspace.html>`_
    """
    return pybamm.Array(np.linspace(start, stop, num, **kwargs))


def meshgrid(
    x: pybamm.Array, y: pybamm.Array, **kwargs
) -> tuple[pybamm.Array, pybamm.Array]:
    """
    Return coordinate matrices as from coordinate vectors by calling
    `numpy.meshgrid` with keyword arguments 'kwargs'. For a list of 'kwargs'
    see the `numpy meshgrid documentation
    <https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html>`_
    """
    [x_grid, y_grid] = np.meshgrid(x.entries, y.entries)
    return pybamm.Array(x_grid), pybamm.Array(y_grid)
