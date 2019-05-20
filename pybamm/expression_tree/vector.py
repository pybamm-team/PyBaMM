#
# Vector classes
#
import pybamm

import numpy as np
from scipy.sparse import csr_matrix


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

    def __init__(self, entries, name=None, domain=[], entries_string=None):
        # make sure that entries are a vector (can be a column vector)
        if entries.ndim == 1:
            entries = entries[:, np.newaxis]
        if entries.shape[1] != 1:
            raise ValueError(
                """
                Entries must have 1 dimension or be column vector, not have shape {}
                """.format(
                    entries.shape
                )
            )
        if name is None:
            name = "Column vector of length {!s}".format(entries.shape[0])

        super().__init__(entries, name, domain, entries_string)

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        # Get indices of state vector
        variable_y_indices = np.arange(variable.y_slice.start, variable.y_slice.stop)
        # Return zeros of correct size
        jac = csr_matrix((np.size(self), np.size(variable_y_indices)))
        return pybamm.Matrix(jac)


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
            if y_slice.start is None:
                name = "y[:{:d}]".format(y_slice.stop)
            else:
                name = "y[{:d}:{:d}]".format(y_slice.start, y_slice.stop)
        super().__init__(name=name, domain=domain)
        self._y_slice = y_slice

    @property
    def y_slice(self):
        """Slice of an external y to read"""
        return self._y_slice

    def _base_evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol._base_evaluate()`. """
        if y is None:
            raise TypeError("StateVector cannot evaluate input 'y=None'")
        if y.shape[0] < self.y_slice.stop:
            raise ValueError(
                "y is too short, so value with slice is smaller than expected"
            )
        else:
            out = y[self._y_slice]
            if out.ndim == 1:
                out = out[:, np.newaxis]
            return out

    def jac(self, variable):
        """
        Differentiate a slice of a StateVector of size m with respect to another
        slice of a StateVector of size n. This returns a (sparse) matrix of size
        m x n with ones where the y slices match, and zeros elsewhere.

        Parameters
        ----------
        variable : :class:`pybamm.Symbol`
            The variable with respect to which to differentiate

        """

        # Get inices of state vectors
        self_y_indices = np.arange(self.y_slice.start, self.y_slice.stop)
        variable_y_indices = np.arange(variable.y_slice.start, variable.y_slice.stop)

        # Return zeros of correct size if no entries match
        if np.size(np.intersect1d(self_y_indices, variable_y_indices)) == 0:
            jac = csr_matrix((np.size(self_y_indices), np.size(variable_y_indices)))
        else:
            # Populate entries corresponding to matching y slices, and shift so
            # that the matrix is the correct size
            row = (
                np.intersect1d(self_y_indices, variable_y_indices) - self.y_slice.start
            )
            col = (
                np.intersect1d(self_y_indices, variable_y_indices)
                - variable.y_slice.start
            )
            data = np.ones_like(row)
            jac = csr_matrix(
                (data, (row, col)),
                shape=(np.size(self_y_indices), np.size(variable_y_indices)),
            )
        return pybamm.Matrix(jac)

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return StateVector(self.y_slice, self.name, domain=[])
