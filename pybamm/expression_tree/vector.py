#
# Vector class
#
import pybamm

import numpy as np


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
