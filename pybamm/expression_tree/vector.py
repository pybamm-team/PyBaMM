#
# Vector class
#
import numpy as np

import pybamm


class Vector(pybamm.Array):
    """
    node in the expression tree that holds a vector type (e.g. :class:`numpy.array`)
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
        if isinstance(entries, (list, np.matrix)):
            entries = np.array(entries)
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

        super().__init__(
            entries, name, domain, auxiliary_domains, domains, entries_string
        )
