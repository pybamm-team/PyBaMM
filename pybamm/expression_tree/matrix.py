#
# Matrix class
#
import numpy as np
from scipy.sparse import csr_matrix, issparse

import pybamm


class Matrix(pybamm.Array):
    """
    Node in the expression tree that holds a matrix type (e.g. :class:`numpy.array`)
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
        if isinstance(entries, list):
            entries = np.array(entries)
        if name is None:
            name = "Matrix {!s}".format(entries.shape)
            if issparse(entries):
                name = "Sparse " + name
        # Convert all sparse matrices to csr
        if issparse(entries) and not isinstance(entries, csr_matrix):
            entries = csr_matrix(entries)
        super().__init__(
            entries, name, domain, auxiliary_domains, domains, entries_string
        )
