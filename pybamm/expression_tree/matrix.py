#
# Matrix class
#
import pybamm
import numpy as np
from scipy.sparse import issparse


class Matrix(pybamm.Array):
    """node in the expression tree that holds a matrix type (e.g. :class:`numpy.array`)

    **Extends:** :class:`Array`

    """

    def __init__(
        self,
        entries,
        name=None,
        domain=None,
        auxiliary_domains=None,
        entries_string=None,
    ):
        if isinstance(entries, list):
            entries = np.array(entries)
        if name is None:
            name = "Matrix {!s}".format(entries.shape)
            if issparse(entries):
                name = "Sparse " + name
        super().__init__(entries, name, domain, auxiliary_domains, entries_string)
