#
# Matrix class
#
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, issparse

import pybamm
from pybamm.type_definitions import DomainType, AuxiliaryDomainType, DomainsType


class Matrix(pybamm.Array):
    """
    Node in the expression tree that holds a matrix type (e.g. :class:`numpy.array`)
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
        if isinstance(entries, list):
            entries = np.array(entries)
        if name is None:
            name = f"Matrix {entries.shape!s}"
            if issparse(entries):
                name = "Sparse " + name
        # Convert all sparse matrices to csr
        if issparse(entries) and not isinstance(entries, csr_matrix):
            entries = csr_matrix(entries)
        super().__init__(
            entries, name, domain, auxiliary_domains, domains, entries_string
        )
