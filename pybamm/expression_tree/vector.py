#
# Vector class
#
from __future__ import annotations
import numpy as np

import pybamm
from pybamm.type_definitions import DomainType, AuxiliaryDomainType, DomainsType


class Vector(pybamm.Array):
    """
    node in the expression tree that holds a vector type (e.g. :class:`numpy.array`)
    """

    def __init__(
        self,
        entries: np.ndarray | list[float] | np.matrix,
        name: str | None = None,
        domain: DomainType = None,
        auxiliary_domains: AuxiliaryDomainType = None,
        domains: DomainsType = None,
        entries_string: str | None = None,
    ) -> None:
        if isinstance(entries, (list, np.matrix)):
            entries = np.array(entries)
        # make sure that entries are a vector (can be a column vector)
        if entries.ndim == 1:
            entries = entries[:, np.newaxis]
        if entries.shape[1] != 1:
            raise ValueError(
                f"""
                Entries must have 1 dimension or be column vector, not have shape {entries.shape}
                """
            )
        if name is None:
            name = f"Column vector of length {entries.shape[0]!s}"

        super().__init__(
            entries, name, domain, auxiliary_domains, domains, entries_string
        )
