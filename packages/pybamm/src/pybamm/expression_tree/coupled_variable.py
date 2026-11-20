from __future__ import annotations

import pybamm
from pybamm.type_definitions import AuxiliaryDomainType, DomainsType, DomainType


class CoupledVariable(pybamm.Symbol):
    """
    A node in the expression tree representing a variable whose equation is set by
    a different model or submodel.

    A CoupledVariable is a placeholder that is replaced by the entry of the same
    name in ``model.variables``. Submodels that define :meth:`build` use them to
    refer to variables owned by other submodels; these are resolved when the model
    is built. Otherwise they are resolved lazily during discretisation.

    Parameters
    ----------
    name : str
        Name of the variable to couple to. Must match a key in model.variables.
    domain : iterable of str, optional
        List of domains that this coupled variable is valid over.
    auxiliary_domains : dict, optional
        Dictionary of auxiliary domains ({'secondary': ..., 'tertiary': ...,
        'quaternary': ...}).
    domains : dict, optional
        Dictionary of all domains, as an alternative to ``domain`` and
        ``auxiliary_domains``.
    on_edges : bool, optional
        Whether the variable it names is evaluated on cell edges in its primary
        dimension, as a flux is. Default is False.
    """

    __slots__ = ("_on_edges",)

    def __init__(
        self,
        name: str,
        domain: DomainType = None,
        auxiliary_domains: AuxiliaryDomainType = None,
        domains: DomainsType = None,
        on_edges: bool = False,
    ) -> None:
        self._on_edges = on_edges
        super().__init__(
            name, domain=domain, auxiliary_domains=auxiliary_domains, domains=domains
        )

    def _evaluates_on_edges(self, dimension):
        return self._on_edges and dimension == "primary"

    def _evaluate_for_shape(self):
        """
        Returns the scalar 'NaN' to represent the shape of a parameter.
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        return pybamm.evaluate_for_shape_using_domain(self._domains)

    def create_copy(
        self,
        new_children: list[pybamm.Symbol] | None = None,
        perform_simplifications: bool = True,
    ):
        """Creates a new copy of the coupled variable."""
        return CoupledVariable(
            self.name, domains=self._domains, on_edges=self._on_edges
        )
