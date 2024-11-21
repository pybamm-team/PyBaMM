#
# IndependentVariable class
#
from __future__ import annotations
from typing import Optional
import sympy
import numpy as np

import pybamm
from pybamm.type_definitions import DomainType, AuxiliaryDomainType, DomainsType


class IndependentVariable(pybamm.Symbol):
    """
    A node in the expression tree representing an independent variable.

    Used for expressing functions depending on a spatial variable or time

    Parameters
    ----------
    domain : iterable of str
        list of domains that this variable is valid over
    auxiliary_domains : dict, optional
        dictionary of auxiliary domains, defaults to empty dict
    domains : dict
        A dictionary equivalent to {'primary': domain, auxiliary_domains}. Either
        'domain' and 'auxiliary_domains', or just 'domains', should be provided
        (not both). In future, the 'domain' and 'auxiliary_domains' arguments may be
        deprecated.
    """

    def __init__(
        self,
        domain: DomainType = None,
        auxiliary_domains: AuxiliaryDomainType = None,
        domains: DomainsType = None,
    ) -> None:
        super().__init__(
            name="independent variable",
            domain=domain,
            auxiliary_domains=auxiliary_domains,
            domains=domains,
        )

    @classmethod
    def _from_json(cls, snippet: dict):
        return cls(snippet["name"], domains=snippet["domains"])

    def _evaluate_for_shape(self):
        """See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`"""
        return pybamm.evaluate_for_shape_using_domain(self.domains)

    def _jac(self, variable) -> pybamm.Scalar:
        """See :meth:`pybamm.Symbol._jac()`."""
        return pybamm.Scalar(0)

    def to_equation(self) -> sympy.Symbol:
        """Convert the node and its subtree into a SymPy equation."""
        if self.print_name is not None:
            return sympy.Symbol(self.print_name)
        else:
            return sympy.Symbol(self.name)

    def create_copy(
        self,
        new_children=None,
        perform_simplifications=True,
    ):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        return self.__class__(self.name, domains=self.domains)


class Time(IndependentVariable):
    """
    A node in the expression tree representing time.
    """

    @classmethod
    def _from_json(cls, snippet: dict):
        return cls()

    def create_copy(
        self,
        new_children=None,
        perform_simplifications=True,
    ):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        return Time()

    def _base_evaluate(
        self,
        t: float | None = None,
        y: np.ndarray | None = None,
        y_dot: np.ndarray | None = None,
        inputs: dict | str | None = None,
    ):
        """See :meth:`pybamm.Symbol._base_evaluate()`."""
        if t is None:
            raise ValueError("t must be provided")
        return t

    def _evaluate_for_shape(self):
        """
        Return the scalar '0' to represent the shape of the independent variable `Time`.
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        return 0

    def to_equation(self):
        """Convert the node and its subtree into a SymPy equation."""
        return sympy.Symbol("t")


class SpatialVariable(IndependentVariable):
    """
    A node in the expression tree representing a spatial variable.

    Parameters
    ----------
    domain : iterable of str
        list of domains that this variable is valid over (e.g. "negative electrode")
    auxiliary_domains : dict, optional
        dictionary of auxiliary domains, defaults to empty dict
    domains : dict
        A dictionary equivalent to {'primary': domain, auxiliary_domains}. Either
        'domain' and 'auxiliary_domains', or just 'domains', should be provided
        (not both). In future, the 'domain' and 'auxiliary_domains' arguments may be
        deprecated.
    dimension : str, optional
        Dimension of the spatial variable, used to identify the spatial variable in
        geometries with multiple dimensions.
    """

    def __init__(
        self,
        domain: DomainType,
        auxiliary_domains: AuxiliaryDomainType = None,
        domains: DomainsType = None,
        dimension: Optional[str] = None,
    ) -> None:
        super().__init__(
            domain=domain, auxiliary_domains=auxiliary_domains, domains=domains
        )
        self.dimension = dimension

    def create_copy(
        self,
        new_children=None,
        perform_simplifications=True,
    ):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        return self.__class__(domains=self.domains, dimension=self.dimension)


class SpatialVariableEdge(SpatialVariable):
    """
    A node in the expression tree representing a spatial variable, which evaluates
    on the edges

    Parameters
    ----------
    domain : iterable of str
        list of domains that this variable is valid over (e.g. "cartesian", "spherical
        polar")
    auxiliary_domains : dict, optional
    """

    def __init__(
        self,
        domain: DomainType = None,
        auxiliary_domains: AuxiliaryDomainType = None,
        domains: DomainsType = None,
    ) -> None:
        super().__init__(
            domain=domain, auxiliary_domains=auxiliary_domains, domains=domains
        )

    def _evaluates_on_edges(self, dimension):
        return True


# the independent variable time
t = Time()
