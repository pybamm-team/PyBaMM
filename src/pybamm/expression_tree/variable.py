#
# Variable class
#
from __future__ import annotations

import numbers

import numpy as np
import sympy

import pybamm
from pybamm.type_definitions import (
    AuxiliaryDomainType,
    DomainsType,
    DomainType,
    Numeric,
)


class VariableBase(pybamm.Symbol):
    """
    A node in the expression tree represending a dependent variable.

    This node will be discretised by :class:`.Discretisation` and converted
    to a :class:`pybamm.StateVector` node.

    Parameters
    ----------
    name : str
        name of the node
    domain : iterable of str
        list of domains that this variable is valid over
    auxiliary_domains : dict
        dictionary of auxiliary domains ({'secondary': ..., 'tertiary': ...,
        'quaternary': ...}). For example, for the single particle model, the particle
        concentration would be a Variable with domain 'negative particle' and secondary
        auxiliary domain 'current collector'. For the DFN, the particle concentration
        would be a Variable with domain 'negative particle', secondary domain
        'negative electrode' and tertiary domain 'current collector'
    domains : dict
        A dictionary equivalent to {'primary': domain, auxiliary_domains}. Either
        'domain' and 'auxiliary_domains', or just 'domains', should be provided
        (not both). In future, the 'domain' and 'auxiliary_domains' arguments may be
        deprecated.
    bounds : tuple, optional
        Physical bounds on the variable
    print_name : str, optional
        The name to use for printing. Default is None, in which case self.name is used.
    scale : float or :class:`pybamm.Symbol`, optional
        The scale of the variable, used for scaling the model when solving. The state
        vector representing this variable will be multiplied by this scale.
        Default is 1.
    reference : float or :class:`pybamm.Symbol`, optional
        The reference value of the variable, used for scaling the model when solving.
        This value will be added to the state vector representing this variable.
        Default is 0.
    """

    def __init__(
        self,
        name: str,
        domain: DomainType = None,
        auxiliary_domains: AuxiliaryDomainType = None,
        domains: DomainsType = None,
        bounds: tuple[Numeric, Numeric] | None = None,
        print_name: str | None = None,
        scale: Numeric | None = None,
        reference: Numeric | None = None,
    ):
        if scale is None:
            scale = 1
        if reference is None:
            reference = 0
        self._scale = pybamm.convert_to_symbol(scale)
        self._reference = pybamm.convert_to_symbol(reference)
        self._bounds = self._process_bounds(bounds)
        super().__init__(
            name,
            domain=domain,
            auxiliary_domains=auxiliary_domains,
            domains=domains,
        )

        if print_name is None:
            print_name = name  # use name by default
        self.print_name = print_name

    def _process_bounds(
        self, values: tuple[Numeric, Numeric] | None
    ) -> tuple[pybamm.Symbol, pybamm.Symbol]:
        if values is None:
            values = (-np.inf, np.inf)

        if (
            all(isinstance(b, numbers.Number) for b in values)
            and values[0] >= values[1]
        ):
            raise ValueError(
                f"Invalid bounds {values}. "
                + "Lower bound should be strictly less than upper bound."
            )

        if len(values) != 2:
            raise ValueError(f"Invalid bounds {values}. Must be a tuple of length 2.")
        lb, ub = values

        return (pybamm.convert_to_symbol(lb), pybamm.convert_to_symbol(ub))

    @property
    def bounds(self) -> tuple[pybamm.Symbol, pybamm.Symbol]:
        """Physical bounds on the variable."""
        return self._bounds

    @bounds.setter
    def bounds(self, values: tuple[Numeric, Numeric]):
        self._bounds = self._process_bounds(values)
        self.set_id()

    def set_id(self):
        domains_tuple = tuple((k, tuple(v)) for k, v in self.domains.items() if v != [])
        self._id = hash(
            (
                self.__class__,
                self._name,
                self._scale,
                self._reference,
                self._bounds,
                domains_tuple,
            )
        )

    def create_copy(
        self,
        new_children=None,
        perform_simplifications=True,
    ):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        return self.__class__(
            self.name,
            domains=self.domains,
            bounds=self.bounds,
            print_name=self._raw_print_name,
            scale=self.scale,
            reference=self.reference,
        )

    def _evaluate_for_shape(self):
        """See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`"""
        return pybamm.evaluate_for_shape_using_domain(self.domains)

    def to_equation(self):
        """Convert the node and its subtree into a SymPy equation."""
        if self.print_name is not None:
            return sympy.Symbol(self.print_name)
        else:
            return self.name

    def to_json(
        self,
    ):
        raise NotImplementedError(
            "pybamm.Variable: Serialisation is only implemented for discretised models."
        )


class Variable(VariableBase):
    """
    A node in the expression tree represending a dependent variable.

    This node will be discretised by :class:`.Discretisation` and converted
    to a :class:`pybamm.StateVector` node.

    Parameters
    ----------

    name : str
        name of the node
        domain : iterable of str, optional
        list of domains that this variable is valid over
    auxiliary_domains : dict, optional
        dictionary of auxiliary domains ({'secondary': ..., 'tertiary': ...,
        'quaternary': ...}). For example, for the single particle model, the particle
        concentration would be a Variable with domain 'negative particle' and secondary
        auxiliary domain 'current collector'. For the DFN, the particle concentration
        would be a Variable with domain 'negative particle', secondary domain
        'negative electrode' and tertiary domain 'current collector'
    domains : dict
        A dictionary equivalent to {'primary': domain, auxiliary_domains}. Either
        'domain' and 'auxiliary_domains', or just 'domains', should be provided
        (not both). In future, the 'domain' and 'auxiliary_domains' arguments may be
        deprecated.
    bounds : tuple, optional
        Physical bounds on the variable
    print_name : str, optional
        The name to use for printing. Default is None, in which case self.name is used.
    scale : float or :class:`pybamm.Symbol`, optional
        The scale of the variable, used for scaling the model when solving. The state
        vector representing this variable will be multiplied by this scale.
        Default is 1.
    reference : float or :class:`pybamm.Symbol`, optional
        The reference value of the variable, used for scaling the model when solving.
        This value will be added to the state vector representing this variable.
        Default is 0.
    """

    def diff(self, variable: pybamm.Symbol):
        if variable == self:
            return pybamm.Scalar(1)
        elif variable == pybamm.t:
            # reference gets differentiated out
            return pybamm.VariableDot(
                self.name + "'", domains=self.domains, scale=self.scale
            )
        else:
            return pybamm.Scalar(0)


class VariableDot(VariableBase):
    """
    A node in the expression tree represending the time derviative of a dependent
    variable

    This node will be discretised by :class:`.Discretisation` and converted
    to a :class:`pybamm.StateVectorDot` node.

    Parameters
    ----------

    name : str
        name of the node
    domain : iterable of str
        list of domains that this variable is valid over
    auxiliary_domains : dict
        dictionary of auxiliary domains ({'secondary': ..., 'tertiary': ...,
        'quaternary': ...}). For example, for the single particle model, the particle
        concentration would be a Variable with domain 'negative particle' and secondary
        auxiliary domain 'current collector'. For the DFN, the particle concentration
        would be a Variable with domain 'negative particle', secondary domain
        'negative electrode' and tertiary domain 'current collector'
    domains : dict
        A dictionary equivalent to {'primary': domain, auxiliary_domains}. Either
        'domain' and 'auxiliary_domains', or just 'domains', should be provided
        (not both). In future, the 'domain' and 'auxiliary_domains' arguments may be
        deprecated.
    bounds : tuple, optional
        Physical bounds on the variable. Included for compatibility with `VariableBase`,
        but ignored.
    print_name : str, optional
        The name to use for printing. Default is None, in which case self.name is used.
    scale : float or :class:`pybamm.Symbol`, optional
        The scale of the variable, used for scaling the model when solving. The state
        vector representing this variable will be multiplied by this scale.
        Default is 1.
    reference : float or :class:`pybamm.Symbol`, optional
        The reference value of the variable, used for scaling the model when solving.
        This value will be added to the state vector representing this variable.
        Default is 0.
    """

    def get_variable(self) -> pybamm.Variable:
        """
        return a :class:`.Variable` corresponding to this VariableDot

        Note: Variable._jac adds a dash to the name of the corresponding VariableDot, so
        we remove this here
        """
        return Variable(self.name[:-1], domains=self.domains, scale=self.scale)

    def diff(self, variable: pybamm.Symbol) -> pybamm.Scalar:
        if variable == self:
            return pybamm.Scalar(1)
        elif variable == pybamm.t:
            raise pybamm.ModelError("cannot take second time derivative of a Variable")
        else:
            return pybamm.Scalar(0)
