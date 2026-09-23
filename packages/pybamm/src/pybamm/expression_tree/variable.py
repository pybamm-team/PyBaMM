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


def _process_bounds(
    values: tuple[Numeric, Numeric] | None,
) -> tuple[pybamm.Symbol, pybamm.Symbol]:
    """Validate ``(lower, upper)`` bounds (default unbounded) and convert them to symbols."""
    if values is None:
        values = (-np.inf, np.inf)

    if all(isinstance(b, numbers.Number) for b in values) and values[0] >= values[1]:
        raise ValueError(
            f"Invalid bounds {values}. "
            + "Lower bound should be strictly less than upper bound."
        )

    if len(values) != 2:
        raise ValueError(f"Invalid bounds {values}. Must be a tuple of length 2.")
    lb, ub = values

    return (pybamm.convert_to_symbol(lb), pybamm.convert_to_symbol(ub))


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

    __slots__ = ("_bounds", "_reference", "_scale")

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
        self._bounds = _process_bounds(bounds)
        super().__init__(
            name,
            domain=domain,
            auxiliary_domains=auxiliary_domains,
            domains=domains,
        )

        if print_name is None:
            print_name = name  # use name by default
        self.print_name = print_name

    _leaf_fields = ("_scale", "_reference", "_bounds")

    bounds = pybamm.expression_tree.legacy_mutation.bounds_property

    def create_copy(
        self,
        new_children=None,
        perform_simplifications=True,
        scale: Numeric | pybamm.Symbol | None = None,
        reference: Numeric | pybamm.Symbol | None = None,
        bounds: tuple[Numeric, Numeric] | None = None,
    ):
        """
        See :meth:`pybamm.Symbol.new_copy()`.

        Parameters
        ----------
        scale, reference, bounds : optional
            Values for the copy. Any left as ``None`` are taken from ``self``.
        """
        return self.__class__(
            self.name,
            domains=self._domains,
            bounds=self.bounds if bounds is None else bounds,
            print_name=self._raw_print_name,
            scale=self.scale if scale is None else scale,
            reference=self.reference if reference is None else reference,
        )

    def _evaluate_for_shape(self):
        """See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`"""
        return pybamm.evaluate_for_shape_using_domain(self._domains)

    def to_equation(self):
        """Convert the node and its subtree into a SymPy equation."""
        if self.print_name is not None:
            return sympy.Symbol(self.print_name)
        else:
            return self.name

    def to_json(self):
        return {
            "name": self.name,
            "domains": self._domains,
            "children": [self._scale, self._reference, self.bounds[0], self.bounds[1]],
            "print_name": self._raw_print_name,
        }

    @classmethod
    def _from_json(cls, snippet):
        children = snippet.get("children") or []
        if len(children) == 4:
            # Kernel-era shape: scale/reference/bounds carried as children.
            scale, reference, lower, upper = children
            bounds = (lower, upper)
        else:
            # Legacy compact shape: scale/reference defaulted, bounds stored as a
            # top-level [lower, upper] pair of (undecoded) nodes, or None for the
            # default. The kernel only decodes "children", so decode bounds here.
            from pybamm.expression_tree.operations.serialise_kernel import decode

            scale = reference = None
            bounds = snippet.get("bounds")
            if bounds is not None:
                bounds = tuple(decode(b) for b in bounds)
        return cls(
            snippet["name"],
            domains=snippet["domains"],
            scale=scale,
            reference=reference,
            bounds=bounds,
            print_name=snippet.get("print_name"),
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

    __slots__ = ()

    def diff(self, variable: pybamm.Symbol):
        if variable == self:
            return pybamm.Scalar(1)
        elif variable == pybamm.t:
            # reference gets differentiated out
            return pybamm.VariableDot(
                self.name + "'", domains=self._domains, scale=self.scale
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

    __slots__ = ()

    def get_variable(self) -> pybamm.Variable:
        """
        return a :class:`.Variable` corresponding to this VariableDot

        Note: Variable._jac adds a dash to the name of the corresponding VariableDot, so
        we remove this here
        """
        return Variable(self.name[:-1], domains=self._domains, scale=self.scale)

    def diff(self, variable: pybamm.Symbol) -> pybamm.Scalar:
        if variable == self:
            return pybamm.Scalar(1)
        elif variable == pybamm.t:
            raise pybamm.ModelError("cannot take second time derivative of a Variable")
        else:
            return pybamm.Scalar(0)
