#
# Variable class
#

import numpy as np
import sympy

import pybamm


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

    *Extends:* :class:`Symbol`
    """

    def __init__(
        self,
        name,
        domain=None,
        auxiliary_domains=None,
        domains=None,
        bounds=None,
        print_name=None,
        scale=1,
        reference=0,
    ):
        self._scale = scale
        self._reference = reference
        super().__init__(
            name,
            domain=domain,
            auxiliary_domains=auxiliary_domains,
            domains=domains,
        )
        if bounds is None:
            bounds = (-np.inf, np.inf)
        else:
            if bounds[0] >= bounds[1]:
                raise ValueError(
                    "Invalid bounds {}. ".format(bounds)
                    + "Lower bound should be strictly less than upper bound."
                )
        self.bounds = bounds
        self.print_name = print_name

    def set_id(self):
        self._id = hash(
            (self.__class__, self.name, self.scale, self.reference)
            + tuple([(k, tuple(v)) for k, v in self.domains.items() if v != []])
        )

    def create_copy(self):
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

    *Extends:* :class:`VariableBase`
    """

    def diff(self, variable):
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

    *Extends:* :class:`VariableBase`
    """

    def get_variable(self):
        """
        return a :class:`.Variable` corresponding to this VariableDot

        Note: Variable._jac adds a dash to the name of the corresponding VariableDot, so
        we remove this here
        """
        return Variable(self.name[:-1], domains=self.domains, scale=self.scale)

    def diff(self, variable):
        if variable == self:
            return pybamm.Scalar(1)
        elif variable == pybamm.t:
            raise pybamm.ModelError("cannot take second time derivative of a Variable")
        else:
            return pybamm.Scalar(0)
