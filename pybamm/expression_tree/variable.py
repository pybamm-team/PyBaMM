#
# Variable class
#
import numbers

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

    *Extends:* :class:`Symbol`
    """

    def __init__(
        self, name, domain=None, auxiliary_domains=None, domains=None, bounds=None
    ):
        super().__init__(
            name, domain=domain, auxiliary_domains=auxiliary_domains, domains=domains
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
        self.print_name = None

    def create_copy(self):
        """See :meth:`pybamm.Symbol.new_copy()`."""

        out = self.__class__(self.name, domains=self.domains, bounds=self.bounds)
        return out

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

    *Extends:* :class:`Symbol`
    """

    def __init__(
        self, name, domain=None, auxiliary_domains=None, domains=None, bounds=None
    ):
        super().__init__(
            name,
            domain=domain,
            auxiliary_domains=auxiliary_domains,
            domains=domains,
            bounds=bounds,
        )

    def diff(self, variable):
        if variable.id == self.id:
            return pybamm.Scalar(1)
        elif variable.id == pybamm.t.id:
            return pybamm.VariableDot(self.name + "'", domains=self.domains)
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

    *Extends:* :class:`Symbol`
    """

    def __init__(
        self, name, domain=None, auxiliary_domains=None, domains=None, bounds=None
    ):
        super().__init__(
            name, domain=domain, auxiliary_domains=auxiliary_domains, domains=domains
        )

    def get_variable(self):
        """
        return a :class:`.Variable` corresponding to this VariableDot

        Note: Variable._jac adds a dash to the name of the corresponding VariableDot, so
        we remove this here
        """
        return Variable(self.name[:-1], domains=self.domains)

    def diff(self, variable):
        if variable.id == self.id:
            return pybamm.Scalar(1)
        elif variable.id == pybamm.t.id:
            raise pybamm.ModelError("cannot take second time derivative of a Variable")
        else:
            return pybamm.Scalar(0)


class ExternalVariable(Variable):
    """
    A node in the expression tree representing an external variable variable.

    This node will be discretised by :class:`.Discretisation` and converted
    to a :class:`.Vector` node.

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

    *Extends:* :class:`pybamm.Variable`
    """

    def __init__(self, name, size, domain=None, auxiliary_domains=None, domains=None):
        self._size = size
        super().__init__(name, domain, auxiliary_domains, domains)

    @property
    def size(self):
        return self._size

    def create_copy(self):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        return ExternalVariable(self.name, self.size, domains=self.domains)

    def _evaluate_for_shape(self):
        """See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`"""
        return np.nan * np.ones((self.size, 1))

    def _base_evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        # inputs should be a dictionary
        # convert 'None' to empty dictionary for more informative error
        if inputs is None:
            inputs = {}
        if not isinstance(inputs, dict):
            # if the special input "shape test" is passed, just return 1
            if inputs == "shape test":
                return self.evaluate_for_shape()
            raise TypeError("inputs should be a dictionary")
        try:
            out = inputs[self.name]
            if isinstance(out, numbers.Number) or out.shape[0] == 1:
                return out * np.ones((self.size, 1))
            elif out.shape[0] != self.size:
                raise ValueError(
                    "External variable input has size {} but should be {}".format(
                        out.shape[0], self.size
                    )
                )
            else:
                if isinstance(out, np.ndarray) and out.ndim == 1:
                    out = out[:, np.newaxis]
                return out
        # raise more informative error if can't find name in dict
        except KeyError:
            raise KeyError("External variable '{}' not found".format(self.name))

    def diff(self, variable):
        if variable.id == self.id:
            return pybamm.Scalar(1)
        elif variable.id == pybamm.t.id:
            raise pybamm.ModelError(
                "cannot take time derivative of an external variable"
            )
        else:
            return pybamm.Scalar(0)
