#
# Variable class
#
import pybamm
import numbers
import numpy as np


class Variable(pybamm.Symbol):
    """A node in the expression tree represending a dependent variable

    This node will be discretised by :class:`.Discretisation` and converted
    to a :class:`pybamm.StateVector` node.

    Parameters
    ----------

    name : str
        name of the node
    domain : iterable of str
        list of domains that this variable is valid over
    auxiliary_domains : dict
        dictionary of auxiliary domains ({'secondary': ..., 'tertiary': ...}). For
        example, for the single particle model, the particle concentration would be a
        Variable with domain 'negative particle' and secondary auxiliary domain 'current
        collector'. For the DFN, the particle concentration would be a Variable with
        domain 'negative particle', secondary domain 'negative electrode' and tertiary
        domain 'current collector'

    *Extends:* :class:`Symbol`
    """

    def __init__(self, name, domain=None, auxiliary_domains=None):
        if domain is None:
            domain = []
        if auxiliary_domains is None:
            auxiliary_domains = {}
        super().__init__(name, domain=domain, auxiliary_domains=auxiliary_domains)

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return self.__class__(self.name, self.domain, self.auxiliary_domains)

    def _evaluate_for_shape(self):
        """ See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()` """
        return pybamm.evaluate_for_shape_using_domain(
            self.domain, self.auxiliary_domains
        )


class ExternalVariable(Variable):
    """A node in the expression tree represending an external variable variable

    This node will be discretised by :class:`.Discretisation` and converted
    to a :class:`.Vector` node.

    Parameters
    ----------

    name : str
        name of the node
    domain : iterable of str
        list of domains that this variable is valid over
    auxiliary_domains : dict
        dictionary of auxiliary domains ({'secondary': ..., 'tertiary': ...}). For
        example, for the single particle model, the particle concentration would be a
        Variable with domain 'negative particle' and secondary auxiliary domain 'current
        collector'. For the DFN, the particle concentration would be a Variable with
        domain 'negative particle', secondary domain 'negative electrode' and tertiary
        domain 'current collector'

    *Extends:* :class:`pybamm.Variable`
    """

    def __init__(self, name, size, domain=None, auxiliary_domains=None):
        self._size = size
        super().__init__(name, domain, auxiliary_domains)

    @property
    def size(self):
        return self._size

    def _evaluate_for_shape(self):
        """ See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()` """
        return np.nan * np.ones((self.size, 1))

    def _base_evaluate(self, t=None, y=None, u=None):
        # u should be a dictionary
        # convert 'None' to empty dictionary for more informative error
        if u is None:
            u = {}
        if not isinstance(u, dict):
            # if the special input "shape test" is passed, just return 1
            if u == "shape test":
                return self.evaluate_for_shape()
            raise TypeError("inputs u should be a dictionary")
        try:
            out = u[self.name]
            if isinstance(out, numbers.Number) or out.shape[0] == 1:
                return out * np.ones((self.size, 1))
            elif out.shape[0] != self.size:
                raise ValueError(
                    "External variable input has size {} but should be {}".format(
                        out.shape[0], self.size
                    )
                )
            else:
                return out
        # raise more informative error if can't find name in dict
        except KeyError:
            raise KeyError("External variable '{}' not found".format(self.name))
