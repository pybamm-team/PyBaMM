#
# Variable class
#
import numpy as np
import pybamm


class Variable(pybamm.Symbol):
    """A node in the expression tree represending a dependent variable

    This node will be discretised by :class:`.Discretisation` and converted
    to a :class:`.Vector` node.

    Parameters
    ----------

    name : str
        name of the node
    domain : iterable of str
        list of domains that this variable is valid over


    *Extends:* :class:`Symbol`
    """

    def __init__(self, name, domain=[]):
        super().__init__(name, domain=domain)

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return Variable(self.name, self.domain)

    def evaluate_for_shape(self):
        """
        Returns a vector of NaNs to represent the shape of a Variable.
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        return np.nan * np.ones(1)
