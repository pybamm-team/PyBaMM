#
# Variable class
#
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
    auxiliary_domains : dict
        dictionary of auxiliary domains ({'secondary': ..., 'tertiary': ...}). For
        example, for the single particle model, the particle concentration would be a
        Variable with domain 'negative partilce' and secondary auxiliary domain 'current
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
        return Variable(self.name, self.domain, self.auxiliary_domains)

    def evaluate_for_shape(self):
        """ See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()` """
        return pybamm.evaluate_for_shape_using_domain(
            self.domain, self.auxiliary_domains
        )
