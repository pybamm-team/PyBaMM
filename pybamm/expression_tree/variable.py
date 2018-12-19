#
# Variable class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Variable(pybamm.Domain, pybamm.Symbol):
    """A node in the expression tree represending a dependent variable

    This node will be discretised by :class:`.BaseDiscretisation` and converted
    to a :class:`.Vector` node.

    A variable has a list of domains (text) that it is valid over
    (inherits from :class:`.Domain`)

    Parameters
    ----------

    name : str
        name of the node
    domain : iterable of str
        list of domains that this variable is valid over


    *Extends:* :class:`Domain` and :class:`Symbol`
    """

    def __init__(self, name, domain=[]):
        super().__init__(name, domain=domain)
