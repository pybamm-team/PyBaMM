#
# Parameter class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Parameter(pybamm.Domain, pybamm.Symbol):
    """A node in the expression tree representing a parameter

    This node will be replaced by a :class:`.Scalar` node by :class`.Parameter`

    A variable has a list of domains (text) that it is valid over
    (inherits from :class:`.Domain`)

    Arguments:

    ``name`` (str)
        name of the node
    ``domain`` (iterable of str)
        list of domains the parameter is valid over

    """

    def __init__(self, name, domain=[]):
        super().__init__(name, domain=domain)
