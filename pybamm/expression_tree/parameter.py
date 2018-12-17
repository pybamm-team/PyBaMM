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

    """

    def __init__(self, name, family=None, domain=[]):
        """
        Args:
            name (str): name of the node
            domain (iterable of str): list of domains the parameter is valid
                over
        """
        super().__init__(name, parent=parent, domain=domain)
        self.family = family

    @property
    def family(self):
        return self._family

    @family.setter
    def family(self, family):
        self._family = family
