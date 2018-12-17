#
# Variable class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Variable(pybamm.Domain, pybamm.Symbol):
    def __init__(self, name, domain=[], parent=None):
        super().__init__(name, parent=parent, domain=domain)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.name == other.name
            and self.domain == other.domain
        )

    def __hash__(self):
        return hash((self.__class__, self.name, tuple(self.domain)))
