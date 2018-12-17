#
# Variable class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Variable(pybamm.Domain, pybamm.Symbol):
    def __init__(self, name, domain=[], parent=None):
        super().__init__(name, parent=parent, domain=domain)

    @property
    def id(self):
        """
        The immutable "identity" of a variable (for identitying y_slices).

        This is identical to what we'd put in a __hash__ function
        However, implementing __hash__ requires also implementing __eq__,
        which would then mess with loop-checking in anytree
        """
        return hash((self.__class__, self.name, tuple(self.domain)))
