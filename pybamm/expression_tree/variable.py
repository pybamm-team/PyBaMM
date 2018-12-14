#
# Variable class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Variable(pybamm.Symbol):
    def __init__(self, name, domain=None, parent=None):
        super().__init__(name, parent)
        self._domain = domain
