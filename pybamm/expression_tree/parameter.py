#
# Parameter class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Parameter(pybamm.Symbol):
    def __init__(self, name, family=None, parent=None):
        super().__init__(name, parent)
        self._family = family
