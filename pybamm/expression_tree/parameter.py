#
# Parameter class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Parameter(pybamm.Symbol):
    def __init__(self, name, family=None, parent=None):
        super().__init__(name, parent)
        self.family = family

    @property
    def family(self):
        return self._family

    @family.setter
    def family(self, family):
        self._family = family
