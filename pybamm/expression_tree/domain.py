#
# Domain class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Domain(object):
    def __init__(self, name, parent=None, domain=[]):
        super().__init__(name, parent=parent)
        try:
            iterator = iter(domain)
        except TypeError:
            raise TypeError('Domain: argument domain is not iterable')
        else:
            self.domain = domain

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        self._domain = domain
