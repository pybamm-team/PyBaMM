#
# Domain class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Domain(object):
    """list of applicable domains

    Arguments:

    ``name`` (str)
        the name of the node
    ``domain`` (iterable of str)
        the list of domains

    """

    def __init__(self, name, domain=[]):
        super().__init__(name)
        try:
            iterator = iter(domain)
        except TypeError:
            raise TypeError('Domain: argument domain is not iterable')
        else:
            self.domain = domain

    @property
    def domain(self):
        """list of applicable domains (iterable of str)"""
        return self._domain

    @domain.setter
    def domain(self, domain):
        self._domain = domain
