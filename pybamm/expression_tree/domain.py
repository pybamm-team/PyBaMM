#
# Domain class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Domain(object):
    """list of applicable domains

    Parameters
    ----------

    name: str
        the name of the node
    domain : iterable of str, or str
        the list of domains

    """

    def __init__(self, name, domain=[]):
        super().__init__(name)
        if isinstance(domain, str):
            domain = [domain]
        try:
            iter(domain)
        except TypeError:
            raise TypeError("Domain: argument domain is not iterable")
        else:
            for d in domain:
                assert d in pybamm.KNOWN_DOMAINS, ValueError(
                    """domain "{}" is not in known domains ({})""".format(
                        d, str(pybamm.KNOWN_DOMAINS)
                    )
                )
            self.domain = domain

    @property
    def domain(self):
        """list of applicable domains

        Returns
        -------
            iterable of str
        """
        return self._domain

    @domain.setter
    def domain(self, domain):
        self._domain = domain
