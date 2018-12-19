#
# IndependentVariable class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class IndependentVariable(pybamm.Domain, pybamm.Symbol):
    """A node in the expression tree represending an independent variable

    Used for expressing functions depending on a spatial variable or time

    A variable has a list of domains (text) that it is valid over
    (inherits from :class:`.Domain`)

    Arguments:

    ``name`` (str)
        name of the node
    ``domain`` (iterable of str)
        list of domains that this variable is valid over


    *Extends:* :class:`Domain` and :class:`Symbol`
    """

    def __init__(self, name, domain=[]):
        super().__init__(name, domain=domain)


#: the independent variable time
t = IndependentVariable("t")
