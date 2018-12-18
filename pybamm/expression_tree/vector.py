#
# Vector class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Vector(pybamm.Array):
    """node in the expression tree that holds a vector type (e.g. :class:`numpy.array`)

    Arguements:
    ``entries``
        the array associated with the node
    ``name``
        the name of the node

    *Extends:* :class:`Array`
    """

    def __init__(self, entries, name=None):
        super().__init__(entries, name=name)
