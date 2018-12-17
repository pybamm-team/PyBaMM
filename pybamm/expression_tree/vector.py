#
# Vector class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Vector(pybamm.Array):
    def __init__(self, entries, name=None):
        super().__init__(entries, name=name)
