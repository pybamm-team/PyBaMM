#
# Expression tree class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import pybamm
import anytree


class Symbol(anytree.Node):
    def __init__(self, name, parent=None):
        super(Symbol, self).__init__(id=name, parent=parent)
        self._name = name
