#
# Concatenation classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class Concatenation(pybamm.Symbol):
    def __init__(self, *children, name=None, parent=None):
        super().__init__(name, children, parent)

    def evaluate(self, y):
        raise NotImplementedError


class NumpyConcatenation(Concatenation):
    def __init__(self, *children, name=None, parent=None):
        # Convert any Scalar symbols in children to Vector for concatenation
        children = list(children)
        for i, child in enumerate(children):
            if isinstance(child, pybamm.Scalar):
                children[i] = pybamm.Vector(np.array([child.value]))

        super().__init__(*children, name=name, parent=parent)

    def evaluate(self, y):
        return np.concatenate([child.evaluate(y) for child in self.children])
