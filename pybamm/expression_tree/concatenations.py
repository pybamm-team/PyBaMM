#
# Concatenation classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class Concatenation(pybamm.Symbol):
    """A node in the expression tree representing a concatenation of symbols

    **Extends**: :class:`pybamm.Symbol`

    Parameters
    ----------
    children : iterable of :class:`pybamm.Symbol`
        The symbols to concatenate

    """

    def __init__(self, *children, name=None):
        if name is None:
            name = "concatenation"
        super().__init__(name, children)

    def evaluate(self, t, y):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        raise NotImplementedError


class NumpyConcatenation(Concatenation):
    """A node in the expression tree representing a concatenation of symbols.
    Upon evaluation, symbols are concatenated using numpy concatenation.

    **Extends**: :class:`pybamm.Concatenation`

    Parameters
    ----------
    children : iterable of :class:`pybamm.Symbol`
        The symbols to concatenate

    """

    def __init__(self, *children):
        # Convert any Scalar symbols in children to Vector for concatenation
        children = list(children)
        for i, child in enumerate(children):
            if isinstance(child, pybamm.Scalar):
                children[i] = pybamm.Vector(np.array([child.value]))

        super().__init__(*children, name="numpy concatenation")

    def evaluate(self, t, y):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return np.concatenate([child.evaluate(t, y) for child in self.children])
