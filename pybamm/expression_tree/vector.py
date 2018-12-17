#
# Vector class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class Vector(pybamm.Symbol):
    """Vector class.

    Parameters
    ----------
    value : slice or :class:`numpy.array`
        If type is slice: a slice representing which chunk of y to take when evaluating.
        If type is numpy.array: fixed entries of the vector.
    """

    def __init__(self, value, name=None, parent=None):
        super().__init__(name, parent)
        if isinstance(value, slice):
            self._y_slice = value
        elif isinstance(value, np.ndarray):
            self._entries = value
        else:
            raise TypeError(
                "value should be slice or numpy array but instead is {}".format(
                    type(value)
                )
            )

    def evaluate(self, y):
        if hasattr(self, "_entries"):
            return self._entries
        else:
            return y[self._y_slice]
