#
# Utility functions and classes for solvers
#
import numpy as np


class NoMemAllocVertcat:
    """
    Acts like a vertcat, but does not allocate new memory.
    """

    def __init__(self, a, b):
        arrays = [a, b]
        self.arrays = arrays
        self.len_a = a.shape[0]
        self.len_b = b.shape[0]
        self.len = self.len_a + self.len_b
        self._shape = (self.len, 1)

    @property
    def shape(self):
        return self._shape

    def get_value(self, out=None):
        if out is None:
            out = np.empty((self.len, 1))
        out[: self.len_a] = self.arrays[0]
        out[self.len_a :] = self.arrays[1]
        return out
