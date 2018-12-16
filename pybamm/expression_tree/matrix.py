#
# Matrix class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Matrix(pybamm.Symbol):
    """
    Parameters
    ----------
    entries : :class:`numpy.array`
        Entries of the matrix
    """

    def __init__(self, entries, name=None, parent=None):
        super().__init__(name, parent)
        self._entries = entries
        self.nrows, self.ncols = entries.shape

    def evaluate(self, y):
        return self._entries

    def __mul__(self, other):
        if isinstance(other, pybamm.Vector):
            return pybamm.MatrixVectorMultiplication(self, other)
        elif isinstance(other, pybamm.Symbol):
            return pybamm.Multiplication(self, other)
        else:
            raise NotImplementedError
