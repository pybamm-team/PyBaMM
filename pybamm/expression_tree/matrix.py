#
# Matrix class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Matrix(pybamm.Array):
    """node in the expression tree that holds a matrix type (e.g. :class:`numpy.array`)

    Arguements:
    ``entries``
        the array associated with the node
    ``name``
        the name of the node

    *Extends:* :class:`Array`
    """

    def __init__(self, entries, name=None):
        super().__init__(entries, name=name)

    def __mul__(self, other):
        if isinstance(other, pybamm.Vector):
            return pybamm.MatrixVectorMultiplication(self, other)
        elif isinstance(other, pybamm.Symbol):
            return pybamm.Multiplication(self, other)
        else:
            raise NotImplementedError
