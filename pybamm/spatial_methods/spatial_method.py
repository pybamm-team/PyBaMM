#
# A general spatial method class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np
from scipy.sparse import spdiags


class SpatialMethod:
    """
    A general spatial methods class. 
    All spatial methods will follow the general form of SpatialMethod in 
    that they contain a method for broadcasting variables onto a mesh, 
    a gradient operator, and a diverence operator.

    Parameters
    ----------
    """

    def __init__(self, mesh):

        self.mesh = mesh

    def spatial_variable(self):
        """this should return a vectorised spatial variable
        with nodes at the correct locations for the spatial method"""
        raise NotImplementedError

    def broadcast(self):
        """this should return a vectorised variable that of length 
        equal to the number of independent variables that the spatial
        method created in discretising a variable"""
        raise NotImplementedError

    def gradient(self):
        raise NotImplementedError

    def divergence(self):
        raise NotImplementedError
