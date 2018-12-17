#
# Finite Volume discretisation class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np
from scipy.sparse import spdiags


class FiniteVolumeDiscretisation(pybamm.MatrixVectorDiscretisation):
    def gradient_matrix(self, domain):
        for dom in domain:
            submesh = getattr(self.mesh, dom)
            n = submesh.npts
            e = np.ones(n)
            data = np.concatenate([-e, e])
            diags = np.array([0, 1])
            # concatenate
            matrix = 1 / submesh.d_centres * spdiags(data, diags, n, n + 1)
        return matrix

    def divergence_matrix(self, domain):
        for dom in domain:
            submesh = getattr(self.mesh, dom)
            n = submesh.npts + 1
            e = np.ones(n)
            data = np.concatenate([-e, e])
            diags = np.array([0, 1])
            matrix = 1 / submesh.d_edges * spdiags(data, diags, n, n + 1)
        return matrix
