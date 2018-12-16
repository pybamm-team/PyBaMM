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
        submesh = getattr(self.mesh, domain)
        n = submesh.npts
        e = np.ones(n)
        data = np.concatenate([-e, e])
        diags = np.array([0, 1])
        return 1 / submesh.d_centres * spdiags(data, diags, n, n + 1)

    def divergence_matrix(self, domain):
        submesh = getattr(self.mesh, domain)
        n = submesh.npts + 1
        e = np.ones(n)
        data = np.concatenate([-e, e])
        diags = np.array([0, 1])
        return 1 / submesh.d_edges * spdiags(data, diags, n, n + 1)
