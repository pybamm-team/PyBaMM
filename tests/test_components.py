from pybat_lead_acid.parameters import Parameters
from pybat_lead_acid.mesh import Mesh, UniformMesh
from pybat_lead_acid.spatial_operators import get_spatial_operators
from pybat_lead_acid.models import components

import numpy as np
from numpy.linalg import norm

import unittest

class TestComponents(unittest.TestCase):

    def test_simple_diffusion_finite_volumes_convergence(self):
        param = Parameters()
        # Test convergence
        ns = [100, 200, 400]
        errs = [0]*len(ns)
        mesh_sizes = [0]*len(ns)
        for i, n in enumerate(ns):
            # Set up
            mesh = Mesh(param, n)
            y0 = np.cos(2*np.pi*mesh.xc)
            grad, div = get_spatial_operators("Finite Volumes", mesh)
            lbc = np.array([0])
            rbc = np.array([0])
            dydt_exact = - 4 * np.pi**2 * y0

            # Calculate solution and errors
            dydt = components.simple_diffusion(y0, grad, div, lbc, rbc)
            errs[i] = norm(dydt-dydt_exact)/norm(dydt_exact)
            mesh_sizes[i] = mesh.n

        # Expect h**2 convergence
        [self.assertLess(errs[i+1]/errs[i],
                         (mesh_sizes[i]/mesh_sizes[i+1])**2 + 0.01)
         for i in range(len(errs)-1)]

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestComponents)
    unittest.TextTestRunner(verbosity=2).run(suite)
