from pybamm.parameters import Parameters
from pybamm.mesh import Mesh, UniformMesh
from pybamm.spatial_operators import Operators
from pybamm.models import components

import numpy as np
from numpy.linalg import norm

import unittest

class TestComponents(unittest.TestCase):

    def test_electrolyte_diffusion_finite_volumes_convergence(self):
        # Finite volume only has h**2 convergence if the mesh is uniform?
        uniform_lengths = {'Ln':1e-3, 'Ls': 1e-3, 'Lp':1e-3}
        param = Parameters(optional_parameters=uniform_lengths)
        # Test convergence
        ns = [100, 200, 400]
        errs = [0]*len(ns)
        mesh_sizes = [0]*len(ns)
        for i, n in enumerate(ns):
            # Set up
            mesh = Mesh(param, n)
            y0 = np.cos(2*np.pi*mesh.xc)
            operators = Operators("Finite Volumes", mesh)
            lbc = np.array([0])
            rbc = np.array([0])
            dydt_exact = - 4 * np.pi**2 * y0

            # Calculate solution and errors
            dydt = components.electrolyte_diffusion(y0, operators, (lbc, rbc))
            errs[i] = norm(dydt-dydt_exact)/norm(dydt_exact)
            mesh_sizes[i] = mesh.n

        # Expect h**2 convergence
        [self.assertLess(errs[i+1]/errs[i],
                         (mesh_sizes[i]/mesh_sizes[i+1])**2 + 0.01)
         for i in range(len(errs)-1)]

    def test_butler_volmer(self):
        param = Parameters()
        I = np.array([1])
        cn = np.array([0.4])
        cs = np.array([0.5])
        cp = np.array([0.56])
        en = np.arcsinh(I / (param.iota_ref_n * cn)) + param.U_Pb(cn)
        ep = (np.arcsinh(- I / (param.iota_ref_p * cp**2 * param.cw(cp)))
              + param.U_PbO2(cp))
        j, jn, jp = components.butler_volmer(param, cn, cs, cp, en, ep)
        self.assertTrue(np.allclose(jn, I))
        self.assertTrue(np.allclose(jp, -I))

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestComponents)
    unittest.TextTestRunner(verbosity=2).run(suite)
