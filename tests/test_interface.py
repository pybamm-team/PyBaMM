#
# Tests for the electrolyte-electrode interface class
#
import pybamm

import numpy as np
import unittest


class TestInterface(unittest.TestCase):
    def setUp(self):
        self.param = pybamm.Parameters()
        self.mesh = pybamm.Mesh(self.param, 50)
        self.interface = pybamm.submodels.Interface()
        self.interface.set_simulation(self.param, self.mesh)

    def test_butler_volmer(self):
        I = self.param.icell(1)
        cn = 0.4 * np.ones_like(self.mesh.xcn)
        cs = 0.5 * np.ones_like(self.mesh.xcs)
        cp = 0.56 * np.ones_like(self.mesh.xcp)
        c = np.concatenate([cn, cs, cp])
        en = np.arcsinh(I / (self.param.iota_ref_n * cn)) + self.param.U_Pb(cn)
        ep = np.arcsinh(
            -I / (self.param.iota_ref_p * cp ** 2 * self.param.cw(cp))
        ) + self.param.U_PbO2(cp)
        e = np.concatenate([en, ep])
        jn = self.interface.butler_volmer("xcn", cn, en)
        js = self.interface.butler_volmer("xcs")
        jp = self.interface.butler_volmer("xcp", cp, ep)
        self.assertTrue(np.allclose(jn, I))
        self.assertTrue(np.all(js == 0))
        self.assertTrue(np.allclose(jp, -I))
        j = self.interface.butler_volmer("xc", c, e)
        self.assertTrue(np.allclose(j, np.concatenate([jn, js, jp])))

    def test_uniform_current_density(self):
        t = 1
        I = self.param.icell(t)
        jn = self.interface.uniform_current_density("xcn", t)
        js = self.interface.uniform_current_density("xcs", t)
        jp = self.interface.uniform_current_density("xcp", t)
        self.assertTrue(np.allclose(jn, I / self.param.ln))
        self.assertTrue(np.all(js == 0))
        self.assertTrue(np.allclose(jp, -I / self.param.lp))
        j = self.interface.uniform_current_density("xc", t)
        self.assertTrue(np.allclose(j, np.concatenate([jn, js, jp])))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
