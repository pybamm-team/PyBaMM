#
# Tests for the electrolyte-electrode interface class
#
import pybamm

import numpy as np
import unittest


class VarsForTesting(object):
    def __init__(self, t, cn, cp, en, ep):
        self.neg = {"t": t, "c": cn, "e": en}
        self.pos = {"t": t, "c": cp, "e": ep}


class TestInterface(unittest.TestCase):
    def setUp(self):
        self.param = pybamm.Parameters()
        self.mesh = pybamm.Mesh(self.param, target_npts=50)

    def test_butler_volmer(self):
        param_n = self.param.neg_reactions
        param_p = self.param.pos_reactions
        bv_neg = pybamm.interface.ButlerVolmer(param_n, self.mesh.xn)
        bv_pos = pybamm.interface.ButlerVolmer(param_p, self.mesh.xp)
        I = self.param.icell(1)
        cn = 0.4 * np.ones_like(self.mesh.xn.centres)
        cp = 0.56 * np.ones_like(self.mesh.xp.centres)
        en = np.arcsinh(I / (param_n.j0(cn))) + param_n.U(cn)
        ep = np.arcsinh(-I / (param_p.j0(cp))) + param_p.U(cp)
        vars = VarsForTesting(0, cn, cp, en, ep)
        jn = bv_neg.reaction(vars.neg)
        jp = bv_pos.reaction(vars.pos)
        np.testing.assert_allclose(jn, I)
        np.testing.assert_allclose(jp, -I)

    @unittest.skip("not yet implemented")
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
