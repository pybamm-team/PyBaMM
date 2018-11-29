#
# Tests for the electrolyte-electrode interface class
#
import pybamm
from tests.shared import VarsForTesting

import numpy as np
import unittest


class TestInterface(unittest.TestCase):
    def setUp(self):
        self.param = pybamm.Parameters()
        self.param_n = self.param.neg_reactions
        self.param_p = self.param.pos_reactions
        self.mesh = pybamm.Mesh(self.param, target_npts=50)

    def tearDown(self):
        del self.param_n
        del self.param_p
        del self.mesh

    def test_butler_volmer(self):
        # Set up
        bv_neg = pybamm.interface.ButlerVolmer(self.param_n, self.mesh.xn)
        bv_pos = pybamm.interface.ButlerVolmer(self.param_p, self.mesh.xp)
        I = self.param.icell(1)
        cn = 0.4 * np.ones_like(self.mesh.xn.centres)
        cp = 0.56 * np.ones_like(self.mesh.xp.centres)
        en = np.arcsinh(I / (self.param_n.j0(cn))) + self.param_n.U(cn)
        ep = np.arcsinh(-I / (self.param_p.j0(cp))) + self.param_p.U(cp)

        vars_n = VarsForTesting(c=cn, e=en)
        vars_p = VarsForTesting(c=cp, e=ep)

        # Test
        jn = bv_neg.reaction(vars_n)
        jp = bv_pos.reaction(vars_p)
        np.testing.assert_allclose(jn, I)
        np.testing.assert_allclose(jp, -I)

    def test_uniform_current_density(self):
        # Set up
        t = 1
        I = self.param.icell(t)
        bv_neg = pybamm.interface.HomogeneousReaction(self.param_n, self.mesh.xn)
        bv_pos = pybamm.interface.HomogeneousReaction(self.param_p, self.mesh.xp)

        vars = VarsForTesting(t=t)

        # Test
        jn = bv_neg.reaction(vars)
        jp = bv_pos.reaction(vars)
        np.testing.assert_allclose(jn * self.param_n.l, I)
        np.testing.assert_allclose(jp * self.param_p.l, -I)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
