"""
Tests for the print_name.py
"""
import unittest

import pybamm


class TestPrintName(unittest.TestCase):
    def test_prettify_print_name(self):
        param = pybamm.LithiumIonParameters()
        param1 = pybamm.standard_variables
        param2 = pybamm.LeadAcidParameters()

        # Test PRINT_NAME_OVERRIDES
        self.assertEqual(param.timescale.print_name, r"\tau")

        # Test superscripts
        self.assertEqual(param.n.U_ref.print_name, r"U_{n}^{ref}")
        self.assertEqual(param.D_e_typ.print_name, r"D_{e}^{typ}")

        # Test subscripts
        self.assertEqual(param.p.prim.a_R.print_name, r"a_{R\,p}")

        # Test dim and dimensional
        self.assertEqual(
            param.n.prim.j0_ref_dimensional.print_name, r"\hat{j0}_{n}^{ref}"
        )
        self.assertEqual(param.n.C_dl_dimensional.print_name, r"\hat{C}_{dl\,n}")

        # Test bar
        self.assertEqual(param1.c_e_av.print_name, r"\bar{c}_{e}")

        # Test greek letters
        self.assertEqual(param2.delta.print_name, r"\delta")

        # Test new_copy()
        a_n = param2.n.prim.a
        a_n.new_copy()

        # Test eps
        self.assertEqual(param1.eps_n.print_name, r"\epsilon_n")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
