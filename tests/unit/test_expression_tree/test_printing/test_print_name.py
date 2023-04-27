"""
Tests for the print_name.py
"""
import unittest

import pybamm


class TestPrintName(unittest.TestCase):
    def test_prettify_print_name(self):
        param = pybamm.LithiumIonParameters()
        param2 = pybamm.LeadAcidParameters()

        # Test PRINT_NAME_OVERRIDES
        self.assertEqual(param.current_with_time.print_name, "I")

        # Test superscripts
        self.assertEqual(param.n.prim.U_init.print_name, r"U_{n}^{init}")

        # Test subscripts
        self.assertEqual(param.n.C_dl.print_name, r"C_{dl\,n}")

        # Test bar
        c_e_av = pybamm.Variable("c_e_av")
        c_e_av.print_name = "c_e_av"
        self.assertEqual(c_e_av.print_name, r"\bar{c}_{e}")

        # Test greek letters
        self.assertEqual(param2.delta.print_name, r"\delta")

        # Test new_copy()
        a_n = param2.n.prim.a
        a_n.new_copy()

        # Test eps
        eps_n = pybamm.Variable("eps_n")
        eps_n.print_name = "eps_n"
        self.assertEqual(eps_n.print_name, r"\epsilon_n")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
