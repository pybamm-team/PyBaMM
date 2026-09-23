"""
Tests for the print_name.py
"""

import pybamm


class TestPrintName:
    def test_prettify_print_name(self):
        param = pybamm.LithiumIonParameters()
        param2 = pybamm.LeadAcidParameters()

        # Test PRINT_NAME_OVERRIDES
        assert param.current_with_time.print_name == "I"

        # Test superscripts
        assert param.n.prim.c_init.print_name == r"c_{\mathrm{n}}^{\mathrm{init}}"

        # Test subscripts
        assert param.n.C_dl(0).print_name == r"C_{\mathrm{dl,n}}"

        # Test bar
        c_e_av = pybamm.Variable("c_e_av")
        c_e_av.print_name = "c_e_av"
        assert c_e_av.print_name == r"\overline{c}_{\mathrm{e}}"

        # Test greek letters
        assert param2.delta.print_name == r"\delta"

        # Test create_copy()
        a_n = param2.n.prim.a
        assert a_n.create_copy().print_name == r"a_{\mathrm{n}}"

        # Test eps
        eps_n = pybamm.Variable("eps_n")
        assert eps_n.print_name == r"\epsilon_{\mathrm{n}}"

        eps_n = pybamm.Variable("eps_c_e_n")
        assert eps_n.print_name == r"(\epsilon c)_{\mathrm{e,n}}"

        # tplus
        t_plus = pybamm.Variable("t_plus")
        assert t_plus.print_name == r"t_{\mathrm{+}}"
