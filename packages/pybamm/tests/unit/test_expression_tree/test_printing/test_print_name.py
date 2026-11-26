"""
Tests for the print_name.py
"""

import pytest

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

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            # The name's own subscripts join the others, to avoid a double subscript
            ("X_j_n", r"X_{\mathrm{j,n}}"),
            ("U0_j_p", r"U_{\mathrm{j,p}}^{\mathrm{0}}"),
            ("lambda__n", r"\lambda_{\mathrm{n}}"),
            ("alpha_SEI_n", r"\alpha_{\mathrm{SEI,n}}"),
            ("Q_Li_particles_init", r"Q_{\mathrm{Li,particles}}^{\mathrm{init}}"),
            ("a_j_scale", r"a_{\mathrm{j,scale}}"),
            # A single subscript of the name's own is kept as it is
            ("U_j", "U_j"),
            # Only whole parts of the name are taken as subscripts
            ("s_plus_S_p", r"s_{\mathrm{plus,S,p}}"),
            ("c_plated_Li_0_p", r"c_{\mathrm{plated,Li,p}}^{\mathrm{0}}"),
            # Names that are not plain identifiers are not split
            ("L_n + L_s", r"L_n + L_{\mathrm{s}}"),
        ],
    )
    def test_prettify_print_name_combines_subscripts(self, name, expected):
        assert pybamm.Variable(name).print_name == expected
