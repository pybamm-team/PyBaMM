#
# Standard variables for the models
#
import pybamm
import numpy as np


class StandardVariables:
    def __init__(self):
        # Discharge capacity
        self.Q = pybamm.Variable("Discharge capacity [A.h]")

        # Electrolyte concentration
        self.c_e_n = pybamm.Variable(
            "Negative electrolyte concentration",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, np.inf),
        )
        self.c_e_s = pybamm.Variable(
            "Separator electrolyte concentration",
            domain="separator",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, np.inf),
        )
        self.c_e_p = pybamm.Variable(
            "Positive electrolyte concentration",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, np.inf),
        )

        self.c_e_av = pybamm.Variable(
            "X-averaged electrolyte concentration",
            domain="current collector",
            bounds=(0, np.inf),
        )

        # Electrolyte porosity times concentration
        self.eps_c_e_n = pybamm.Variable(
            "Negative electrode porosity times concentration",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, np.inf),
        )
        self.eps_c_e_s = pybamm.Variable(
            "Separator porosity times concentration",
            domain="separator",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, np.inf),
        )
        self.eps_c_e_p = pybamm.Variable(
            "Positive electrode porosity times concentration",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, np.inf),
        )

        self.eps_c_e_av = pybamm.Variable(
            "X-averaged porosity times concentration",
            domain="current collector",
            bounds=(0, np.inf),
        )

        # Electrolyte potential
        self.phi_e_n = pybamm.Variable(
            "Negative electrolyte potential",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.phi_e_s = pybamm.Variable(
            "Separator electrolyte potential",
            domain="separator",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.phi_e_p = pybamm.Variable(
            "Positive electrolyte potential",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )

        # Electrode potential
        self.phi_s_n = pybamm.Variable(
            "Negative electrode potential",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.phi_s_p = pybamm.Variable(
            "Positive electrode potential",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )

        # Potential difference
        self.delta_phi_n = pybamm.Variable(
            "Negative electrode surface potential difference",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.delta_phi_p = pybamm.Variable(
            "Positive electrode surface potential difference",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )

        self.delta_phi_n_av = pybamm.Variable(
            "X-averaged negative electrode surface potential difference",
            domain="current collector",
        )
        self.delta_phi_p_av = pybamm.Variable(
            "X-averaged positive electrode surface potential difference",
            domain="current collector",
        )

        # current collector variables
        self.phi_s_cn = pybamm.Variable(
            "Negative current collector potential", domain="current collector"
        )
        self.phi_s_cp = pybamm.Variable(
            "Positive current collector potential", domain="current collector"
        )
        self.i_boundary_cc = pybamm.Variable(
            "Current collector current density", domain="current collector"
        )
        self.phi_s_cn_composite = pybamm.Variable(
            "Composite negative current collector potential", domain="current collector"
        )
        self.phi_s_cp_composite = pybamm.Variable(
            "Composite positive current collector potential", domain="current collector"
        )
        self.i_boundary_cc_composite = pybamm.Variable(
            "Composite current collector current density", domain="current collector"
        )

        # Particle concentration
        self.c_s_n = pybamm.Variable(
            "Negative particle concentration",
            domain="negative particle",
            auxiliary_domains={
                "secondary": "negative electrode",
                "tertiary": "current collector",
            },
            bounds=(0, 1),
        )
        self.c_s_p = pybamm.Variable(
            "Positive particle concentration",
            domain="positive particle",
            auxiliary_domains={
                "secondary": "positive electrode",
                "tertiary": "current collector",
            },
            bounds=(0, 1),
        )
        self.c_s_n_xav = pybamm.Variable(
            "X-averaged negative particle concentration",
            domain="negative particle",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, 1),
        )
        self.c_s_p_xav = pybamm.Variable(
            "X-averaged positive particle concentration",
            domain="positive particle",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, 1),
        )
        self.c_s_n_rav = pybamm.Variable(
            "R-averaged negative particle concentration",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, 1),
        )
        self.c_s_p_rav = pybamm.Variable(
            "R-averaged positive particle concentration",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, 1),
        )
        self.c_s_n_av = pybamm.Variable(
            "Average negative particle concentration",
            domain="current collector",
            bounds=(0, 1),
        )
        self.c_s_p_av = pybamm.Variable(
            "Average positive particle concentration",
            domain="current collector",
            bounds=(0, 1),
        )
        self.c_s_n_surf = pybamm.Variable(
            "Negative particle surface concentration",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, 1),
        )
        self.c_s_p_surf = pybamm.Variable(
            "Positive particle surface concentration",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, 1),
        )
        self.c_s_n_surf_xav = pybamm.Variable(
            "X-averaged negative particle surface concentration",
            domain="current collector",
            bounds=(0, 1),
        )
        self.c_s_p_surf_xav = pybamm.Variable(
            "X-averaged positive particle surface concentration",
            domain="current collector",
            bounds=(0, 1),
        )
        # Average particle concentration gradient (for polynomial particle concentration
        # models). Note: we make the distinction here between the flux defined as
        # N = -D*dc/dr and the concentration gradient q = dc/dr
        self.q_s_n_rav = pybamm.Variable(
            "R-averaged negative particle concentration gradient",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.q_s_p_rav = pybamm.Variable(
            "R-averaged positive particle concentration gradient",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.q_s_n_av = pybamm.Variable(
            "Average negative particle concentration gradient",
            domain="current collector",
        )
        self.q_s_p_av = pybamm.Variable(
            "Average positive particle concentration gradient",
            domain="current collector",
        )

        # Porosity
        self.eps_n = pybamm.Variable(
            "Negative electrode porosity",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, 1),
        )
        self.eps_s = pybamm.Variable(
            "Separator porosity",
            domain="separator",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, 1),
        )
        self.eps_p = pybamm.Variable(
            "Positive electrode porosity",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, 1),
        )

        # Piecewise constant (for asymptotic models)
        self.eps_n_pc = pybamm.Variable(
            "X-averaged negative electrode porosity",
            domain="current collector",
            bounds=(0, 1),
        )
        self.eps_s_pc = pybamm.Variable(
            "X-averaged separator porosity", domain="current collector", bounds=(0, 1)
        )
        self.eps_p_pc = pybamm.Variable(
            "X-averaged positive electrode porosity",
            domain="current collector",
            bounds=(0, 1),
        )

        # Temperature
        self.T_cn = pybamm.Variable(
            "Negative currents collector temperature", domain="current collector"
        )
        self.T_n = pybamm.Variable(
            "Negative electrode temperature",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.T_s = pybamm.Variable(
            "Separator temperature",
            domain="separator",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.T_p = pybamm.Variable(
            "Positive electrode temperature",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.T_cp = pybamm.Variable(
            "Positive currents collector temperature", domain="current collector"
        )
        self.T_av = pybamm.Variable(
            "X-averaged cell temperature", domain="current collector"
        )
        self.T_vol_av = pybamm.Variable("Volume-averaged cell temperature")

        # SEI variables
        self.L_inner_av = pybamm.Variable(
            "X-averaged inner SEI thickness",
            domain="current collector",
        )
        self.L_inner = pybamm.Variable(
            "Inner SEI thickness",
            domain=["negative electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        self.L_outer_av = pybamm.Variable(
            "X-averaged outer SEI thickness",
            domain="current collector",
        )
        self.L_outer = pybamm.Variable(
            "Outer SEI thickness",
            domain=["negative electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        # For SEI reaction at the li metal/separator interface in a li metal model
        self.L_inner_interface = pybamm.Variable(
            "Inner SEI thickness",
            domain=["current collector"],
        )
        self.L_outer_interface = pybamm.Variable(
            "Outer SEI thickness",
            domain=["current collector"],
        )

        # Interface utilisation
        self.u_n = pybamm.Variable(
            "Negative electrode interface utilisation",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.u_p = pybamm.Variable(
            "Positive electrode interface utilisation",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.u_n_xav = pybamm.Variable(
            "X-averaged negative electrode interface utilisation",
            domain="current collector",
        )
        self.u_p_xav = pybamm.Variable(
            "X-averaged positive electrode interface utilisation",
            domain="current collector",
        )

    def __setattr__(self, name, value):
        value.print_name = name
        super().__setattr__(name, value)


standard_variables = StandardVariables()
