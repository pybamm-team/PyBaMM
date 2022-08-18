#
# Standard variables for the models
#
import pybamm
import numpy as np


class StandardVariables:
    def __init__(self):
        # Discharge capacity and energy
        self.Q_Ah = pybamm.Variable("Discharge capacity [A.h]")
        self.Q_Wh = pybamm.Variable("Discharge energy [W.h]")

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
