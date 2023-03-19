#
# Standard parameters for lead-acid battery models
#

import pybamm
from .base_parameters import BaseParameters, NullParameters


class LeadAcidParameters(BaseParameters):
    """
    Standard Parameters for lead-acid battery models
    """

    def __init__(self):
        # Get geometric, electrical and thermal parameters
        self.geo = pybamm.geometric_parameters
        self.elec = pybamm.electrical_parameters
        self.therm = pybamm.thermal_parameters

        # Initialize domain parameters
        self.n = DomainLeadAcidParameters("negative", self)
        self.s = DomainLeadAcidParameters("separator", self)
        self.p = DomainLeadAcidParameters("positive", self)
        self.domain_params = {
            "negative": self.n,
            "separator": self.s,
            "positive": self.p,
        }

        # Set parameters and scales
        self._set_parameters()

    def _set_parameters(self):
        """Defines the dimensional parameters."""

        # Physical constants
        self.R = pybamm.Parameter("Ideal gas constant [J.K-1.mol-1]")
        self.F = pybamm.Parameter("Faraday constant [C.mol-1]")
        self.k_b = pybamm.Parameter("Boltzmann constant [J.K-1]")
        self.q_e = pybamm.Parameter("Elementary charge [C]")

        # Thermal parameters
        self.T_ref = self.therm.T_ref
        self.T_init = self.therm.T_init
        self.T_amb = self.therm.T_amb
        self.h_edge = self.therm.h_edge
        self.h_total = self.therm.h_total
        self.rho_c_p_eff = self.therm.rho_c_p_eff

        # Macroscale geometry
        self.L_x = self.geo.L_x
        self.L_y = self.geo.L_y
        self.L_z = self.geo.L_z
        self.A_cc = self.geo.A_cc
        self.A_cooling = self.geo.A_cooling
        self.V_cell = self.geo.V_cell
        self.L = self.L_x
        self.W = self.L_y
        self.H = self.L_z
        self.A_cc = self.A_cc
        self.delta = self.L_x / self.H

        # Electrical
        self.current_with_time = self.elec.current_with_time
        self.current_density_with_time = self.elec.current_density_with_time
        self.Q = self.elec.Q
        self.R_contact = self.elec.R_contact
        self.n_electrodes_parallel = self.elec.n_electrodes_parallel
        self.n_cells = self.elec.n_cells
        self.voltage_low_cut = self.elec.voltage_low_cut
        self.voltage_high_cut = self.elec.voltage_high_cut

        # Electrolyte properties
        self.c_e_init = pybamm.Parameter(
            "Initial concentration in electrolyte [mol.m-3]"
        )
        self.c_e_init_av = self.c_e_init
        self.V_w = pybamm.Parameter("Partial molar volume of water [m3.mol-1]")
        self.V_plus = pybamm.Parameter("Partial molar volume of cations [m3.mol-1]")
        self.V_minus = pybamm.Parameter("Partial molar volume of anions [m3.mol-1]")
        self.V_e = (
            self.V_minus + self.V_plus
        )  # Partial molar volume of electrolyte [m3.mol-1]
        self.nu_plus = pybamm.Parameter("Cation stoichiometry")
        self.nu_minus = pybamm.Parameter("Anion stoichiometry")
        self.nu = self.nu_plus + self.nu_minus

        # Other species properties
        self.c_ox_init = pybamm.Parameter("Initial oxygen concentration [mol.m-3]")
        self.c_ox_typ = (
            self.c_e_init
        )  # pybamm.Parameter("Typical oxygen concentration [mol.m-3]")

        # Electrode properties
        self.V_Pb = pybamm.Parameter("Molar volume of lead [m3.mol-1]")
        self.V_PbO2 = pybamm.Parameter("Molar volume of lead-dioxide [m3.mol-1]")
        self.V_PbSO4 = pybamm.Parameter("Molar volume of lead sulfate [m3.mol-1]")
        # Oxygen
        self.s_plus_Ox = pybamm.Parameter(
            "Signed stoichiometry of cations (oxygen reaction)"
        )
        self.s_w_Ox = pybamm.Parameter(
            "Signed stoichiometry of water (oxygen reaction)"
        )
        self.s_ox_Ox = pybamm.Parameter(
            "Signed stoichiometry of oxygen (oxygen reaction)"
        )
        self.ne_Ox = pybamm.Parameter("Electrons in oxygen reaction")
        self.U_Ox = pybamm.Parameter("Oxygen reference OCP vs SHE [V]")
        # Hydrogen
        self.s_plus_Hy = pybamm.Parameter(
            "Signed stoichiometry of cations (hydrogen reaction)"
        )
        self.s_hy_Hy = pybamm.Parameter(
            "Signed stoichiometry of hydrogen (hydrogen reaction)"
        )
        self.ne_Hy = pybamm.Parameter("Electrons in hydrogen reaction")
        self.U_Hy = pybamm.Parameter("Hydrogen reference OCP vs SHE [V]")

        # Electrolyte properties
        self.M_w = pybamm.Parameter("Molar mass of water [kg.mol-1]")
        self.M_plus = pybamm.Parameter("Molar mass of cations [kg.mol-1]")
        self.M_minus = pybamm.Parameter("Molar mass of anions [kg.mol-1]")
        self.M_e = self.M_minus + self.M_plus  # Molar mass of electrolyte [kg.mol-1]

        # Other species properties
        self.D_ox = pybamm.Parameter("Oxygen diffusivity [m2.s-1]")
        self.D_hy = pybamm.Parameter("Hydrogen diffusivity [m2.s-1]")
        self.V_ox = pybamm.Parameter(
            "Partial molar volume of oxygen molecules [m3.mol-1]"
        )
        self.V_hy = pybamm.Parameter(
            "Partial molar volume of hydrogen molecules [m3.mol-1]"
        )
        self.M_ox = pybamm.Parameter("Molar mass of oxygen molecules [kg.mol-1]")
        self.M_hy = pybamm.Parameter("Molar mass of hydrogen molecules [kg.mol-1]")

        # SEI parameters (for compatibility)
        self.R_sei = pybamm.Scalar(0)

        for domain in self.domain_params.values():
            domain._set_parameters()

        # Electrolyte volumetric capacity
        self.Q_e_max = (
            (
                self.n.L * self.n.eps_max
                + self.s.L * self.s.eps_max
                + self.p.L * self.p.eps_max
            )
            / self.L_x
            / (self.p.prim.s_plus_S - self.n.prim.s_plus_S)
        )
        self.Q_e_max = self.Q_e_max * self.c_e_init * self.F
        self.capacity = self.Q_e_max * self.n_electrodes_parallel * self.A_cc * self.L_x

        # Initial conditions
        self.q_init = pybamm.Parameter("Initial State of Charge")
        self.ocv_init = self.p.prim.U_init - self.n.prim.U_init

        # Concatenations
        self.epsilon_init = pybamm.concatenation(
            pybamm.FullBroadcast(
                self.n.epsilon_init, ["negative electrode"], "current collector"
            ),
            pybamm.FullBroadcast(
                self.s.epsilon_init, ["separator"], "current collector"
            ),
            pybamm.FullBroadcast(
                self.p.epsilon_init, ["positive electrode"], "current collector"
            ),
        )

        # Some scales
        self.thermal_voltage = self.R * self.T_ref / self.F
        self.I_typ = self.Q / (self.A_cc * self.n_electrodes_parallel)
        self.a_j_scale = self.I_typ / self.n.L

    def t_plus(self, c_e, T):
        """Transference number"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e}
        return pybamm.FunctionParameter("Cation transference number", inputs)

    def D_e(self, c_e, T):
        """Dimensional diffusivity in electrolyte."""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e}
        return pybamm.FunctionParameter("Electrolyte diffusivity [m2.s-1]", inputs)

    def kappa_e(self, c_e, T):
        """Dimensional electrolyte conductivity."""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e}
        return pybamm.FunctionParameter("Electrolyte conductivity [S.m-1]", inputs)

    def c_T(self, c_e, c_ox=0, c_hy=0):
        """
        Total liquid molarity [mol.m-3], from thermodynamics. c_k in [mol.m-3].
        """
        return (
            1
            + (2 * self.V_w - self.V_e) * c_e
            + (self.V_w - self.V_ox) * c_ox
            + (self.V_w - self.V_hy) * c_hy
        ) / self.V_w

    def m(self, c_e):
        """
        Dimensional electrolyte molar mass [mol.kg-1], from thermodynamics.
        c_e in [mol.m-3].
        """
        return c_e * self.V_w / ((1 - c_e * self.V_e) * self.M_w)

    def chiRT_over_Fc(self, c_e, T):
        """
        chi * RT/F / c,
        as it appears in the electrolyte potential equation
        """
        return self.chi(c_e, T) * self.R * T / c_e / self.F

    def chi(self, c_e, T, c_ox=0, c_hy=0):
        """Thermodynamic factor"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e}
        chi = pybamm.FunctionParameter("Darken thermodynamic factor", inputs)
        return (
            chi
            * (2 * (1 - self.t_plus(c_e, T)))
            / (self.V_w * self.c_T(c_e, c_ox, c_hy))
        )


class DomainLeadAcidParameters(BaseParameters):
    def __init__(self, domain, main_param):
        self.domain = domain
        self.main_param = main_param

        self.geo = getattr(main_param.geo, domain[0])
        self.therm = getattr(main_param.therm, domain[0])

        if domain != "separator":
            self.prim = PhaseLeadAcidParameters("primary", self)
        else:
            self.prim = NullParameters()

        self.phase_params = {"primary": self.prim}

    def _set_parameters(self):
        domain = self.domain
        Domain = self.domain.capitalize()
        main = self.main_param

        # Macroscale geometry
        self.L = self.geo.L
        # In lead-acid the current collector and electrodes are the same (same
        # thickness)
        self.L_cc = self.L

        # Thermal
        self.rho_c_p = self.therm.rho_c_p
        self.lambda_ = self.therm.lambda_
        self.h_cc = self.therm.h_cc
        self.h_tab = self.therm.h_tab

        if self.domain == "separator":
            self.eps_max = pybamm.Parameter("Maximum porosity of separator")
            self.epsilon_init = self.eps_max
            self.b_e = self.geo.b_e
            self.epsilon_inactive = pybamm.Scalar(0)
            return
        # for lead-acid the electrodes and current collector are the same
        self.rho_c_p_cc = self.therm.rho_c_p
        self.lambda_cc = self.therm.lambda_

        # Microstructure
        self.b_e = self.geo.b_e
        self.b_s = self.geo.b_s
        self.xi = pybamm.Parameter(f"{Domain} electrode morphological parameter")
        self.d = pybamm.Parameter(f"{Domain} electrode pore size [m]")
        self.eps_max = pybamm.Parameter(f"Maximum porosity of {domain} electrode")
        self.epsilon_init = self.eps_max
        # no binder
        self.epsilon_inactive = pybamm.Scalar(0)

        for phase in self.phase_params.values():
            phase._set_parameters()

        # Electrode properties
        if self.domain == "negative":
            DeltaVsurf = (
                main.V_Pb - main.V_PbSO4
            )  # Net Molar Volume consumed in neg electrode [m3.mol-1]
            DeltaVliq = (
                main.V_minus - main.V_plus
            )  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]
        elif self.domain == "positive":
            DeltaVsurf = (
                main.V_PbSO4 - main.V_PbO2
            )  # Net Molar Volume consumed in pos electrode [m3.mol-1]
            DeltaVliq = (
                2 * main.V_w - main.V_minus - 3 * main.V_plus
            )  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]

        self.DeltaVsurf = DeltaVsurf / self.prim.ne_S
        self.DeltaVliq = DeltaVliq / self.prim.ne_S
        self.DeltaV = self.DeltaVsurf + self.DeltaVliq

        self.Q_max = pybamm.Parameter(f"{Domain} electrode volumetric capacity [C.m-3]")
        self.C_dl = pybamm.Parameter(
            f"{Domain} electrode double-layer capacity [F.m-2]"
        )

        # In lead-acid the current collector and electrodes are the same (same
        # conductivity) but we correct here for Bruggeman. Note that because for
        # lithium-ion we allow electrode conductivity to be a function of temperature,
        # but not the current collector conductivity, here the latter is evaluated at
        # T_ref.
        self.sigma_cc = self.sigma(main.T_ref) * (1 - self.eps_max) ** self.b_s

    def sigma(self, T):
        """Dimensional electrical conductivity"""
        inputs = {"Temperature [K]": T}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} electrode conductivity [S.m-1]", inputs
        )


class PhaseLeadAcidParameters(BaseParameters):
    def __init__(self, phase, domain_param):
        self.phase = phase

        self.domain_param = domain_param
        self.domain = domain_param.domain
        self.main_param = domain_param.main_param
        self.geo = domain_param.geo.prim

    def _set_parameters(self):
        main = self.main_param
        domain, Domain = self.domain_Domain  # Microstructure
        x = pybamm.SpatialVariable(
            f"x_{domain[0]}",
            domain=[f"{domain} electrode"],
            auxiliary_domains={"secondary": "current collector"},
            coord_sys="cartesian",
        )
        self.a = pybamm.FunctionParameter(
            f"{Domain} electrode surface area to volume ratio [m-1]",
            {"Through-cell distance (x) [m]": x},
        )

        # Microstructure
        self.epsilon_s = 1 - self.domain_param.eps_max

        # Electrochemical reactions
        # Main
        if domain == "negative":
            self.s_plus_S = pybamm.Scalar(1)
        elif domain == "positive":
            self.s_plus_S = pybamm.Scalar(3)
        self.ne_S = pybamm.Scalar(2)
        self.ne = self.ne_S
        self.s_plus_S = self.s_plus_S / self.ne_S
        self.alpha_bv = pybamm.Parameter(
            f"{Domain} electrode Butler-Volmer transfer coefficient"
        )

        # Initial conditions
        self.U_init = self.U(main.c_e_init, main.T_init)

    def U(self, c_e, T):
        """Dimensional open-circuit voltage [V]"""
        inputs = {"Electrolyte molar mass [mol.kg-1]": self.main_param.m(c_e)}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} electrode open-circuit potential [V]", inputs
        )

    def j0(self, c_e, T):
        """Dimensional exchange-current density [A.m-2]"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} electrode exchange-current density [A.m-2]", inputs
        )

    def j0_Ox(self, c_e, T):
        """Dimensional oxygen electrode exchange-current density [A.m-2]"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} electrode oxygen exchange-current density [A.m-2]", inputs
        )
