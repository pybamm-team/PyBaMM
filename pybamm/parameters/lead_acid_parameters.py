#
# Standard parameters for lead-acid battery models
#

import pybamm
from .base_parameters import BaseParameters, NullParameters


class LeadAcidParameters(BaseParameters):
    """
    Standard Parameters for lead-acid battery models

    Layout:
        1. Dimensional Parameters
        2. Dimensional Functions
        3. Scalings
        4. Dimensionless Parameters
        5. Dimensionless Functions
        6. Input Current
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
        self._set_dimensional_parameters()
        self._set_scales()
        self._set_dimensionless_parameters()

        # Set input current
        self._set_input_current()

    def _set_dimensional_parameters(self):
        """Defines the dimensional parameters."""

        # Physical constants
        self.R = pybamm.constants.R
        self.F = pybamm.constants.F
        self.T_ref = self.therm.T_ref

        # Macroscale geometry
        self.L_x = self.geo.L_x
        self.L_y = self.geo.L_y
        self.L_z = self.geo.L_z
        self.A_cc = self.geo.A_cc
        self.A_cooling = self.geo.A_cooling
        self.V_cell = self.geo.V_cell
        self.W = self.L_y
        self.H = self.L_z
        self.A_cs = self.A_cc
        self.delta = self.L_x / self.H

        # Electrical
        self.I_typ = self.elec.I_typ
        self.Q = self.elec.Q
        self.R_contact = self.elec.R_contact
        self.C_rate = self.elec.C_rate
        self.n_electrodes_parallel = self.elec.n_electrodes_parallel
        self.n_cells = self.elec.n_cells
        self.i_typ = self.elec.i_typ
        self.voltage_low_cut_dimensional = self.elec.voltage_low_cut_dimensional
        self.voltage_high_cut_dimensional = self.elec.voltage_high_cut_dimensional

        # Electrolyte properties
        self.c_e_typ = pybamm.Parameter("Typical electrolyte concentration [mol.m-3]")
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
        self.c_ox_init_dim = pybamm.Parameter("Initial oxygen concentration [mol.m-3]")
        self.c_ox_typ = (
            self.c_e_typ
        )  # pybamm.Parameter("Typical oxygen concentration [mol.m-3]")

        # Electrode properties
        self.V_Pb = pybamm.Parameter("Molar volume of lead [m3.mol-1]")
        self.V_PbO2 = pybamm.Parameter("Molar volume of lead-dioxide [m3.mol-1]")
        self.V_PbSO4 = pybamm.Parameter("Molar volume of lead sulfate [m3.mol-1]")
        # Oxygen
        self.s_plus_Ox_dim = pybamm.Parameter(
            "Signed stoichiometry of cations (oxygen reaction)"
        )
        self.s_w_Ox_dim = pybamm.Parameter(
            "Signed stoichiometry of water (oxygen reaction)"
        )
        self.s_ox_Ox_dim = pybamm.Parameter(
            "Signed stoichiometry of oxygen (oxygen reaction)"
        )
        self.ne_Ox = pybamm.Parameter("Electrons in oxygen reaction")
        self.U_Ox_dim = pybamm.Parameter("Oxygen reference OCP vs SHE [V]")
        # Hydrogen
        self.s_plus_Hy_dim = pybamm.Parameter(
            "Signed stoichiometry of cations (hydrogen reaction)"
        )
        self.s_hy_Hy_dim = pybamm.Parameter(
            "Signed stoichiometry of hydrogen (hydrogen reaction)"
        )
        self.ne_Hy = pybamm.Parameter("Electrons in hydrogen reaction")
        self.U_Hy_dim = pybamm.Parameter("Hydrogen reference OCP vs SHE [V]")

        # Electrolyte properties
        self.M_w = pybamm.Parameter("Molar mass of water [kg.mol-1]")
        self.M_plus = pybamm.Parameter("Molar mass of cations [kg.mol-1]")
        self.M_minus = pybamm.Parameter("Molar mass of anions [kg.mol-1]")
        self.M_e = self.M_minus + self.M_plus  # Molar mass of electrolyte [kg.mol-1]

        # Other species properties
        self.D_ox_dimensional = pybamm.Parameter("Oxygen diffusivity [m2.s-1]")
        self.D_hy_dimensional = pybamm.Parameter("Hydrogen diffusivity [m2.s-1]")
        self.V_ox = pybamm.Parameter(
            "Partial molar volume of oxygen molecules [m3.mol-1]"
        )
        self.V_hy = pybamm.Parameter(
            "Partial molar volume of hydrogen molecules [m3.mol-1]"
        )
        self.M_ox = pybamm.Parameter("Molar mass of oxygen molecules [kg.mol-1]")
        self.M_hy = pybamm.Parameter("Molar mass of hydrogen molecules [kg.mol-1]")

        # Thermal
        self.Delta_T = self.therm.Delta_T

        # SEI parameters (for compatibility)
        self.R_sei_dimensional = pybamm.Scalar(0)
        self.beta_sei = pybamm.Scalar(0)

        for domain in self.domain_params.values():
            domain._set_dimensional_parameters()

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
        self.Q_e_max_dimensional = self.Q_e_max * self.c_e_typ * self.F
        self.capacity = (
            self.Q_e_max_dimensional * self.n_electrodes_parallel * self.A_cs * self.L_x
        )

    def t_plus(self, c_e, T):
        """Dimensionless transference number (i.e. c_e is dimensionless)"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e * self.c_e_typ}
        return pybamm.FunctionParameter("Cation transference number", inputs)

    def D_e_dimensional(self, c_e, T):
        """Dimensional diffusivity in electrolyte."""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e}
        return pybamm.FunctionParameter("Electrolyte diffusivity [m2.s-1]", inputs)

    def kappa_e_dimensional(self, c_e, T):
        """Dimensional electrolyte conductivity."""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e}
        return pybamm.FunctionParameter("Electrolyte conductivity [S.m-1]", inputs)

    def chi_dimensional(self, c_e):
        inputs = {"Electrolyte concentration [mol.m-3]": c_e}
        return pybamm.FunctionParameter("Darken thermodynamic factor", inputs)

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

    def rho_dimensional(self, c_e, c_ox=0, c_hy=0):
        """
        Dimensional density of electrolyte [kg.m-3], from thermodynamics.
        c_k in [mol.m-3].
        """
        return (
            self.M_w / self.V_w
            + (self.M_e - self.V_e * self.M_w / self.V_w) * c_e
            + (self.M_ox - self.V_ox * self.M_w / self.V_w) * c_ox
            + (self.M_hy - self.V_hy * self.M_w / self.V_w) * c_hy
        )

    def m_dimensional(self, c_e):
        """
        Dimensional electrolyte molar mass [mol.kg-1], from thermodynamics.
        c_e in [mol.m-3].
        """
        return c_e * self.V_w / ((1 - c_e * self.V_e) * self.M_w)

    def mu_dimensional(self, c_e):
        """
        Dimensional viscosity of electrolyte [kg.m-1.s-1].
        """
        inputs = {"Electrolyte concentration [mol.m-3]": c_e}
        return pybamm.FunctionParameter("Electrolyte viscosity [kg.m-1.s-1]", inputs)

    def _set_scales(self):
        """Define the scales used in the non-dimensionalisation scheme"""
        for domain in self.domain_params.values():
            domain._set_scales()

        # Concentrations
        self.electrolyte_concentration_scale = self.c_e_typ

        # Electrical
        self.potential_scale = self.R * self.T_ref / self.F
        self.current_scale = self.i_typ

        # Reaction velocity scale
        self.velocity_scale = self.i_typ / (self.c_e_typ * self.F)

        # Discharge timescale
        self.tau_discharge = self.F * self.c_e_typ * self.L_x / self.i_typ

        # Electrolyte diffusion timescale
        self.D_e_typ = self.D_e_dimensional(self.c_e_typ, self.T_ref)
        self.tau_diffusion_e = self.L_x**2 / self.D_e_typ

        # Thermal diffusion timescale
        self.tau_th_yz = self.therm.tau_th_yz

        # Choose discharge timescale
        self.timescale = self.tau_discharge

        # Density
        self.rho_typ = self.rho_dimensional(self.c_e_typ)

        # Viscosity
        self.mu_typ = self.mu_dimensional(self.c_e_typ)

    def _set_dimensionless_parameters(self):
        """Defines the dimensionless parameters"""

        # Timescale ratios
        self.C_th = self.tau_th_yz / self.timescale

        # Macroscale Geometry
        self.l_x = self.geo.l_x
        self.l_y = self.geo.l_y
        self.l_z = self.geo.l_z
        self.a_cc = self.geo.a_cc
        self.a_cooling = self.geo.a_cooling
        self.v_cell = self.geo.v_cell
        self.l = self.geo.l
        self.delta = self.geo.delta

        # Diffusive kinematic relationship coefficient
        self.omega_i = (
            self.c_e_typ
            * self.M_e
            / self.rho_typ
            * (self.t_plus(1, self.T_ref) + self.M_minus / self.M_e)
        )
        # Migrative kinematic relationship coefficient (electrolyte)
        self.omega_c_e = (
            self.c_e_typ
            * self.M_e
            / self.rho_typ
            * (1 - self.M_w * self.V_e / self.V_w * self.M_e)
        )
        self.C_e = self.tau_diffusion_e / self.timescale
        # Ratio of viscous pressure scale to osmotic pressure scale (electrolyte)
        self.pi_os_e = (
            self.mu_typ
            * self.velocity_scale
            * self.L_x
            / (self.n.d**2 * self.R * self.T_ref * self.c_e_typ)
        )
        # ratio of electrolyte concentration to electrode concentration, undefined
        self.gamma_e = pybamm.Scalar(1)
        # Reynolds number
        self.Re = self.rho_typ * self.velocity_scale * self.L_x / self.mu_typ

        # Other species properties
        # Oxygen
        self.curlyD_ox = self.D_ox_dimensional / self.D_e_typ
        self.omega_c_ox = (
            self.c_e_typ
            * self.M_ox
            / self.rho_typ
            * (1 - self.M_w * self.V_ox / self.V_w * self.M_ox)
        )
        # Hydrogen
        self.curlyD_hy = self.D_hy_dimensional / self.D_e_typ
        self.omega_c_hy = (
            self.c_e_typ
            * self.M_hy
            / self.rho_typ
            * (1 - self.M_w * self.V_hy / self.V_w * self.M_hy)
        )

        # Electrochemical reactions
        # Oxygen
        self.s_plus_Ox = self.s_plus_Ox_dim / self.ne_Ox
        self.s_w_Ox = self.s_w_Ox_dim / self.ne_Ox
        self.s_ox_Ox = self.s_ox_Ox_dim / self.ne_Ox
        # j0_n_Ox_ref = j0_n_Ox_ref_dimensional / j_scale_n
        # Hydrogen
        self.s_plus_Hy = self.s_plus_Hy_dim / self.ne_Hy
        self.s_hy_Hy = self.s_hy_Hy_dim / self.ne_Hy
        # j0_n_Hy_ref = j0_n_Hy_ref_dimensional / j_scale_n
        # j0_p_Hy_ref = j0_p_Hy_ref_dimensional / j_scale_p

        # Electrolyte properties
        self.beta_Ox = -self.c_e_typ * (
            self.s_plus_Ox * self.V_plus
            + self.s_w_Ox * self.V_w
            + self.s_ox_Ox * self.V_ox
        )
        self.beta_Hy = -self.c_e_typ * (
            self.s_plus_Hy * self.V_plus + self.s_hy_Hy * self.V_hy
        )

        # Electrical
        self.ocv_ref = self.p.U_ref - self.n.U_ref
        self.voltage_low_cut = (
            self.voltage_low_cut_dimensional - self.ocv_ref
        ) / self.potential_scale
        self.voltage_high_cut = (
            self.voltage_high_cut_dimensional - self.ocv_ref
        ) / self.potential_scale

        # Thermal
        self.Theta = self.therm.Theta
        self.rho = self.therm.rho

        self.h_edge = self.therm.h_edge
        self.h_total = self.therm.h_total

        self.B = (
            self.i_typ
            * self.R
            * self.T_ref
            * self.tau_th_yz
            / (self.therm.rho_eff_dim_ref * self.F * self.Delta_T * self.L_x)
        )

        self.T_amb_dim = self.therm.T_amb_dim
        self.T_amb = self.therm.T_amb

        # Initial conditions
        self.T_init = self.therm.T_init
        self.q_init = pybamm.Parameter("Initial State of Charge")
        self.c_e_init = self.q_init
        self.c_ox_init = self.c_ox_init_dim / self.c_ox_typ

        for domain in self.domain_params.values():
            domain._set_dimensionless_parameters()

        self.ocv_init = self.p.prim.U_init - self.n.prim.U_init
        # Concatenations
        self.s_plus_S = pybamm.concatenation(
            pybamm.FullBroadcast(
                self.n.prim.s_plus_S, ["negative electrode"], "current collector"
            ),
            pybamm.FullBroadcast(0, ["separator"], "current collector"),
            pybamm.FullBroadcast(
                self.p.prim.s_plus_S, ["positive electrode"], "current collector"
            ),
        )
        self.beta_surf = pybamm.concatenation(
            pybamm.FullBroadcast(
                self.n.beta_surf, ["negative electrode"], "current collector"
            ),
            pybamm.FullBroadcast(0, ["separator"], "current collector"),
            pybamm.FullBroadcast(
                self.p.beta_surf, ["positive electrode"], "current collector"
            ),
        )
        self.beta = pybamm.concatenation(
            pybamm.FullBroadcast(
                self.n.beta, "negative electrode", "current collector"
            ),
            pybamm.FullBroadcast(0, "separator", "current collector"),
            pybamm.FullBroadcast(
                self.p.beta, "positive electrode", "current collector"
            ),
        )
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

    def D_e(self, c_e, T):
        """Dimensionless electrolyte diffusivity"""
        c_e_dimensional = c_e * self.c_e_typ
        return self.D_e_dimensional(c_e_dimensional, self.T_ref) / self.D_e_typ

    def kappa_e(self, c_e, T):
        """Dimensionless electrolyte conductivity"""
        c_e_dimensional = c_e * self.c_e_typ
        kappa_scale = self.F**2 * self.D_e_typ * self.c_e_typ / (self.R * self.T_ref)
        return self.kappa_e_dimensional(c_e_dimensional, self.T_ref) / kappa_scale

    def chiRT_over_Fc(self, c_e, T):
        """
        chi * (1 + Theta * T) / c,
        as it appears in the electrolyte potential equation
        """
        return self.chi(c_e, T) * (1 + self.Theta * T) / c_e

    def chi(self, c_e, T, c_ox=0, c_hy=0):
        """Thermodynamic factor"""
        return (
            self.chi_dimensional(self.c_e_typ * c_e)
            * (2 * (1 - self.t_plus(c_e, T)))
            / (
                self.V_w
                * self.c_T(self.c_e_typ * c_e, self.c_e_typ * c_ox, self.c_e_typ * c_hy)
            )
        )

    def _set_input_current(self):
        """Set the input current"""

        self.dimensional_current_with_time = pybamm.FunctionParameter(
            "Current function [A]", {"Time [s]": pybamm.t * self.timescale}
        )
        self.dimensional_current_density_with_time = (
            self.dimensional_current_with_time
            / (self.n_electrodes_parallel * self.geo.A_cc)
        )
        self.current_with_time = (
            self.dimensional_current_with_time / self.I_typ * pybamm.sign(self.I_typ)
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

    def _set_dimensional_parameters(self):
        Domain = self.domain.capitalize()
        main = self.main_param

        if self.domain == "separator":
            self.eps_max = pybamm.Parameter("Maximum porosity of separator")
            self.L = self.geo.L
            self.b_e = self.geo.b_e
            self.epsilon_inactive = pybamm.Scalar(0)
            return

        for phase in self.phase_params.values():
            phase._set_dimensional_parameters()

        # Macroscale geometry
        self.L = self.geo.L

        # Microstructure
        self.b_e = self.geo.b_e
        self.b_s = self.geo.b_s
        self.xi = pybamm.Parameter(f"{Domain} electrode morphological parameter")
        # no binder
        self.epsilon_inactive = pybamm.Scalar(0)

        # Electrode properties
        if self.domain == "negative":
            self.DeltaVsurf = (
                main.V_Pb - main.V_PbSO4
            )  # Net Molar Volume consumed in neg electrode [m3.mol-1]
            self.DeltaVliq = (
                main.V_minus - main.V_plus
            )  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]
        elif self.domain == "positive":
            self.DeltaVsurf = (
                main.V_PbSO4 - main.V_PbO2
            )  # Net Molar Volume consumed in pos electrode [m3.mol-1]
            self.DeltaVliq = (
                2 * main.V_w - main.V_minus - 3 * main.V_plus
            )  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]

        self.d = pybamm.Parameter(f"{Domain} electrode pore size [m]")
        self.eps_max = pybamm.Parameter("Maximum porosity of negative electrode")
        self.Q_max_dimensional = pybamm.Parameter(
            f"{Domain} electrode volumetric capacity [C.m-3]"
        )

        self.C_dl_dimensional = pybamm.Parameter(
            f"{Domain} electrode double-layer capacity [F.m-2]"
        )

        # In lead-acid the current collector and electrodes are the same (same
        # conductivity) but we correct here for Bruggeman. Note that because for
        # lithium-ion we allow electrode conductivity to be a function of temperature,
        # but not the current collector conductivity, here the latter is evaluated at
        # T_ref.
        self.sigma_cc_dimensional = (
            self.sigma_dimensional(main.T_ref) * (1 - self.eps_max) ** self.b_s
        )

    def sigma_dimensional(self, T):
        """Dimensional electrical conductivity"""
        inputs = {"Temperature [K]": T}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} electrode conductivity [S.m-1]", inputs
        )

    def _set_scales(self):
        """Define the scales used in the non-dimensionalisation scheme"""
        Domain = self.domain.capitalize()
        if self.domain == "separator":
            return

        for phase in self.phase_params.values():
            phase._set_scales()

        # Reference OCP
        inputs = {"Electrolyte concentration [mol.m-3]": pybamm.Scalar(1)}
        self.U_ref = pybamm.FunctionParameter(
            f"{Domain} electrode open-circuit potential [V]", inputs
        )

    def _set_dimensionless_parameters(self):
        """Defines the dimensionless parameters"""
        main = self.main_param

        if self.domain == "separator":
            self.l = self.geo.l
            self.epsilon_init = self.eps_max
            self.rho = self.therm.rho
            self.lambda_ = self.therm.lambda_
            return

        for phase in self.phase_params.values():
            phase._set_dimensionless_parameters()

        # Macroscale Geometry
        self.l = self.geo.l

        # In lead-acid the current collector and electrodes are the same (same
        # thickness)
        self.l_cc = self.l

        # Tab geometry
        self.l_tab = self.geo.l_tab
        self.centre_y_tab = self.geo.centre_y_tab
        self.centre_z_tab = self.geo.centre_z_tab

        # Electrode Properties
        self.sigma_cc = (
            self.sigma_cc_dimensional * main.potential_scale / main.i_typ / main.L_x
        )
        self.sigma_cc_prime = self.sigma_cc * main.delta**2
        self.Q_max = self.Q_max_dimensional / (main.c_e_typ * main.F)
        self.beta_U = 1 / self.Q_max

        # Electrolyte properties
        self.beta_surf = (
            -main.c_e_typ * self.DeltaVsurf / self.prim.ne_S
        )  # Molar volume change (lead)
        self.beta_liq = (
            -main.c_e_typ * self.DeltaVliq / self.prim.ne_S
        )  # Molar volume change (electrolyte, neg)
        self.beta = (self.beta_surf + self.beta_liq) * pybamm.Parameter(
            "Volume change factor"
        )

        self.C_dl = (
            self.C_dl_dimensional
            * main.potential_scale
            / self.prim.j_scale
            / main.timescale
        )

        # Thermal
        self.rho_cc = self.therm.rho_cc
        self.rho = self.therm.rho

        self.lambda_cc = self.therm.lambda_cc
        self.lambda_ = self.therm.lambda_

        self.h_tab = self.therm.h_tab
        self.h_cc = self.therm.h_cc

        # Initial conditions
        self.c_init = main.c_e_init
        sgn = -1 if self.domain == "negative" else 1
        self.epsilon_init = (
            self.eps_max
            + sgn * self.beta_surf * main.Q_e_max / self.l * (1 - main.q_init)
        )
        self.curlyU_init = main.Q_e_max * (1.2 - main.q_init) / (self.Q_max * self.l)

    def sigma(self, T):
        """Dimensionless negative electrode electrical conductivity"""
        T_dim = self.main_param.Delta_T * T + self.main_param.T_ref
        return (
            self.sigma_dimensional(T_dim)
            * self.main_param.potential_scale
            / self.main_param.current_scale
            / self.main_param.L_x
        )

    def sigma_prime(self, T):
        """Rescaled dimensionless negative electrode electrical conductivity"""
        return self.sigma(T) * self.main_param.delta**2


class PhaseLeadAcidParameters(BaseParameters):
    def __init__(self, phase, domain_param):
        self.phase = phase

        self.domain_param = domain_param
        self.domain = domain_param.domain
        self.main_param = domain_param.main_param
        self.geo = domain_param.geo.prim

    def _set_dimensional_parameters(self):
        domain, Domain = self.domain_Domain  # Microstructure
        x = (
            pybamm.SpatialVariable(
                f"x_{domain[0]}",
                domain=[f"{domain} electrode"],
                auxiliary_domains={"secondary": "current collector"},
                coord_sys="cartesian",
            )
            * self.main_param.L_x
        )
        self.a_dimensional = pybamm.FunctionParameter(
            f"{Domain} electrode surface area to volume ratio [m-1]",
            {"Through-cell distance (x) [m]": x},
        )

        # Electrochemical reactions
        # Main
        self.s_plus_S_dim = pybamm.Parameter(
            f"{Domain} electrode cation signed stoichiometry"
        )
        self.ne_S = pybamm.Parameter(f"{Domain} electrode electrons in reaction")
        self.s_plus_S = self.s_plus_S_dim / self.ne_S
        self.alpha_bv = pybamm.Parameter(
            f"{Domain} electrode Butler-Volmer transfer coefficient"
        )

    def U_dimensional(self, c_e, T):
        """Dimensional open-circuit voltage [V]"""
        inputs = {
            "Electrolyte molar mass [mol.kg-1]": self.main_param.m_dimensional(c_e)
        }
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} electrode open-circuit potential [V]", inputs
        )

    def j0_dimensional(self, c_e, T):
        """Dimensional exchange-current density [A.m-2]"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} electrode exchange-current density [A.m-2]", inputs
        )

    def j0_Ox_dimensional(self, c_e, T):
        """Dimensional oxygen electrode exchange-current density [A.m-2]"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} electrode oxygen exchange-current density [A.m-2]", inputs
        )

    def _set_scales(self):
        """Define the scales used in the non-dimensionalisation scheme"""
        # Microscale (typical values at electrode/current collector interface)
        self.a_typ = pybamm.xyz_average(self.a_dimensional)

        # Electrical
        self.j_scale = self.main_param.i_typ / (self.a_typ * self.main_param.L_x)

    def _set_dimensionless_parameters(self):
        """Defines the dimensionless parameters"""
        main = self.main_param

        # Microstructure
        self.a = self.a_dimensional / self.a_typ
        self.delta_pore = 1 / (self.a_typ * main.L_x)
        self.epsilon_s = 1 - self.domain_param.eps_max

        # Electrochemical reactions
        # Main
        self.ne = self.ne_S

        # Initial conditions
        self.c_init = main.c_e_init
        self.U_init = self.U(main.c_e_init, main.T_init)

        # Electrochemical reactions
        # Oxygen
        self.U_Ox = (main.U_Ox_dim - self.domain_param.U_ref) / main.potential_scale
        self.U_Hy = (main.U_Hy_dim - self.domain_param.U_ref) / main.potential_scale

    def U(self, c_e, T):
        """Dimensionless open-circuit voltage in the negative electrode"""
        c_e_dimensional = c_e * self.main_param.c_e_typ
        T_dim = self.main_param.Delta_T * T + self.main_param.T_ref
        return (
            self.U_dimensional(c_e_dimensional, T_dim) - self.domain_param.U_ref
        ) / self.main_param.potential_scale

    def j0(self, c_e, T):
        """Dimensionless exchange-current density in the negative electrode"""
        c_e_dim = c_e * self.main_param.c_e_typ
        T_dim = self.main_param.Delta_T * T + self.main_param.T_ref
        return self.j0_dimensional(c_e_dim, T_dim) / self.j_scale

    def j0_Ox(self, c_e, T):
        """Dimensionless oxygen exchange-current density in the positive electrode"""
        c_e_dim = c_e * self.main_param.c_e_typ
        T_dim = self.main_param.Delta_T * T + self.main_param.T_ref
        return self.j0_Ox_dimensional(c_e_dim, T_dim) / self.j_scale
