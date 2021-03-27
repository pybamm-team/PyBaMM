#
# Standard parameters for lead-acid battery models
#

import pybamm
import numpy as np


class LeadAcidParameters:
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
        self.L_n = self.geo.L_n
        self.L_s = self.geo.L_s
        self.L_p = self.geo.L_p
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

        # Microstructure
        # Note: the surface area to volume ratio can be set as a function of
        # through-cell position, so is defined later as a function
        self.b_e_n = self.geo.b_e_n
        self.b_e_s = self.geo.b_e_s
        self.b_e_p = self.geo.b_e_p
        self.b_s_n = self.geo.b_s_n
        self.b_s_p = self.geo.b_s_p
        self.xi_n = pybamm.Parameter("Negative electrode morphological parameter")
        self.xi_p = pybamm.Parameter("Positive electrode morphological parameter")
        # no binder
        self.epsilon_inactive_n = pybamm.Scalar(0)
        self.epsilon_inactive_s = pybamm.Scalar(0)
        self.epsilon_inactive_p = pybamm.Scalar(0)

        # Electrode properties
        self.V_Pb = pybamm.Parameter("Molar volume of lead [m3.mol-1]")
        self.V_PbO2 = pybamm.Parameter("Molar volume of lead-dioxide [m3.mol-1]")
        self.V_PbSO4 = pybamm.Parameter("Molar volume of lead sulfate [m3.mol-1]")
        self.DeltaVsurf_n = (
            self.V_Pb - self.V_PbSO4
        )  # Net Molar Volume consumed in neg electrode [m3.mol-1]
        self.DeltaVsurf_p = (
            self.V_PbSO4 - self.V_PbO2
        )  # Net Molar Volume consumed in pos electrode [m3.mol-1]
        self.d_n = pybamm.Parameter("Negative electrode pore size [m]")
        self.d_p = pybamm.Parameter("Positive electrode pore size [m]")
        self.eps_n_max = pybamm.Parameter("Maximum porosity of negative electrode")
        self.eps_s_max = pybamm.Parameter("Maximum porosity of separator")
        self.eps_p_max = pybamm.Parameter("Maximum porosity of positive electrode")
        self.Q_n_max_dimensional = pybamm.Parameter(
            "Negative electrode volumetric capacity [C.m-3]"
        )
        self.Q_p_max_dimensional = pybamm.Parameter(
            "Positive electrode volumetric capacity [C.m-3]"
        )
        self.sigma_n_dim = pybamm.Parameter("Negative electrode conductivity [S.m-1]")
        self.sigma_p_dim = pybamm.Parameter("Positive electrode conductivity [S.m-1]")
        # In lead-acid the current collector and electrodes are the same (same
        # conductivity) but we correct here for Bruggeman
        self.sigma_cn_dimensional = (
            self.sigma_n_dim * (1 - self.eps_n_max) ** self.b_s_n
        )
        self.sigma_cp_dimensional = (
            self.sigma_p_dim * (1 - self.eps_p_max) ** self.b_s_p
        )

        # Electrochemical reactions
        # Main
        self.s_plus_n_S_dim = pybamm.Parameter(
            "Negative electrode cation signed stoichiometry"
        )
        self.s_plus_p_S_dim = pybamm.Parameter(
            "Positive electrode cation signed stoichiometry"
        )
        self.ne_n_S = pybamm.Parameter("Negative electrode electrons in reaction")
        self.ne_p_S = pybamm.Parameter("Positive electrode electrons in reaction")
        self.C_dl_n_dimensional = pybamm.Parameter(
            "Negative electrode double-layer capacity [F.m-2]"
        )
        self.C_dl_p_dimensional = pybamm.Parameter(
            "Positive electrode double-layer capacity [F.m-2]"
        )
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

        self.DeltaVliq_n = (
            self.V_minus - self.V_plus
        )  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]
        self.DeltaVliq_p = (
            2 * self.V_w - self.V_minus - 3 * self.V_plus
        )  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]

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

        # Electrode properties
        self.V_Pb = pybamm.Parameter("Molar volume of lead [m3.mol-1]")
        self.V_PbO2 = pybamm.Parameter("Molar volume of lead-dioxide [m3.mol-1]")
        self.V_PbSO4 = pybamm.Parameter("Molar volume of lead sulfate [m3.mol-1]")
        self.DeltaVsurf_n = (
            self.V_Pb - self.V_PbSO4
        )  # Net Molar Volume consumed in neg electrode [m3.mol-1]
        self.DeltaVsurf_p = (
            self.V_PbSO4 - self.V_PbO2
        )  # Net Molar Volume consumed in pos electrode [m3.mol-1]
        self.d_n = pybamm.Parameter("Negative electrode pore size [m]")
        self.d_p = pybamm.Parameter("Positive electrode pore size [m]")
        self.eps_n_max = pybamm.Parameter("Maximum porosity of negative electrode")
        self.eps_s_max = pybamm.Parameter("Maximum porosity of separator")
        self.eps_p_max = pybamm.Parameter("Maximum porosity of positive electrode")
        self.Q_n_max_dimensional = pybamm.Parameter(
            "Negative electrode volumetric capacity [C.m-3]"
        )
        self.Q_p_max_dimensional = pybamm.Parameter(
            "Positive electrode volumetric capacity [C.m-3]"
        )

        # Thermal
        self.Delta_T = self.therm.Delta_T

        # SEI parameters (for compatibility)
        self.R_sei_dimensional = pybamm.Scalar(0)
        self.beta_sei_n = pybamm.Scalar(0)

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

    def U_n_dimensional(self, c_e, T):
        """Dimensional open-circuit voltage in the negative electrode [V]"""
        inputs = {"Electrolyte molar mass [mol.kg-1]": self.m_dimensional(c_e)}
        return pybamm.FunctionParameter(
            "Negative electrode open-circuit potential [V]", inputs
        )

    def U_p_dimensional(self, c_e, T):
        """Dimensional open-circuit voltage in the positive electrode [V]"""
        inputs = {"Electrolyte molar mass [mol.kg-1]": self.m_dimensional(c_e)}
        return pybamm.FunctionParameter(
            "Positive electrode open-circuit potential [V]", inputs
        )

    def j0_n_dimensional(self, c_e, T):
        """Dimensional negative electrode exchange-current density [A.m-2]"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        return pybamm.FunctionParameter(
            "Negative electrode exchange-current density [A.m-2]", inputs
        )

    def j0_p_dimensional(self, c_e, T):
        """Dimensional positive electrode exchange-current density [A.m-2]"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        return pybamm.FunctionParameter(
            "Positive electrode exchange-current density [A.m-2]", inputs
        )

    def j0_p_Ox_dimensional(self, c_e, T):
        """Dimensional oxygen positive electrode exchange-current density [A.m-2]"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        return pybamm.FunctionParameter(
            "Positive electrode oxygen exchange-current density [A.m-2]", inputs
        )

    def a_n_dimensional(self, x):
        """
        Negative electrode surface area to volume ratio as a function of
        through-cell distance
        """
        inputs = {"Through-cell distance (x_n) [m]": x}
        return pybamm.FunctionParameter(
            "Negative electrode surface area to volume ratio [m-1]", inputs
        )

    def a_p_dimensional(self, x):
        """
        Positive electrode surface area to volume ratio as a function of
        through-cell distance
        """
        inputs = {"Through-cell distance (x_p) [m]": x}
        return pybamm.FunctionParameter(
            "Positive electrode surface area to volume ratio [m-1]", inputs
        )

    def epsilon_s_n(self, x):
        """
        Negative electrode active material volume fraction, specified for compatibility
        with lithium-ion submodels. Note that this does not change even though porosity
        changes, since the material being created is inactive.
        """
        return pybamm.FullBroadcast(
            1 - self.eps_n_max, "negative electrode", "current collector"
        )

    def epsilon_s_p(self, x):
        """
        Positive electrode active material volume fraction, specified for compatibility
        with lithium-ion submodels. Note that this does not change even though porosity
        changes, since the material being created is inactive.
        """
        return pybamm.FullBroadcast(
            1 - self.eps_p_max, "positive electrode", "current collector"
        )

    def _set_scales(self):
        """Define the scales used in the non-dimensionalisation scheme"""

        # Microscale (typical values at electrode/current collector interface)
        self.a_n_typ = self.a_n_dimensional(0)
        self.a_p_typ = self.a_p_dimensional(self.L_x)

        # Concentrations
        self.electrolyte_concentration_scale = self.c_e_typ

        # Electrical
        self.potential_scale = self.R * self.T_ref / self.F
        self.current_scale = self.i_typ
        self.j_scale_n = self.i_typ / (self.a_n_typ * self.L_x)
        self.j_scale_p = self.i_typ / (self.a_p_typ * self.L_x)

        # Reaction velocity scale
        self.velocity_scale = self.i_typ / (self.c_e_typ * self.F)

        # Discharge timescale
        self.tau_discharge = self.F * self.c_e_typ * self.L_x / self.i_typ

        # Electrolyte diffusion timescale
        self.D_e_typ = self.D_e_dimensional(self.c_e_typ, self.T_ref)
        self.tau_diffusion_e = self.L_x ** 2 / self.D_e_typ

        # Thermal diffusion timescale
        self.tau_th_yz = self.therm.tau_th_yz

        # Choose discharge timescale
        self.timescale = self.tau_discharge

        # Density
        self.rho_typ = self.rho_dimensional(self.c_e_typ)

        # Viscosity
        self.mu_typ = self.mu_dimensional(self.c_e_typ)

        # Reference OCP
        inputs = {"Electrolyte concentration [mol.m-3]": pybamm.Scalar(1)}
        self.U_n_ref = pybamm.FunctionParameter(
            "Negative electrode open-circuit potential [V]", inputs
        )
        inputs = {"Electrolyte concentration [mol.m-3]": pybamm.Scalar(1)}
        self.U_p_ref = pybamm.FunctionParameter(
            "Positive electrode open-circuit potential [V]", inputs
        )

    def _set_dimensionless_parameters(self):
        """Defines the dimensionless parameters"""

        # Timescale ratios
        self.C_th = self.tau_th_yz / self.tau_discharge

        # Macroscale Geometry
        self.l_n = self.geo.l_n
        self.l_s = self.geo.l_s
        self.l_p = self.geo.l_p
        self.l_x = self.geo.l_x
        self.l_y = self.geo.l_y
        self.l_z = self.geo.l_z
        self.a_cc = self.geo.a_cc
        self.a_cooling = self.geo.a_cooling
        self.v_cell = self.geo.v_cell
        self.l = self.geo.l
        self.delta = self.geo.delta
        # In lead-acid the current collector and electrodes are the same (same
        # thickness)
        self.l_cn = self.l_n
        self.l_cp = self.l_p

        # Tab geometry
        self.l_tab_n = self.geo.l_tab_n
        self.centre_y_tab_n = self.geo.centre_y_tab_n
        self.centre_z_tab_n = self.geo.centre_z_tab_n
        self.l_tab_p = self.geo.l_tab_p
        self.centre_y_tab_p = self.geo.centre_y_tab_p
        self.centre_z_tab_p = self.geo.centre_z_tab_p

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
        self.C_e = self.tau_diffusion_e / self.tau_discharge
        # Ratio of viscous pressure scale to osmotic pressure scale (electrolyte)
        self.pi_os_e = (
            self.mu_typ
            * self.velocity_scale
            * self.L_x
            / (self.d_n ** 2 * self.R * self.T_ref * self.c_e_typ)
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

        # Electrode Properties
        self.sigma_cn = (
            self.sigma_cn_dimensional * self.potential_scale / self.i_typ / self.L_x
        )
        self.sigma_n = (
            self.sigma_n_dim * self.potential_scale / self.current_scale / self.L_x
        )
        self.sigma_p = (
            self.sigma_p_dim * self.potential_scale / self.current_scale / self.L_x
        )
        self.sigma_cp = (
            self.sigma_cp_dimensional * self.potential_scale / self.i_typ / self.L_x
        )
        self.sigma_n_prime = self.sigma_n * self.delta ** 2
        self.sigma_p_prime = self.sigma_p * self.delta ** 2
        self.sigma_cn_prime = self.sigma_cn * self.delta ** 2
        self.sigma_cp_prime = self.sigma_cp * self.delta ** 2
        self.delta_pore_n = 1 / (self.a_n_typ * self.L_x)
        self.delta_pore_p = 1 / (self.a_p_typ * self.L_x)
        self.Q_n_max = self.Q_n_max_dimensional / (self.c_e_typ * self.F)
        self.Q_p_max = self.Q_p_max_dimensional / (self.c_e_typ * self.F)
        self.beta_U_n = 1 / self.Q_n_max
        self.beta_U_p = -1 / self.Q_p_max

        # Electrochemical reactions
        # Main
        self.s_plus_n_S = self.s_plus_n_S_dim / self.ne_n_S
        self.s_plus_p_S = self.s_plus_p_S_dim / self.ne_p_S
        self.s_plus_S = pybamm.Concatenation(
            pybamm.FullBroadcast(
                self.s_plus_n_S, ["negative electrode"], "current collector"
            ),
            pybamm.FullBroadcast(0, ["separator"], "current collector"),
            pybamm.FullBroadcast(
                self.s_plus_p_S, ["positive electrode"], "current collector"
            ),
        )
        self.C_dl_n = (
            self.C_dl_n_dimensional
            * self.potential_scale
            / self.j_scale_n
            / self.tau_discharge
        )
        self.C_dl_p = (
            self.C_dl_p_dimensional
            * self.potential_scale
            / self.j_scale_p
            / self.tau_discharge
        )
        self.ne_n = self.ne_n_S
        self.ne_p = self.ne_p_S
        # Oxygen
        self.s_plus_Ox = self.s_plus_Ox_dim / self.ne_Ox
        self.s_w_Ox = self.s_w_Ox_dim / self.ne_Ox
        self.s_ox_Ox = self.s_ox_Ox_dim / self.ne_Ox
        # j0_n_Ox_ref = j0_n_Ox_ref_dimensional / j_scale_n
        self.U_n_Ox = (self.U_Ox_dim - self.U_n_ref) / self.potential_scale
        self.U_p_Ox = (self.U_Ox_dim - self.U_p_ref) / self.potential_scale
        # Hydrogen
        self.s_plus_Hy = self.s_plus_Hy_dim / self.ne_Hy
        self.s_hy_Hy = self.s_hy_Hy_dim / self.ne_Hy
        # j0_n_Hy_ref = j0_n_Hy_ref_dimensional / j_scale_n
        # j0_p_Hy_ref = j0_p_Hy_ref_dimensional / j_scale_p
        self.U_n_Hy = (self.U_Hy_dim - self.U_n_ref) / self.potential_scale
        self.U_p_Hy = (self.U_Hy_dim - self.U_p_ref) / self.potential_scale

        # Electrolyte properties
        self.beta_surf_n = (
            -self.c_e_typ * self.DeltaVsurf_n / self.ne_n_S
        )  # Molar volume change (lead)
        self.beta_surf_p = (
            -self.c_e_typ * self.DeltaVsurf_p / self.ne_p_S
        )  # Molar volume change (lead dioxide)
        self.beta_surf = pybamm.Concatenation(
            pybamm.FullBroadcast(
                self.beta_surf_n, ["negative electrode"], "current collector"
            ),
            pybamm.FullBroadcast(0, ["separator"], "current collector"),
            pybamm.FullBroadcast(
                self.beta_surf_p, ["positive electrode"], "current collector"
            ),
        )
        self.beta_liq_n = (
            -self.c_e_typ * self.DeltaVliq_n / self.ne_n_S
        )  # Molar volume change (electrolyte, neg)
        self.beta_liq_p = (
            -self.c_e_typ * self.DeltaVliq_p / self.ne_p_S
        )  # Molar volume change (electrolyte, pos)
        self.beta_n = (self.beta_surf_n + self.beta_liq_n) * pybamm.Parameter(
            "Volume change factor"
        )
        self.beta_p = (self.beta_surf_p + self.beta_liq_p) * pybamm.Parameter(
            "Volume change factor"
        )
        self.beta = pybamm.Concatenation(
            pybamm.FullBroadcast(
                self.beta_n, "negative electrode", "current collector"
            ),
            pybamm.FullBroadcast(0, "separator", "current collector"),
            pybamm.FullBroadcast(
                self.beta_p, "positive electrode", "current collector"
            ),
        )
        self.beta_Ox = -self.c_e_typ * (
            self.s_plus_Ox * self.V_plus
            + self.s_w_Ox * self.V_w
            + self.s_ox_Ox * self.V_ox
        )
        self.beta_Hy = -self.c_e_typ * (
            self.s_plus_Hy * self.V_plus + self.s_hy_Hy * self.V_hy
        )

        # Electrical
        self.voltage_low_cut = (
            self.voltage_low_cut_dimensional - (self.U_p_ref - self.U_n_ref)
        ) / self.potential_scale
        self.voltage_high_cut = (
            self.voltage_high_cut_dimensional - (self.U_p_ref - self.U_n_ref)
        ) / self.potential_scale

        # Electrolyte volumetric capacity
        self.Q_e_max = (
            self.l_n * self.eps_n_max
            + self.l_s * self.eps_s_max
            + self.l_p * self.eps_p_max
        ) / (self.s_plus_p_S - self.s_plus_n_S)
        self.Q_e_max_dimensional = self.Q_e_max * self.c_e_typ * self.F
        self.capacity = (
            self.Q_e_max_dimensional * self.n_electrodes_parallel * self.A_cs * self.L_x
        )

        # Thermal
        self.rho_cn = self.therm.rho_cn
        self.rho_n = self.therm.rho_n
        self.rho_s = self.therm.rho_s
        self.rho_p = self.therm.rho_p
        self.rho_cp = self.therm.rho_cp

        self.lambda_cn = self.therm.lambda_cn
        self.lambda_n = self.therm.lambda_n
        self.lambda_s = self.therm.lambda_s
        self.lambda_p = self.therm.lambda_p
        self.lambda_cp = self.therm.lambda_cp

        self.Theta = self.therm.Theta

        self.h_edge = self.therm.h_edge
        self.h_tab_n = self.therm.h_tab_n
        self.h_tab_p = self.therm.h_tab_p
        self.h_cn = self.therm.h_cn
        self.h_cp = self.therm.h_cp
        self.h_total = self.therm.h_total

        self.B = (
            self.i_typ
            * self.R
            * self.T_ref
            * self.tau_th_yz
            / (self.therm.rho_eff_dim(self.T_ref) * self.F * self.Delta_T * self.L_x)
        )

        self.T_amb_dim = self.therm.T_amb_dim
        self.T_amb = self.therm.T_amb

        # Initial conditions
        self.T_init = self.therm.T_init
        self.q_init = pybamm.Parameter("Initial State of Charge")
        self.c_e_init = self.q_init
        self.c_ox_init = self.c_ox_init_dim / self.c_ox_typ
        self.epsilon_n_init = (
            self.eps_n_max
            - self.beta_surf_n * self.Q_e_max / self.l_n * (1 - self.q_init)
        )
        self.epsilon_s_init = self.eps_s_max
        self.epsilon_p_init = (
            self.eps_p_max
            + self.beta_surf_p * self.Q_e_max / self.l_p * (1 - self.q_init)
        )
        self.epsilon_init = pybamm.Concatenation(
            pybamm.FullBroadcast(
                self.epsilon_n_init, ["negative electrode"], "current collector"
            ),
            pybamm.FullBroadcast(
                self.epsilon_s_init, ["separator"], "current collector"
            ),
            pybamm.FullBroadcast(
                self.epsilon_p_init, ["positive electrode"], "current collector"
            ),
        )
        self.curlyU_n_init = (
            self.Q_e_max * (1.2 - self.q_init) / (self.Q_n_max * self.l_n)
        )
        self.curlyU_p_init = (
            self.Q_e_max * (1.2 - self.q_init) / (self.Q_p_max * self.l_p)
        )

    def D_e(self, c_e, T):
        """Dimensionless electrolyte diffusivity"""
        c_e_dimensional = c_e * self.c_e_typ
        return self.D_e_dimensional(c_e_dimensional, self.T_ref) / self.D_e_typ

    def kappa_e(self, c_e, T):
        """Dimensionless electrolyte conductivity"""
        c_e_dimensional = c_e * self.c_e_typ
        kappa_scale = self.F ** 2 * self.D_e_typ * self.c_e_typ / (self.R * self.T_ref)
        return self.kappa_e_dimensional(c_e_dimensional, self.T_ref) / kappa_scale

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

    def U_n(self, c_e_n, T):
        """Dimensionless open-circuit voltage in the negative electrode"""
        c_e_n_dimensional = c_e_n * self.c_e_typ
        T_dim = self.Delta_T * T + self.T_ref
        return (
            self.U_n_dimensional(c_e_n_dimensional, T_dim) - self.U_n_ref
        ) / self.potential_scale

    def U_p(self, c_e_p, T):
        """Dimensionless open-circuit voltage in the positive electrode"""
        c_e_p_dimensional = c_e_p * self.c_e_typ
        T_dim = self.Delta_T * T + self.T_ref
        return (
            self.U_p_dimensional(c_e_p_dimensional, T_dim) - self.U_p_ref
        ) / self.potential_scale

    def j0_n(self, c_e, T):
        """Dimensionless exchange-current density in the negative electrode"""
        c_e_dim = c_e * self.c_e_typ
        T_dim = self.Delta_T * T + self.T_ref
        return self.j0_n_dimensional(c_e_dim, T_dim) / self.j_scale_n

    def j0_p(self, c_e, T):
        """Dimensionless exchange-current density in the positive electrode"""
        c_e_dim = c_e * self.c_e_typ
        T_dim = self.Delta_T * T + self.T_ref
        return self.j0_p_dimensional(c_e_dim, T_dim) / self.j_scale_p

    def j0_p_Ox(self, c_e, T):
        """Dimensionless oxygen exchange-current density in the positive electrode"""
        c_e_dim = c_e * self.c_e_typ
        T_dim = self.Delta_T * T + self.T_ref
        return self.j0_p_Ox_dimensional(c_e_dim, T_dim) / self.j_scale_p

    def c_n_init(self, x):
        """
        Dimensionless initial concentration (as a function of dimensionless position x
        to be consistent with lithium-ion)
        """
        return self.c_e_init

    def c_p_init(self, x):
        """
        Dimensionless initial concentration (as a function of dimensionless position x
        to be consistent with lithium-ion)
        """
        return self.c_e_init

    def a_n(self, x):
        """
        Dimensionless negative electrode surface area to volume ratio as a
        function of dimensionless position x
        """
        x_dim = x * self.L_x
        return self.a_n_dimensional(x_dim) / self.a_n_typ

    def a_p(self, x):
        """
        Dimensionless positive electrode surface area to volume ratio as a
        function of dimensionless position x
        """
        x_dim = x * self.L_x
        return self.a_p_dimensional(x_dim) / self.a_p_typ

    def rho(self, T):
        """Dimensionless effective volumetric heat capacity"""
        return (
            self.rho_cn(T) * self.l_cn
            + self.rho_n(T) * self.l_n
            + self.rho_s(T) * self.l_s
            + self.rho_p(T) * self.l_p
            + self.rho_cp(T) * self.l_cp
        ) / self.l

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
            self.dimensional_current_with_time
            / self.I_typ
            * pybamm.Function(np.sign, self.I_typ)
        )
