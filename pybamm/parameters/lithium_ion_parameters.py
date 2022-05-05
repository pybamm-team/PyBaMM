#
# Standard parameters for lithium-ion battery models
#
import pybamm
from .base_parameters import BaseParameters


class LithiumIonParameters(BaseParameters):
    """
    Standard parameters for lithium-ion battery models

    Layout:
        1. Dimensional Parameters
        2. Dimensional Functions
        3. Scalings
        4. Dimensionless Parameters
        5. Dimensionless Functions
        6. Input Current

    Parameters
    ----------

    options : dict, optional
        A dictionary of options to be passed to the parameters. The options that
        can be set are listed below.

            * "particle shape" : str, optional
                Sets the model shape of the electrode particles. This is used to
                calculate the surface area to volume ratio. Can be "spherical"
                (default). TODO: implement "cylindrical" and "platelet".
            * "working electrode": str
                Which electrode(s) intercalates and which is counter. If "both"
                (default), the model is a standard battery. Otherwise can be "negative"
                or "positive" to indicate a half-cell model.

    """

    def __init__(self, options=None):
        self.options = options
        # Save whether the submodel is a half-cell submodel
        self.half_cell = self.options["working electrode"] != "both"

        # Get geometric, electrical and thermal parameters
        self.geo = pybamm.geometric_parameters
        self.elec = pybamm.electrical_parameters
        self.therm = pybamm.thermal_parameters

        # Spatial variables
        r_n = pybamm.standard_spatial_vars.r_n * self.geo.R_n_typ
        r_p = pybamm.standard_spatial_vars.r_p * self.geo.R_p_typ
        x_n = pybamm.standard_spatial_vars.x_n * self.geo.L_x
        x_s = pybamm.standard_spatial_vars.x_s * self.geo.L_x
        x_p = pybamm.standard_spatial_vars.x_p * self.geo.L_x

        # Initialize domain parameters
        self.n = DomainParameters("Negative", self, x_n, r_n)
        self.s = DomainParameters("Separator", self, x_s, None)
        self.p = DomainParameters("Positive", self, x_p, r_p)

        # Set parameters and scales
        self._set_dimensional_parameters()
        self._set_scales()
        self._set_dimensionless_parameters()

        # Set input current
        self._set_input_current()

    def _set_dimensional_parameters(self):
        """Defines the dimensional parameters"""
        self.n._set_dimensional_parameters()
        self.s._set_dimensional_parameters()
        self.p._set_dimensional_parameters()

        # Spatial variables
        r_n = pybamm.standard_spatial_vars.r_n * self.geo.R_n_typ
        r_p = pybamm.standard_spatial_vars.r_p * self.geo.R_p_typ
        x_n = pybamm.standard_spatial_vars.x_n * self.geo.L_x
        x_p = pybamm.standard_spatial_vars.x_p * self.geo.L_x

        # Physical constants
        self.R = pybamm.constants.R
        self.F = pybamm.constants.F
        self.k_b = pybamm.constants.k_b
        self.q_e = pybamm.constants.q_e

        self.T_ref = self.therm.T_ref

        # Macroscale geometry
        self.L_cn = self.geo.L_cn
        self.L_n = self.geo.L_n
        self.L_s = self.geo.L_s
        self.L_p = self.geo.L_p
        self.L_cp = self.geo.L_cp
        self.L_x = self.geo.L_x
        self.L = self.geo.L
        self.L_y = self.geo.L_y
        self.L_z = self.geo.L_z
        self.r_inner_dimensional = self.geo.r_inner_dimensional
        self.r_outer_dimensional = self.geo.r_outer_dimensional
        self.A_cc = self.geo.A_cc
        self.A_cooling = self.geo.A_cooling
        self.V_cell = self.geo.V_cell
        # Note: the surface area to volume ratio is defined later with the function
        # parameters. The particle size as a function of through-cell position is
        # already defined in geometric_parameters.py
        self.R_n_dimensional = self.geo.R_n_dimensional
        self.R_p_dimensional = self.geo.R_p_dimensional

        # Tab geometry (for pouch cells)
        self.L_tab_n = self.geo.L_tab_n
        self.Centre_y_tab_n = self.geo.Centre_y_tab_n
        self.Centre_z_tab_n = self.geo.Centre_z_tab_n
        self.L_tab_p = self.geo.L_tab_p
        self.Centre_y_tab_p = self.geo.Centre_y_tab_p
        self.Centre_z_tab_p = self.geo.Centre_z_tab_p
        self.A_tab_n = self.geo.A_tab_n
        self.A_tab_p = self.geo.A_tab_p

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

        if self.half_cell:
            self.epsilon_init = pybamm.concatenation(
                self.s.epsilon_init, self.p.epsilon_init
            )
        else:
            self.epsilon_init = pybamm.concatenation(
                self.n.epsilon_init, self.s.epsilon_init, self.p.epsilon_init
            )

        self.b_e_n = self.geo.b_e_n
        self.b_e_s = self.geo.b_e_s
        self.b_e_p = self.geo.b_e_p
        self.b_s_n = self.geo.b_s_n
        self.b_s_p = self.geo.b_s_p

        # Particle-size distribution parameters
        self.R_min_n_dim = self.geo.R_min_n_dim
        self.R_min_p_dim = self.geo.R_min_p_dim
        self.R_max_n_dim = self.geo.R_max_n_dim
        self.R_max_p_dim = self.geo.R_max_p_dim
        self.sd_a_n_dim = self.geo.sd_a_n_dim
        self.sd_a_p_dim = self.geo.sd_a_p_dim
        self.f_a_dist_n_dimensional = self.geo.f_a_dist_n_dimensional
        self.f_a_dist_p_dimensional = self.geo.f_a_dist_p_dimensional

        # SEI parameters
        self.V_bar_inner_dimensional = pybamm.Parameter(
            "Inner SEI partial molar volume [m3.mol-1]"
        )
        self.V_bar_outer_dimensional = pybamm.Parameter(
            "Outer SEI partial molar volume [m3.mol-1]"
        )

        self.m_sei_dimensional = pybamm.Parameter(
            "SEI reaction exchange current density [A.m-2]"
        )

        self.R_sei_dimensional = pybamm.Parameter("SEI resistivity [Ohm.m]")
        self.D_sol_dimensional = pybamm.Parameter(
            "Outer SEI solvent diffusivity [m2.s-1]"
        )
        self.c_sol_dimensional = pybamm.Parameter(
            "Bulk solvent concentration [mol.m-3]"
        )
        self.m_ratio = pybamm.Parameter(
            "Ratio of inner and outer SEI exchange current densities"
        )
        self.U_inner_dimensional = pybamm.Parameter(
            "Inner SEI open-circuit potential [V]"
        )
        self.U_outer_dimensional = pybamm.Parameter(
            "Outer SEI open-circuit potential [V]"
        )
        self.kappa_inner_dimensional = pybamm.Parameter(
            "Inner SEI electron conductivity [S.m-1]"
        )
        self.D_li_dimensional = pybamm.Parameter(
            "Inner SEI lithium interstitial diffusivity [m2.s-1]"
        )
        self.c_li_0_dimensional = pybamm.Parameter(
            "Lithium interstitial reference concentration [mol.m-3]"
        )
        self.L_inner_0_dim = pybamm.Parameter("Initial inner SEI thickness [m]")
        self.L_outer_0_dim = pybamm.Parameter("Initial outer SEI thickness [m]")
        self.L_sei_0_dim = self.L_inner_0_dim + self.L_outer_0_dim

        # EC reaction
        self.c_ec_0_dim = pybamm.Parameter(
            "EC initial concentration in electrolyte [mol.m-3]"
        )
        self.D_ec_dim = pybamm.Parameter("EC diffusivity [m2.s-1]")
        self.k_sei_dim = pybamm.Parameter("SEI kinetic rate constant [m.s-1]")
        self.U_sei_dim = pybamm.Parameter("SEI open-circuit potential [V]")

        # Li plating parameters
        self.V_bar_plated_Li = pybamm.Parameter(
            "Lithium metal partial molar volume [m3.mol-1]"
        )
        self.c_plated_Li_0_dim = pybamm.Parameter(
            "Initial plated lithium concentration [mol.m-3]"
        )

        # Initial conditions
        # Note: the initial concentration in the electrodes can be set as a function
        # of through-cell position, so is defined later as a function
        self.c_e_init_dimensional = pybamm.Parameter(
            "Initial concentration in electrolyte [mol.m-3]"
        )

        self.alpha_T_cell_dim = pybamm.Parameter(
            "Cell thermal expansion coefficient [m.K-1]"
        )

        # Total lithium
        c_e_av_init = pybamm.xyz_average(self.epsilon_init) * self.c_e_typ
        self.n_Li_e_init = c_e_av_init * self.L_x * self.A_cc

        if self.options["working electrode"] == "both":
            self.epsilon_s_n = pybamm.FunctionParameter(
                "Negative electrode active material volume fraction",
                {"Through-cell distance (x_n) [m]": x_n},
            )
            self.epsilon_inactive_n = 1 - self.n.epsilon_init - self.epsilon_s_n
            self.c_n_init = (
                pybamm.FunctionParameter(
                    "Initial concentration in negative electrode [mol.m-3]",
                    {
                        "Radial distance (r_n) [m]": r_n,
                        "Through-cell distance (x_n) [m]": pybamm.PrimaryBroadcast(
                            x_n, "negative particle"
                        ),
                    },
                )
                / self.n.c_max
            )
            c_n_init_av = pybamm.xyz_average(pybamm.r_average(self.c_n_init))
            eps_c_n_init_av = pybamm.xyz_average(
                self.epsilon_s_n * pybamm.r_average(self.c_n_init)
            )
            self.n_Li_n_init = eps_c_n_init_av * self.n.c_max * self.L_n * self.A_cc

            eps_s_n_av = pybamm.xyz_average(self.epsilon_s_n)
            self.neg_elec_loading = eps_s_n_av * self.L_n * self.n.c_max * self.F / 3600
            self.C_n_init = self.neg_elec_loading * self.A_cc
        else:
            self.n_Li_n_init = pybamm.Scalar(0)

        self.epsilon_s_p = pybamm.FunctionParameter(
            "Positive electrode active material volume fraction",
            {"Through-cell distance (x_p) [m]": x_p},
        )
        self.c_p_init = (
            pybamm.FunctionParameter(
                "Initial concentration in positive electrode [mol.m-3]",
                {
                    "Radial distance (r_p) [m]": r_p,
                    "Through-cell distance (x_p) [m]": pybamm.PrimaryBroadcast(
                        x_p, "positive particle"
                    ),
                },
            )
            / self.p.c_max
        )
        c_p_init_av = pybamm.xyz_average(pybamm.r_average(self.c_p_init))
        eps_c_p_init_av = pybamm.xyz_average(
            self.epsilon_s_p * pybamm.r_average(self.c_p_init)
        )
        self.n_Li_p_init = eps_c_p_init_av * self.p.c_max * self.L_p * self.A_cc

        self.n_Li_particles_init = self.n_Li_n_init + self.n_Li_p_init
        self.n_Li_init = self.n_Li_particles_init + self.n_Li_e_init

        self.epsilon_inactive_s = 1 - self.s.epsilon_init
        self.epsilon_inactive_p = 1 - self.p.epsilon_init - self.epsilon_s_p

        eps_s_p_av = pybamm.xyz_average(self.epsilon_s_p)
        self.pos_elec_loading = eps_s_p_av * self.L_p * self.p.c_max * self.F / 3600
        self.C_p_init = self.pos_elec_loading * self.A_cc

        # Reference OCP based on initial concentration
        self.T_init_dim = self.therm.T_init_dim
        self.T_init = self.therm.T_init
        if self.options["working electrode"] == "both":
            self.U_n_ref = self.U_n_dimensional(c_n_init_av, self.T_ref)
            self.U_n_init_dim = self.U_n_dimensional(c_n_init_av, self.T_init_dim)
        else:
            self.U_n_ref = pybamm.Scalar(0)
            self.U_n_init_dim = pybamm.Scalar(0)
        self.U_p_ref = self.U_p_dimensional(c_p_init_av, self.T_ref)

        self.ocv_ref = self.U_p_ref - self.U_n_ref
        self.U_p_init_dim = self.U_p_dimensional(c_p_init_av, self.T_init_dim)
        self.ocv_init_dim = self.U_p_init_dim - self.U_n_init_dim

    def D_e_dimensional(self, c_e, T):
        """Dimensional diffusivity in electrolyte"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        return pybamm.FunctionParameter("Electrolyte diffusivity [m2.s-1]", inputs)

    def kappa_e_dimensional(self, c_e, T):
        """Dimensional electrolyte conductivity"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        return pybamm.FunctionParameter("Electrolyte conductivity [S.m-1]", inputs)

    def j0_stripping_dimensional(self, c_e, c_Li, T):
        """Dimensional exchange-current density for stripping [A.m-2]"""
        inputs = {
            "Electrolyte concentration [mol.m-3]": c_e,
            "Plated lithium concentration [mol.m-3]": c_Li,
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Exchange-current density for stripping [A.m-2]", inputs
        )

    def j0_plating_dimensional(self, c_e, c_Li, T):
        """Dimensional exchange-current density for plating [A.m-2]"""
        inputs = {
            "Electrolyte concentration [mol.m-3]": c_e,
            "Plated lithium concentration [mol.m-3]": c_Li,
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Exchange-current density for plating [A.m-2]", inputs
        )

    def _set_scales(self):
        """Define the scales used in the non-dimensionalisation scheme"""

        # Microscale
        self.R_n_typ = self.geo.R_n_typ
        self.R_p_typ = self.geo.R_p_typ
        if self.options["particle shape"] == "spherical":
            if self.half_cell:
                self.a_n_typ = pybamm.Scalar(1)
            else:
                self.a_n_typ = 3 * pybamm.xyz_average(self.epsilon_s_n) / self.R_n_typ
            self.a_p_typ = 3 * pybamm.xyz_average(self.epsilon_s_p) / self.R_p_typ

        # Concentration
        self.electrolyte_concentration_scale = self.c_e_typ

        # Electrical
        # Both potential scales are the same but they have different units
        self.potential_scale = self.R * self.T_ref / self.F  # volts
        self.potential_scale_eV = self.k_b / self.q_e * self.T_ref  # eV
        self.current_scale = self.i_typ
        self.current_scale.print_name = "I_typ"

        # Thermal
        self.Delta_T = self.therm.Delta_T

        # Velocity scale
        self.velocity_scale = pybamm.Scalar(1)

        # Discharge timescale
        if self.options["working electrode"] == "positive":
            self.c_max = self.p.c_max
        else:
            self.c_max = self.n.c_max
        self.tau_discharge = self.F * self.c_max * self.L_x / self.i_typ

        # Electrolyte diffusion timescale
        self.D_e_typ = self.D_e_dimensional(self.c_e_typ, self.T_ref)
        self.tau_diffusion_e = self.L_x ** 2 / self.D_e_typ

        # Thermal diffusion timescale
        self.tau_th_yz = self.therm.tau_th_yz

        # Choose discharge timescale
        if self.options["timescale"] == "default":
            self.timescale = self.tau_discharge
        else:
            self.timescale = pybamm.Scalar(self.options["timescale"])

    def _set_dimensionless_parameters(self):
        """Defines the dimensionless parameters"""

        # Timescale ratios
        self.C_n = self.tau_diffusion_n / self.timescale
        self.C_p = self.tau_diffusion_p / self.timescale
        self.C_e = self.tau_diffusion_e / self.timescale
        self.C_r_n = self.tau_r_n / self.timescale
        self.C_r_p = self.tau_r_p / self.timescale
        self.C_th = self.tau_th_yz / self.timescale

        # Concentration ratios
        self.gamma_e = (self.tau_discharge / self.timescale) * self.c_e_typ / self.c_max

        # Macroscale Geometry
        self.l_cn = self.geo.l_cn
        self.l_n = self.geo.l_n
        self.l_s = self.geo.l_s
        self.l_p = self.geo.l_p
        self.l_cp = self.geo.l_cp
        self.l_x = self.geo.l_x
        self.l_y = self.geo.l_y
        self.l_z = self.geo.l_z
        self.r_inner = self.geo.r_inner
        self.r_outer = self.geo.r_outer
        self.a_cc = self.geo.a_cc
        self.a_cooling = self.geo.a_cooling
        self.v_cell = self.geo.v_cell
        self.l = self.geo.l
        self.delta = self.geo.delta

        # Tab geometry (for pouch cells)
        self.l_tab_n = self.geo.l_tab_n
        self.centre_y_tab_n = self.geo.centre_y_tab_n
        self.centre_z_tab_n = self.geo.centre_z_tab_n
        self.l_tab_p = self.geo.l_tab_p
        self.centre_y_tab_p = self.geo.centre_y_tab_p
        self.centre_z_tab_p = self.geo.centre_z_tab_p

        # Microscale geometry
        self.R_n = self.geo.R_n
        self.R_p = self.geo.R_p
        self.a_R_n = self.a_n_typ * self.R_n_typ
        self.a_R_p = self.a_p_typ * self.R_p_typ

        # Particle-size distribution geometry
        self.R_min_n = self.geo.R_min_n
        self.R_min_p = self.geo.R_min_p
        self.R_max_n = self.geo.R_max_n
        self.R_max_p = self.geo.R_max_p
        self.sd_a_n = self.geo.sd_a_n
        self.sd_a_p = self.geo.sd_a_p
        self.f_a_dist_n = self.geo.f_a_dist_n
        self.f_a_dist_p = self.geo.f_a_dist_p

        # Electrolyte Properties
        self.beta_surf = pybamm.Scalar(0)

        # Electrical
        self.voltage_low_cut = (
            self.voltage_low_cut_dimensional - (self.U_p_ref - self.U_n_ref)
        ) / self.potential_scale
        self.voltage_high_cut = (
            self.voltage_high_cut_dimensional - (self.U_p_ref - self.U_n_ref)
        ) / self.potential_scale

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

        # SEI parameters
        self.C_sei_reaction = (self.j_scale_n / self.m_sei_dimensional) * pybamm.exp(
            -(self.F * self.U_n_ref / (2 * self.R * self.T_ref))
        )

        self.C_sei_solvent = (
            self.j_scale_n
            * self.L_sei_0_dim
            / (self.c_sol_dimensional * self.F * self.D_sol_dimensional)
        )

        self.C_sei_electron = (
            self.j_scale_n
            * self.F
            * self.L_sei_0_dim
            / (self.kappa_inner_dimensional * self.R * self.T_ref)
        )

        self.C_sei_inter = (
            self.j_scale_n
            * self.L_sei_0_dim
            / (self.D_li_dimensional * self.c_li_0_dimensional * self.F)
        )

        self.U_inner_electron = self.F * self.U_inner_dimensional / self.R / self.T_ref

        self.R_sei = (
            self.F
            * self.j_scale_n
            * self.R_sei_dimensional
            * self.L_sei_0_dim
            / self.R
            / self.T_ref
        )

        self.v_bar = self.V_bar_outer_dimensional / self.V_bar_inner_dimensional
        self.c_sei_scale = (
            self.L_sei_0_dim * self.a_n_typ / self.V_bar_inner_dimensional
        )
        self.c_sei_outer_scale = (
            self.L_sei_0_dim * self.a_n_typ / self.V_bar_outer_dimensional
        )

        self.L_inner_0 = self.L_inner_0_dim / self.L_sei_0_dim
        self.L_outer_0 = self.L_outer_0_dim / self.L_sei_0_dim

        # ratio of SEI reaction scale to intercalation reaction
        self.Gamma_SEI = (
            self.V_bar_inner_dimensional * self.j_scale_n * self.timescale
        ) / (self.F * self.L_sei_0_dim)

        # EC reaction
        self.C_ec = (
            self.L_sei_0_dim
            * self.j_scale_n
            / (self.F * self.c_ec_0_dim * self.D_ec_dim)
        )
        self.C_sei_ec = (
            self.F
            * self.k_sei_dim
            * self.c_ec_0_dim
            / self.j_scale_n
            * (
                pybamm.exp(
                    -(
                        self.F
                        * (self.U_n_ref - self.U_sei_dim)
                        / (2 * self.R * self.T_ref)
                    )
                )
            )
        )
        self.beta_sei = self.a_n_typ * self.L_sei_0_dim * self.Gamma_SEI
        self.c_sei_init = self.c_ec_0_dim / self.c_sei_outer_scale

        # lithium plating parameters
        self.c_Li_typ = pybamm.Parameter(
            "Typical plated lithium concentration [mol.m-3]"
        )
        self.c_plated_Li_0 = self.c_plated_Li_0_dim / self.c_Li_typ

        # ratio of lithium plating reaction scaled to intercalation reaction
        self.Gamma_plating = (self.a_n_typ * self.j_scale_n * self.timescale) / (
            self.F * self.c_Li_typ
        )

        self.beta_plating = self.Gamma_plating * self.V_bar_plated_Li * self.c_Li_typ

        # Initial conditions
        self.c_e_init = self.c_e_init_dimensional / self.c_e_typ
        self.ocv_init = (self.ocv_init_dim - self.ocv_ref) / self.potential_scale

        # Dimensionless mechanical parameters
        self.t0_cr = 3600 / self.C_rate / self.timescale

    def chi(self, c_e, T):
        """
        Thermodynamic factor:
            (1-2*t_plus) is for Nernst-Planck,
            2*(1-t_plus) for Stefan-Maxwell,
        see Bizeray et al (2016) "Resolving a discrepancy ...".
        """
        return (2 * (1 - self.t_plus(c_e, T))) * (self.one_plus_dlnf_dlnc(c_e, T))

    def t_plus(self, c_e, T):
        """Cation transference number (dimensionless)"""
        inputs = {
            "Electrolyte concentration [mol.m-3]": c_e * self.c_e_typ,
            "Temperature [K]": self.Delta_T * T + self.T_ref,
        }
        return pybamm.FunctionParameter("Cation transference number", inputs)

    def one_plus_dlnf_dlnc(self, c_e, T):
        """Thermodynamic factor (dimensionless)"""
        inputs = {
            "Electrolyte concentration [mol.m-3]": c_e * self.c_e_typ,
            "Temperature [K]": self.Delta_T * T + self.T_ref,
        }
        return pybamm.FunctionParameter("1 + dlnf/dlnc", inputs)

    def D_e(self, c_e, T):
        """Dimensionless electrolyte diffusivity"""
        c_e_dimensional = c_e * self.c_e_typ
        T_dim = self.Delta_T * T + self.T_ref
        return self.D_e_dimensional(c_e_dimensional, T_dim) / self.D_e_typ

    def kappa_e(self, c_e, T):
        """Dimensionless electrolyte conductivity"""
        c_e_dimensional = c_e * self.c_e_typ
        kappa_scale = self.F ** 2 * self.D_e_typ * self.c_e_typ / (self.R * self.T_ref)
        T_dim = self.Delta_T * T + self.T_ref
        return self.kappa_e_dimensional(c_e_dimensional, T_dim) / kappa_scale

    def j0_stripping(self, c_e, c_Li, T):
        """Dimensionless exchange-current density for stripping"""
        c_e_dim = c_e * self.c_e_typ
        c_Li_dim = c_Li * self.c_Li_typ
        T_dim = self.Delta_T * T + self.T_ref

        return self.j0_stripping_dimensional(c_e_dim, c_Li_dim, T_dim) / self.j_scale_n

    def j0_plating(self, c_e, c_Li, T):
        """Dimensionless reverse plating current"""
        c_e_dim = c_e * self.c_e_typ
        c_Li_dim = c_Li * self.c_Li_typ
        T_dim = self.Delta_T * T + self.T_ref

        return self.j0_plating_dimensional(c_e_dim, c_Li_dim, T_dim) / self.j_scale_n

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
            self.dimensional_current_with_time / self.I_typ * pybamm.sign(self.I_typ)
        )

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, extra_options):
        self._options = pybamm.BatteryModelOptions(extra_options)


class DomainParameters(BaseParameters):
    def __init__(self, domain, main_param, x, r):
        self.domain = domain
        self.main_param = main_param
        self.x = x
        self.r = r

    def _set_dimensional_parameters(self):
        main = self.main_param
        Domain = self.domain
        domain = Domain.lower()
        d = domain[0]
        x = self.x

        if Domain == "Separator":
            self.epsilon_init = pybamm.FunctionParameter(
                "Separator porosity", {"Through-cell distance (x_s) [m]": x}
            )
            return

        # Particle properties
        self.c_max = pybamm.Parameter(
            f"Maximum concentration in {domain} electrode [mol.m-3]"
        )
        self.sigma_cc_dimensional = pybamm.Parameter(
            f"{Domain} current collector conductivity [S.m-1]"
        )
        self.epsilon_init = pybamm.FunctionParameter(
            f"{Domain} electrode porosity",
            {f"Through-cell distance (x_{d}) [m]": x},
        )

        # Electrochemical reactions
        self.ne = pybamm.Parameter(f"{Domain} electrode electrons in reaction")
        self.C_dl_dimensional = pybamm.Parameter(
            f"{Domain} electrode double-layer capacity [F.m-2]"
        )

        # Intercalation kinetics
        self.mhc_lambda_dimensional = pybamm.Parameter(
            f"{Domain} electrode reorganization energy [eV]"
        )
        self.alpha_bv = pybamm.Parameter(
            f"{Domain} electrode Butler-Volmer transfer coefficient"
        )

        # Mechanical parameters
        self.nu = pybamm.Parameter(f"{Domain} electrode Poisson's ratio")
        self.E = pybamm.Parameter(f"{Domain} electrode Young's modulus [Pa]")
        self.c_0_dim = pybamm.Parameter(
            f"{Domain} electrode reference concentration for free of deformation "
            "[mol.m-3]"
        )
        self.Omega = pybamm.Parameter(
            f"{Domain} electrode partial molar volume [m3.mol-1]"
        )
        self.l_cr_0 = pybamm.Parameter(f"{Domain} electrode initial crack length [m]")
        self.w_cr = pybamm.Parameter(f"{Domain} electrode initial crack width [m]")
        self.rho_cr_dim = pybamm.Parameter(
            f"{Domain} electrode number of cracks per unit area [m-2]"
        )
        self.b_cr = pybamm.Parameter(f"{Domain} electrode Paris' law constant b")
        self.m_cr = pybamm.Parameter(f"{Domain} electrode Paris' law constant m")
        self.Eac_cr = pybamm.Parameter(
            f"{Domain} electrode activation energy for cracking rate [kJ.mol-1]"
        )
        # intermediate variables  [K*m^3/mol]
        self.theta_dim = (
            (self.Omega / main.R) * 2 * self.Omega * self.E / 9 / (1 - self.nu)
        )

        # Loss of active material parameters
        self.m_LAM = pybamm.Parameter(
            f"{Domain} electrode LAM constant exponential term"
        )
        self.beta_LAM_dimensional = pybamm.Parameter(
            f"{Domain} electrode LAM constant proportional term [s-1]"
        )
        self.stress_critical_dim = pybamm.Parameter(
            f"{Domain} electrode critical stress [Pa]"
        )
        self.beta_LAM_sei_dimensional = pybamm.Parameter(
            f"{Domain} electrode reaction-driven LAM factor [m3.mol-1]"
        )

        # utilisation parameters
        self.u_init = pybamm.Parameter(
            f"Initial {domain} electrode interface utilisation"
        )
        self.beta_utilisation_dimensional = pybamm.Parameter(
            f"{Domain} electrode current-driven interface utilisation factor [m3.mol-1]"
        )

    def sigma_dimensional(self, T):
        """Dimensional electrical conductivity in negative electrode"""
        inputs = {"Temperature [K]": T}
        return pybamm.FunctionParameter(
            f"{self.domain} electrode conductivity [S.m-1]", inputs
        )

    def D_dimensional(self, sto, T):
        """Dimensional diffusivity in negative particle. Note this is defined as a
        function of stochiometry"""
        inputs = {f"{self.domain} particle stoichiometry": sto, "Temperature [K]": T}
        return pybamm.FunctionParameter(
            f"{self.domain} electrode diffusivity [m2.s-1]", inputs
        )

    def j0_dimensional(self, c_e, c_s_surf, T):
        """Dimensional negative exchange-current density [A.m-2]"""
        inputs = {
            "Electrolyte concentration [mol.m-3]": c_e,
            f"{self.domain} particle surface concentration [mol.m-3]": c_s_surf,
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            f"{self.domain} electrode exchange-current density [A.m-2]", inputs
        )

    def j0_stripping_dimensional(self, c_e, c_Li, T):
        """Dimensional exchange-current density for stripping [A.m-2]"""
        inputs = {
            "Electrolyte concentration [mol.m-3]": c_e,
            "Plated lithium concentration [mol.m-3]": c_Li,
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Exchange-current density for stripping [A.m-2]", inputs
        )

    def j0_plating_dimensional(self, c_e, c_Li, T):
        """Dimensional exchange-current density for plating [A.m-2]"""
        inputs = {
            "Electrolyte concentration [mol.m-3]": c_e,
            "Plated lithium concentration [mol.m-3]": c_Li,
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Exchange-current density for plating [A.m-2]", inputs
        )

    def U_dimensional(self, sto, T):
        """Dimensional open-circuit potential in the negative electrode [V]"""
        inputs = {f"{self.domain} particle stoichiometry": sto}
        u_ref = pybamm.FunctionParameter(f"{self.domain} electrode OCP [V]", inputs)
        # add a term to ensure that the OCP goes to infinity at 0 and -infinity at 1
        # this will not affect the OCP for most values of sto
        # see #1435
        u_ref = u_ref + 1e-6 * (1 / sto + 1 / (sto - 1))
        dudt_dim_func = self.dUdT_dimensional(sto)
        d = self.domain.lower()[0]
        dudt_dim_func.print_name = r"\frac{dU}{dT_{" + d + r"}}"
        return u_ref + (T - self.T_ref) * dudt_dim_func

    def dUdT_dimensional(self, sto):
        """
        Dimensional entropic change of the negative electrode open-circuit
        potential [V.K-1]
        """
        inputs = {f"{self.domain} particle stoichiometry": sto}
        return pybamm.FunctionParameter(
            f"{self.domain} electrode OCP entropic change [V.K-1]", inputs
        )

    def _set_scales(self):
        """Define the scales used in the non-dimensionalisation scheme"""
        main = self.main_param

        # Microscale
        if self.domain == "Negative":
            self.R_typ = main.geo.R_n_typ
        elif self.domain == "Positive":
            self.R_typ = main.geo.R_p_typ

        if self.half_cell and self.domain == "Negative":
            self.a_typ = pybamm.Scalar(1)
        elif self.options["particle shape"] == "spherical":
            self.a_typ = 3 * pybamm.xyz_average(self.epsilon_s) / self.R_typ

        # Concentration
        self.electrolyte_concentration_scale = self.c_e_typ
        self.particle_concentration_scale = self.c_max

        # Scale for interfacial current density in A/m2
        if self.half_cell and self.domain == "Negative":
            # metal electrode (boundary condition between negative and separator)
            self.j_scale = self.i_typ
        else:
            # porous electrode
            self.j_scale = self.i_typ / (self.a_typ * self.L_x)

        # Reference exchange-current density
        self.j0_ref_dimensional = (
            self.j0_dimensional(self.c_e_typ, self.c_max / 2, self.T_ref) * 2
        )

        # Reaction timescales
        self.tau_r = main.F * self.c_max / (self.j0_ref_dimensional * self.a_typ)
        # Particle diffusion timescales
        self.D_typ_dim = self.D_dimensional(pybamm.Scalar(1), self.T_ref)
        self.tau_diffusion = self.R_typ ** 2 / self.D_typ_dim

    def _set_dimensionless_parameters(self):
        main = self.main_param

        # Timescale ratios
        self.C_diff = self.tau_diffusion / main.timescale
        self.C_r = self.tau_r / main.timescale

        # Concentration ratios
        # In most cases gamma_n will be equal to 1
        self.gamma = (main.tau_discharge / main.timescale) * self.c_max / self.c_max

        # Electrode Properties
        self.sigma_cc = (
            self.sigma_cc_dimensional * self.potential_scale / self.i_typ / self.L_x
        )
        self.sigma_cc_prime = self.sigma_cc * self.delta ** 2
        self.sigma_cc_dbl_prime = self.sigma_cc_prime * self.delta

        # Electrolyte Properties
        self.beta_surf = pybamm.Scalar(0)

        # Electrochemical Reactions
        self.C_dl = (
            self.C_dl_dimensional * self.potential_scale / self.j_scale / main.timescale
        )

        # Intercalation kinetics
        self.mhc_lambda = self.mhc_lambda_dimensional / self.potential_scale_eV

        # Initial conditions
        if self.half_cell and self.domain == "Negative":
            self.U_init = pybamm.Scalar(0)
        else:
            self.U_init = (self.U_init_dim - self.U_ref) / self.potential_scale

        # Dimensionless mechanical parameters
        self.rho_cr = self.rho_cr_dim * self.l_cr_0 * self.w_cr
        self.theta = self.theta_dim * self.c_max / self.T_ref
        self.c_0 = self.c_0_dim / self.c_max
        self.beta_LAM = self.beta_LAM_dimensional * main.timescale
        # normalised typical time for one cycle
        self.stress_critical = self.stress_critical_dim / self.E
        # Reaction-driven LAM parameters
        self.beta_LAM_sei = (
            self.beta_LAM_sei_dimensional * self.a_typ * self.j_scale * main.timescale
        ) / main.F
        # Utilisation factors
        self.beta_utilisation = (
            self.beta_utilisation_dimensional
            * self.a_typ
            * self.j_scale
            * main.timescale
        ) / main.F

    def sigma(self, T):
        """Dimensionless electrode electrical conductivity"""
        T_dim = self.Delta_T * T + self.T_ref
        return (
            self.sigma_dimensional(T_dim) * self.potential_scale / self.i_typ / self.L_x
        )

    def sigma_prime(self, T):
        """Rescaled dimensionless electrode electrical conductivity"""
        return self.sigma(T) * self.delta

    def D(self, c_s, T):
        """Dimensionless particle diffusivity"""
        sto = c_s
        T_dim = self.Delta_T * T + self.T_ref
        return self.D_dimensional(sto, T_dim) / self.D_typ_dim

    def j0(self, c_e, c_s_surf, T):
        """Dimensionless exchange-current density"""
        c_e_dim = c_e * self.c_e_typ
        c_s_surf_dim = c_s_surf * self.n.c_max
        T_dim = self.Delta_T * T + self.T_ref

        return (
            self.j0_dimensional(c_e_dim, c_s_surf_dim, T_dim) / self.j0_ref_dimensional
        )

    def U(self, c_s, T):
        """Dimensionless open-circuit potential in the electrode"""
        sto = c_s
        T_dim = self.Delta_T * T + self.T_ref
        return (self.U_dimensional(sto, T_dim) - self.U_ref) / self.potential_scale

    def dUdT(self, c_s):
        """Dimensionless entropic change in open-circuit potential"""
        sto = c_s
        return self.dUdT_dimensional(sto) * self.Delta_T / self.potential_scale

    def t_change(self, sto):
        """
        Dimensionless volume change for the electrode;
        sto should be R-averaged
        """
        return pybamm.FunctionParameter(
            f"{self.domain} electrode volume change", {"Particle stoichiometry": sto}
        )

    def k_cr(self, T):
        """
        Dimensionless cracking rate for the electrode;
        """
        T_dim = self.Delta_T * T + self.T_ref
        delta_k_cr = self.E ** self.m_cr * self.l_cr_0 ** (self.m_cr / 2 - 1)
        return (
            pybamm.FunctionParameter(
                f"{self.domain} electrode cracking rate", {"Temperature [K]": T_dim}
            )
            * delta_k_cr
        )
