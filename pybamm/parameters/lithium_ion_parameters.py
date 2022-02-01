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

        # Set parameters and scales
        self._set_dimensional_parameters()
        self._set_scales()
        self._set_dimensionless_parameters()

        # Set input current
        self._set_input_current()

    def _set_dimensional_parameters(self):
        """Defines the dimensional parameters"""

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

        # Electrode properties
        self.c_n_max = pybamm.Parameter(
            "Maximum concentration in negative electrode [mol.m-3]"
        )
        self.c_p_max = pybamm.Parameter(
            "Maximum concentration in positive electrode [mol.m-3]"
        )
        self.sigma_cn_dimensional = pybamm.Parameter(
            "Negative current collector conductivity [S.m-1]"
        )
        self.sigma_cp_dimensional = pybamm.Parameter(
            "Positive current collector conductivity [S.m-1]"
        )

        # Microscale geometry
        # Note: the surface area to volume ratio is defined later with the function
        # parameters. The particle size as a function of through-cell position is
        # already defined in geometric_parameters.py
        self.R_n_dimensional = self.geo.R_n_dimensional
        self.R_p_dimensional = self.geo.R_p_dimensional

        inputs = {
            "Through-cell distance (x_n) [m]": pybamm.standard_spatial_vars.x_n
            * self.L_x
        }
        self.epsilon_n_init = pybamm.FunctionParameter(
            "Negative electrode porosity", inputs
        )
        inputs = {
            "Through-cell distance (x_s) [m]": pybamm.standard_spatial_vars.x_s
            * self.L_x
        }
        self.epsilon_s_init = pybamm.FunctionParameter("Separator porosity", inputs)

        inputs = {
            "Through-cell distance (x_p) [m]": pybamm.standard_spatial_vars.x_p
            * self.L_x
        }
        self.epsilon_p_init = pybamm.FunctionParameter(
            "Positive electrode porosity", inputs
        )

        if self.half_cell:
            self.epsilon_init = pybamm.concatenation(
                self.epsilon_s_init, self.epsilon_p_init
            )
        else:
            self.epsilon_init = pybamm.concatenation(
                self.epsilon_n_init, self.epsilon_s_init, self.epsilon_p_init
            )

        self.epsilon_inactive_n = (
            1 - self.epsilon_n_init - self.epsilon_s_n(pybamm.standard_spatial_vars.x_n)
        )
        self.epsilon_inactive_s = 1 - self.epsilon_s_init
        self.epsilon_inactive_p = (
            1 - self.epsilon_p_init - self.epsilon_s_p(pybamm.standard_spatial_vars.x_p)
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

        # Electrochemical reactions
        self.ne_n = pybamm.Parameter("Negative electrode electrons in reaction")
        self.ne_p = pybamm.Parameter("Positive electrode electrons in reaction")
        self.C_dl_n_dimensional = pybamm.Parameter(
            "Negative electrode double-layer capacity [F.m-2]"
        )
        self.C_dl_p_dimensional = pybamm.Parameter(
            "Positive electrode double-layer capacity [F.m-2]"
        )

        # Intercalation kinetics
        self.mhc_lambda_n_dimensional = pybamm.Parameter(
            "Negative electrode reorganization energy [eV]"
        )
        self.mhc_lambda_p_dimensional = pybamm.Parameter(
            "Positive electrode reorganization energy [eV]"
        )
        self.alpha_bv_n = pybamm.Parameter(
            "Negative electrode Butler-Volmer transfer coefficient"
        )
        self.alpha_bv_p = pybamm.Parameter(
            "Positive electrode Butler-Volmer transfer coefficient"
        )

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

        # Mechanical parameters
        self.nu_n = pybamm.Parameter("Negative electrode Poisson's ratio")
        self.nu_p = pybamm.Parameter("Positive electrode Poisson's ratio")
        self.E_n = pybamm.Parameter("Negative electrode Young's modulus [Pa]")
        self.E_p = pybamm.Parameter("Positive electrode Young's modulus [Pa]")
        self.c_n_0_dim = pybamm.Parameter(
            "Negative electrode reference concentration for free of deformation "
            "[mol.m-3]"
        )
        self.c_p_0_dim = pybamm.Parameter(
            "Positive electrode reference concentration for free of deformation "
            "[mol.m-3]"
        )
        self.Omega_n = pybamm.Parameter(
            "Negative electrode partial molar volume [m3.mol-1]"
        )
        self.Omega_p = pybamm.Parameter(
            "Positive electrode partial molar volume [m3.mol-1]"
        )
        self.l_cr_n_0 = pybamm.Parameter("Negative electrode initial crack length [m]")
        self.l_cr_p_0 = pybamm.Parameter("Positive electrode initial crack length [m]")
        self.w_cr = pybamm.Parameter("Negative electrode initial crack width [m]")
        self.rho_cr_n_dim = pybamm.Parameter(
            "Negative electrode number of cracks per unit area [m-2]"
        )
        self.rho_cr_p_dim = pybamm.Parameter(
            "Positive electrode number of cracks per unit area [m-2]"
        )
        self.b_cr_n = pybamm.Parameter("Negative electrode Paris' law constant b")
        self.b_cr_p = pybamm.Parameter("Positive electrode Paris' law constant b")
        self.m_cr_n = pybamm.Parameter("Negative electrode Paris' law constant m")
        self.m_cr_p = pybamm.Parameter("Positive electrode Paris' law constant m")
        self.Eac_cr_n = pybamm.Parameter(
            "Negative electrode activation energy for cracking rate [kJ.mol-1]"
        )  # noqa
        self.Eac_cr_p = pybamm.Parameter(
            "Positive electrode activation energy for cracking rate [kJ.mol-1]"
        )  # noqa
        # intermediate variables  [K*m^3/mol]
        self.theta_n_dim = (
            (self.Omega_n / self.R) * 2 * self.Omega_n * self.E_n / 9 / (1 - self.nu_n)
        )
        self.theta_p_dim = (
            (self.Omega_p / self.R) * 2 * self.Omega_p * self.E_p / 9 / (1 - self.nu_p)
        )
        self.alpha_T_cell_dim = pybamm.Parameter(
            "Cell thermal expansion coefficient [m.K-1]"
        )

        # Electrode capacities
        x_n = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian"
        )
        x_p = pybamm.SpatialVariable(
            "x_p", domain=["positive electrode"], coord_sys="cartesian"
        )

        eps_s_n_av = pybamm.x_average(self.epsilon_s_n(x_n))
        eps_s_p_av = pybamm.x_average(self.epsilon_s_p(x_p))
        self.neg_elec_loading = eps_s_n_av * self.L_n * self.c_n_max * self.F / 3600
        self.pos_elec_loading = eps_s_p_av * self.L_p * self.c_p_max * self.F / 3600
        self.C_n_init = self.neg_elec_loading * self.A_cc
        self.C_p_init = self.pos_elec_loading * self.A_cc

        # Total lithium
        c_e_av_init = pybamm.x_average(self.epsilon_init) * self.c_e_typ
        self.n_Li_e_init = c_e_av_init * self.L_x * self.A_cc

        if self.options["working electrode"] == "both":
            eps_s_n = self.epsilon_s_n(x_n)
            c_n = self.c_n_init(x_n)
            c_n_av = pybamm.x_average(eps_s_n * c_n)
            self.n_Li_n_init = c_n_av * self.c_n_max * self.L_n * self.A_cc
        else:
            self.n_Li_n_init = pybamm.Scalar(0)

        eps_s_p = self.epsilon_s_p(x_p)
        c_p = self.c_p_init(x_p)
        c_p_av = pybamm.x_average(eps_s_p * c_p)
        self.n_Li_p_init = c_p_av * self.c_p_max * self.L_p * self.A_cc

        self.n_Li_particles_init = self.n_Li_n_init + self.n_Li_p_init
        self.n_Li_init = self.n_Li_particles_init + self.n_Li_e_init

        # Loss of active material parameters
        self.m_LAM_n = pybamm.Parameter(
            "Negative electrode LAM constant exponential term"
        )
        self.beta_LAM_n_dimensional = pybamm.Parameter(
            "Negative electrode LAM constant proportional term [s-1]"
        )
        self.stress_critical_n_dim = pybamm.Parameter(
            "Negative electrode critical stress [Pa]"
        )
        self.m_LAM_p = pybamm.Parameter(
            "Positive electrode LAM constant exponential term"
        )
        self.beta_LAM_p_dimensional = pybamm.Parameter(
            "Positive electrode LAM constant proportional term [s-1]"
        )
        self.stress_critical_p_dim = pybamm.Parameter(
            "Positive electrode critical stress [Pa]"
        )
        self.beta_LAM_sei_n_dimensional = pybamm.Parameter(
            "Negative electrode reaction-driven LAM factor [m3.mol-1]"
        )
        self.beta_LAM_sei_p_dimensional = pybamm.Parameter(
            "Positive electrode reaction-driven LAM factor [m3.mol-1]"
        )

        # utilisation parameters
        self.u_n_init = pybamm.Parameter(
            "Initial negative electrode interface utilisation"
        )
        self.u_p_init = pybamm.Parameter(
            "Initial positive electrode interface utilisation"
        )
        self.beta_utilisation_n_dimensional = pybamm.Parameter(
            "Negative electrode current-driven interface utilisation factor [m3.mol-1]"
        )
        self.beta_utilisation_p_dimensional = pybamm.Parameter(
            "Positive electrode current-driven interface utilisation factor [m3.mol-1]"
        )

    def sigma_n_dimensional(self, T):
        """Dimensional electrical conductivity in negative electrode"""
        inputs = {"Temperature [K]": T}
        return pybamm.FunctionParameter(
            "Negative electrode conductivity [S.m-1]", inputs
        )

    def sigma_p_dimensional(self, T):
        """Dimensional electrical conductivity in positive electrode"""
        inputs = {"Temperature [K]": T}
        return pybamm.FunctionParameter(
            "Positive electrode conductivity [S.m-1]", inputs
        )

    def D_e_dimensional(self, c_e, T):
        """Dimensional diffusivity in electrolyte"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        return pybamm.FunctionParameter("Electrolyte diffusivity [m2.s-1]", inputs)

    def kappa_e_dimensional(self, c_e, T):
        """Dimensional electrolyte conductivity"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        return pybamm.FunctionParameter("Electrolyte conductivity [S.m-1]", inputs)

    def D_n_dimensional(self, sto, T):
        """Dimensional diffusivity in negative particle. Note this is defined as a
        function of stochiometry"""
        inputs = {"Negative particle stoichiometry": sto, "Temperature [K]": T}
        return pybamm.FunctionParameter(
            "Negative electrode diffusivity [m2.s-1]", inputs
        )

    def D_p_dimensional(self, sto, T):
        """Dimensional diffusivity in positive particle. Note this is defined as a
        function of stochiometry"""
        inputs = {"Positive particle stoichiometry": sto, "Temperature [K]": T}
        return pybamm.FunctionParameter(
            "Positive electrode diffusivity [m2.s-1]", inputs
        )

    def j0_n_dimensional(self, c_e, c_s_surf, T):
        """Dimensional negative exchange-current density [A.m-2]"""
        inputs = {
            "Electrolyte concentration [mol.m-3]": c_e,
            "Negative particle surface concentration [mol.m-3]": c_s_surf,
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Negative electrode exchange-current density [A.m-2]", inputs
        )

    def j0_p_dimensional(self, c_e, c_s_surf, T):
        """Dimensional negative exchange-current density [A.m-2]"""
        inputs = {
            "Electrolyte concentration [mol.m-3]": c_e,
            "Positive particle surface concentration [mol.m-3]": c_s_surf,
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Positive electrode exchange-current density [A.m-2]", inputs
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

    def U_n_dimensional(self, sto, T):
        """Dimensional open-circuit potential in the negative electrode [V]"""
        inputs = {"Negative particle stoichiometry": sto}
        u_ref = pybamm.FunctionParameter("Negative electrode OCP [V]", inputs)
        # add a term to ensure that the OCP goes to infinity at 0 and -infinity at 1
        # this will not affect the OCP for most values of sto
        # see #1435
        u_ref = u_ref + 1e-6 * (1 / sto + 1 / (sto - 1))
        dudt_n_dim_func = self.dUdT_n_dimensional(sto)
        dudt_n_dim_func.print_name = r"\frac{dU}{dT_n}"
        return u_ref + (T - self.T_ref) * dudt_n_dim_func

    def U_p_dimensional(self, sto, T):
        """Dimensional open-circuit potential in the positive electrode [V]"""
        inputs = {"Positive particle stoichiometry": sto}
        u_ref = pybamm.FunctionParameter("Positive electrode OCP [V]", inputs)
        # add a term to ensure that the OCP goes to infinity at 0 and -infinity at 1
        # this will not affect the OCP for most values of sto
        # see #1435
        u_ref = u_ref + 1e-6 * (1 / sto + 1 / (sto - 1))
        dudt_p_dim_func = self.dUdT_p_dimensional(sto)
        dudt_p_dim_func.print_name = r"\frac{dU}{dT_p}"
        return u_ref + (T - self.T_ref) * dudt_p_dim_func

    def dUdT_n_dimensional(self, sto):
        """
        Dimensional entropic change of the negative electrode open-circuit
        potential [V.K-1]
        """
        inputs = {"Negative particle stoichiometry": sto}
        return pybamm.FunctionParameter(
            "Negative electrode OCP entropic change [V.K-1]", inputs
        )

    def dUdT_p_dimensional(self, sto):
        """
        Dimensional entropic change of the positive electrode open-circuit
        potential [V.K-1]
        """
        inputs = {"Positive particle stoichiometry": sto}
        return pybamm.FunctionParameter(
            "Positive electrode OCP entropic change [V.K-1]", inputs
        )

    def epsilon_s_n(self, x):
        """Negative electrode active material volume fraction"""
        inputs = {"Through-cell distance (x_n) [m]": x * self.L_x}
        return pybamm.FunctionParameter(
            "Negative electrode active material volume fraction", inputs
        )

    def epsilon_s_p(self, x):
        """Positive electrode active material volume fraction"""
        inputs = {"Through-cell distance (x_p) [m]": x * self.L_x}
        return pybamm.FunctionParameter(
            "Positive electrode active material volume fraction", inputs
        )

    def c_n_init_dimensional(self, x):
        """Initial concentration as a function of dimensionless position x"""
        inputs = {"Dimensionless through-cell position (x_n)": x}
        return pybamm.FunctionParameter(
            "Initial concentration in negative electrode [mol.m-3]", inputs
        )

    def c_p_init_dimensional(self, x):
        """Initial concentration as a function of dimensionless position x"""
        inputs = {"Dimensionless through-cell position (x_p)": x}
        return pybamm.FunctionParameter(
            "Initial concentration in positive electrode [mol.m-3]", inputs
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
                self.a_n_typ = 3 * self.epsilon_s_n(0) / self.R_n_typ
            self.a_p_typ = 3 * self.epsilon_s_p(1) / self.R_p_typ

        # Concentration
        self.electrolyte_concentration_scale = self.c_e_typ
        self.negative_particle_concentration_scale = self.c_n_max
        self.positive_particle_concentration_scale = self.c_p_max

        # Electrical
        # Both potential scales are the same but they have different units
        self.potential_scale = self.R * self.T_ref / self.F  # volts
        self.potential_scale_eV = self.k_b / self.q_e * self.T_ref  # eV
        self.current_scale = self.i_typ
        self.current_scale.print_name = "I_typ"
        # Scale for interfacial current density in A/m2
        if self.options["working electrode"] == "both":
            # porous electrode
            self.j_scale_n = self.i_typ / (self.a_n_typ * self.L_x)
        else:
            # metal electrode (boundary condition between negative and separator)
            self.j_scale_n = self.i_typ
        self.j_scale_p = self.i_typ / (self.a_p_typ * self.L_x)

        # Reference OCP based on initial concentration at
        # current collector/electrode interface
        if self.options["working electrode"] == "both":
            sto_n_init = self.c_n_init_dimensional(0) / self.c_n_max
            self.U_n_ref = self.U_n_dimensional(sto_n_init, self.T_ref)
        else:
            self.U_n_ref = pybamm.Scalar(0)

        # Reference OCP based on initial concentration at
        # current collector/electrode interface
        sto_p_init = self.c_p_init_dimensional(1) / self.c_p_max
        self.U_p_ref = self.U_p_dimensional(sto_p_init, self.T_ref)

        # Reference exchange-current density
        self.j0_n_ref_dimensional = (
            self.j0_n_dimensional(self.c_e_typ, self.c_n_max / 2, self.T_ref) * 2
        )
        self.j0_p_ref_dimensional = (
            self.j0_p_dimensional(self.c_e_typ, self.c_p_max / 2, self.T_ref) * 2
        )

        # Thermal
        self.Delta_T = self.therm.Delta_T

        # Velocity scale
        self.velocity_scale = pybamm.Scalar(1)

        # Discharge timescale
        if self.options["working electrode"] == "positive":
            self.c_max = self.c_p_max
        else:
            self.c_max = self.c_n_max
        self.tau_discharge = self.F * self.c_max * self.L_x / self.i_typ

        # Reaction timescales
        self.tau_r_n = (
            self.F * self.c_n_max / (self.j0_n_ref_dimensional * self.a_n_typ)
        )
        self.tau_r_p = (
            self.F * self.c_p_max / (self.j0_p_ref_dimensional * self.a_p_typ)
        )

        # Electrolyte diffusion timescale
        self.D_e_typ = self.D_e_dimensional(self.c_e_typ, self.T_ref)
        self.tau_diffusion_e = self.L_x ** 2 / self.D_e_typ

        # Particle diffusion timescales
        self.D_n_typ_dim = self.D_n_dimensional(pybamm.Scalar(1), self.T_ref)
        self.D_p_typ_dim = self.D_p_dimensional(pybamm.Scalar(1), self.T_ref)

        self.tau_diffusion_n = self.R_n_typ ** 2 / self.D_n_typ_dim
        self.tau_diffusion_p = self.R_p_typ ** 2 / self.D_p_typ_dim

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
        # In most cases gamma_n will be equal to 1
        self.gamma_n = (self.tau_discharge / self.timescale) * self.c_n_max / self.c_max
        self.gamma_p = (self.tau_discharge / self.timescale) * self.c_p_max / self.c_max

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

        # Electrode Properties
        self.sigma_cn = (
            self.sigma_cn_dimensional * self.potential_scale / self.i_typ / self.L_x
        )
        self.sigma_cp = (
            self.sigma_cp_dimensional * self.potential_scale / self.i_typ / self.L_x
        )
        self.sigma_cn_prime = self.sigma_cn * self.delta ** 2
        self.sigma_cp_prime = self.sigma_cp * self.delta ** 2
        self.sigma_cn_dbl_prime = self.sigma_cn_prime * self.delta
        self.sigma_cp_dbl_prime = self.sigma_cp_prime * self.delta

        # Electrolyte Properties
        self.beta_surf = pybamm.Scalar(0)
        self.beta_surf_n = pybamm.Scalar(0)
        self.beta_surf_p = pybamm.Scalar(0)

        # Electrochemical Reactions
        self.C_dl_n = (
            self.C_dl_n_dimensional
            * self.potential_scale
            / self.j_scale_n
            / self.timescale
        )
        self.C_dl_p = (
            self.C_dl_p_dimensional
            * self.potential_scale
            / self.j_scale_p
            / self.timescale
        )

        # Intercalation kinetics
        self.mhc_lambda_n = self.mhc_lambda_n_dimensional / self.potential_scale_eV
        self.mhc_lambda_p = self.mhc_lambda_p_dimensional / self.potential_scale_eV

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
        self.T_init = self.therm.T_init
        self.c_e_init = self.c_e_init_dimensional / self.c_e_typ

        if self.options["working electrode"] == "both":
            self.U_n_init = self.U_n(self.c_n_init(0), self.T_init)
        else:
            self.U_n_init = pybamm.Scalar(0)
        self.U_p_init = self.U_p(self.c_p_init(1), self.T_init)

        # Dimensionless mechanical parameters
        self.rho_cr_n = self.rho_cr_n_dim * self.l_cr_n_0 * self.w_cr
        self.rho_cr_p = self.rho_cr_p_dim * self.l_cr_p_0 * self.w_cr
        self.theta_p = self.theta_p_dim * self.c_p_max / self.T_ref
        self.theta_n = self.theta_n_dim * self.c_n_max / self.T_ref
        self.c_p_0 = self.c_p_0_dim / self.c_p_max
        self.c_n_0 = self.c_n_0_dim / self.c_n_max
        self.t0_cr = 3600 / self.C_rate / self.timescale
        self.beta_LAM_n = self.beta_LAM_n_dimensional * self.timescale
        self.beta_LAM_p = self.beta_LAM_p_dimensional * self.timescale
        # normalised typical time for one cycle
        self.stress_critical_n = self.stress_critical_n_dim / self.E_n
        self.stress_critical_p = self.stress_critical_p_dim / self.E_p
        # Reaction-driven LAM parameters
        self.beta_LAM_sei_n = (
            self.beta_LAM_sei_n_dimensional
            * self.a_n_typ
            * self.j_scale_n
            * self.timescale
        ) / self.F
        self.beta_LAM_sei_p = (
            self.beta_LAM_sei_p_dimensional
            * self.a_p_typ
            * self.j_scale_p
            * self.timescale
        ) / self.F
        # Utilisation factors
        self.beta_utilisation_n = (
            self.beta_utilisation_n_dimensional
            * self.a_n_typ
            * self.j_scale_n
            * self.timescale
        ) / self.F
        self.beta_utilisation_p = (
            self.beta_utilisation_p_dimensional
            * self.a_p_typ
            * self.j_scale_p
            * self.timescale
        ) / self.F

    def sigma_n(self, T):
        """Dimensionless negative electrode electrical conductivity"""
        T_dim = self.Delta_T * T + self.T_ref
        return (
            self.sigma_n_dimensional(T_dim)
            * self.potential_scale
            / self.i_typ
            / self.L_x
        )

    def sigma_p(self, T):
        """Dimensionless positive electrode electrical conductivity"""
        T_dim = self.Delta_T * T + self.T_ref
        return (
            self.sigma_p_dimensional(T_dim)
            * self.potential_scale
            / self.i_typ
            / self.L_x
        )

    def sigma_n_prime(self, T):
        """Rescaled dimensionless negative electrode electrical conductivity"""
        return self.sigma_n(T) * self.delta

    def sigma_p_prime(self, T):
        """Rescaled dimensionless positive electrode electrical conductivity"""
        return self.sigma_p(T) * self.delta

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

    def D_n(self, c_s_n, T):
        """Dimensionless negative particle diffusivity"""
        sto = c_s_n
        T_dim = self.Delta_T * T + self.T_ref
        return self.D_n_dimensional(sto, T_dim) / self.D_n_typ_dim

    def D_p(self, c_s_p, T):
        """Dimensionless positive particle diffusivity"""
        sto = c_s_p
        T_dim = self.Delta_T * T + self.T_ref
        return self.D_p_dimensional(sto, T_dim) / self.D_p_typ_dim

    def j0_n(self, c_e, c_s_surf, T):
        """Dimensionless negative exchange-current density"""
        c_e_dim = c_e * self.c_e_typ
        c_s_surf_dim = c_s_surf * self.c_n_max
        T_dim = self.Delta_T * T + self.T_ref

        return (
            self.j0_n_dimensional(c_e_dim, c_s_surf_dim, T_dim)
            / self.j0_n_ref_dimensional
        )

    def j0_p(self, c_e, c_s_surf, T):
        """Dimensionless positive exchange-current density"""
        c_e_dim = c_e * self.c_e_typ
        c_s_surf_dim = c_s_surf * self.c_p_max
        T_dim = self.Delta_T * T + self.T_ref

        return (
            self.j0_p_dimensional(c_e_dim, c_s_surf_dim, T_dim)
            / self.j0_p_ref_dimensional
        )

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

    def U_n(self, c_s_n, T):
        """Dimensionless open-circuit potential in the negative electrode"""
        sto = c_s_n
        T_dim = self.Delta_T * T + self.T_ref
        return (self.U_n_dimensional(sto, T_dim) - self.U_n_ref) / self.potential_scale

    def U_p(self, c_s_p, T):
        """Dimensionless open-circuit potential in the positive electrode"""
        sto = c_s_p
        T_dim = self.Delta_T * T + self.T_ref
        return (self.U_p_dimensional(sto, T_dim) - self.U_p_ref) / self.potential_scale

    def dUdT_n(self, c_s_n):
        """Dimensionless entropic change in negative open-circuit potential"""
        sto = c_s_n
        return self.dUdT_n_dimensional(sto) * self.Delta_T / self.potential_scale

    def dUdT_p(self, c_s_p):
        """Dimensionless entropic change in positive open-circuit potential"""
        sto = c_s_p
        return self.dUdT_p_dimensional(sto) * self.Delta_T / self.potential_scale

    def c_n_init(self, x):
        """
        Dimensionless initial concentration as a function of dimensionless position x.
        """
        return self.c_n_init_dimensional(x) / self.c_n_max

    def c_p_init(self, x):
        """
        Dimensionless initial concentration as a function of dimensionless position x.
        """
        return self.c_p_init_dimensional(x) / self.c_p_max

    def rho(self, T):
        """Dimensionless effective volumetric heat capacity"""
        return (
            self.rho_cn(T) * self.l_cn
            + self.rho_n(T) * self.l_n
            + self.rho_s(T) * self.l_s
            + self.rho_p(T) * self.l_p
            + self.rho_cp(T) * self.l_cp
        ) / self.l

    def t_n_change(self, sto):
        """
        Dimensionless volume change for the negative electrode;
        sto should be R-averaged
        """
        return pybamm.FunctionParameter(
            "Negative electrode volume change", {"Particle stoichiometry": sto}
        )

    def t_p_change(self, sto):
        """
        Dimensionless volume change for the positive electrode;
        sto should be R-averaged
        """
        return pybamm.FunctionParameter(
            "Positive electrode volume change", {"Particle stoichiometry": sto}
        )

    def k_cr_p(self, T):
        """
        Dimensionless cracking rate for the positive electrode;
        """
        T_dim = self.Delta_T * T + self.T_ref
        delta_k_cr = self.E_p ** self.m_cr_p * self.l_cr_p_0 ** (self.m_cr_p / 2 - 1)
        return (
            pybamm.FunctionParameter(
                "Positive electrode cracking rate", {"Temperature [K]": T_dim}
            )
            * delta_k_cr
        )

    def k_cr_n(self, T):
        """
        Dimensionless cracking rate for the negative electrode;
        """
        T_dim = self.Delta_T * T + self.T_ref
        delta_k_cr = self.E_n ** self.m_cr_n * self.l_cr_n_0 ** (self.m_cr_n / 2 - 1)
        return (
            pybamm.FunctionParameter(
                "Negative electrode cracking rate", {"Temperature [K]": T_dim}
            )
            * delta_k_cr
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

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, extra_options):
        self._options = pybamm.BatteryModelOptions(extra_options)
