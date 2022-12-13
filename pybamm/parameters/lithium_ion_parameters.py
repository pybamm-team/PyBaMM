#
# Standard parameters for lithium-ion battery models
#
import pybamm
from .base_parameters import BaseParameters, NullParameters


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

        # Get geometric, electrical and thermal parameters
        self.geo = pybamm.GeometricParameters(options)
        self.elec = pybamm.electrical_parameters
        self.therm = pybamm.thermal_parameters

        # Initialize domain parameters
        self.n = DomainLithiumIonParameters("negative", self)
        self.s = DomainLithiumIonParameters("separator", self)
        self.p = DomainLithiumIonParameters("positive", self)
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
        """Defines the dimensional parameters"""
        # Physical constants
        self.R = pybamm.constants.R
        self.F = pybamm.constants.F
        self.k_b = pybamm.constants.k_b
        self.q_e = pybamm.constants.q_e

        self.T_ref = self.therm.T_ref
        self.T_init_dim = self.therm.T_init_dim
        self.T_init = self.therm.T_init

        # Macroscale geometry
        self.L_x = self.geo.L_x
        self.L = self.geo.L
        self.L_y = self.geo.L_y
        self.L_z = self.geo.L_z
        self.r_inner_dimensional = self.geo.r_inner_dimensional
        self.r_outer_dimensional = self.geo.r_outer_dimensional
        self.A_cc = self.geo.A_cc
        self.A_cooling = self.geo.A_cooling
        self.V_cell = self.geo.V_cell

        # Electrical
        self.I_typ = self.elec.I_typ
        self.Q = self.elec.Q
        self.C_rate = self.elec.C_rate
        self.n_electrodes_parallel = self.elec.n_electrodes_parallel
        self.n_cells = self.elec.n_cells
        self.i_typ = self.elec.i_typ
        self.voltage_low_cut_dimensional = self.elec.voltage_low_cut_dimensional
        self.voltage_high_cut_dimensional = self.elec.voltage_high_cut_dimensional

        # Domain parameters
        for domain in self.domain_params.values():
            domain._set_dimensional_parameters()

        # Electrolyte properties
        self.c_e_typ = pybamm.Parameter("Typical electrolyte concentration [mol.m-3]")

        self.epsilon_init = pybamm.concatenation(
            *[
                self.domain_params[domain.split()[0]].epsilon_init
                for domain in self.options.whole_cell_domains
            ]
        )

        # Lithium plating parameters
        self.V_bar_plated_Li = pybamm.Parameter(
            "Lithium metal partial molar volume [m3.mol-1]"
        )
        self.c_Li_typ = pybamm.Parameter(
            "Typical plated lithium concentration [mol.m-3]"
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
        # Electrolyte
        c_e_av_init = pybamm.xyz_average(self.epsilon_init) * self.c_e_typ
        self.n_Li_e_init = c_e_av_init * self.L_x * self.A_cc

        self.n_Li_particles_init = self.n.n_Li_init + self.p.n_Li_init
        self.n_Li_init = self.n_Li_particles_init + self.n_Li_e_init
        self.Q_Li_particles_init = self.n_Li_particles_init * self.F / 3600
        self.Q_Li_init = self.n_Li_init * self.F / 3600

        # Reference OCP based on initial concentration
        self.ocv_ref = self.p.U_ref - self.n.U_ref
        self.ocv_init_dim = self.p.prim.U_init_dim - self.n.prim.U_init_dim

    def chi_dimensional(self, c_e, T):
        """
        Thermodynamic factor:
            (1-2*t_plus) is for Nernst-Planck,
            2*(1-t_plus) for Stefan-Maxwell,
        see Bizeray et al (2016) "Resolving a discrepancy ...".
        """
        return (2 * (1 - self.t_plus_dimensional(c_e, T))) * (
            self.one_plus_dlnf_dlnc_dimensional(c_e, T)
        )

    def t_plus_dimensional(self, c_e, T):
        """Cation transference number (dimensionless)"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        return pybamm.FunctionParameter("Cation transference number", inputs)

    def one_plus_dlnf_dlnc_dimensional(self, c_e, T):
        """Thermodynamic factor (dimensionless)"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        return pybamm.FunctionParameter("1 + dlnf/dlnc", inputs)

    def D_e_dimensional(self, c_e, T):
        """Dimensional diffusivity in electrolyte"""
        tol = pybamm.settings.tolerances["D_e__c_e"]
        c_e = pybamm.maximum(c_e, tol)
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        return pybamm.FunctionParameter("Electrolyte diffusivity [m2.s-1]", inputs)

    def kappa_e_dimensional(self, c_e, T):
        """Dimensional electrolyte conductivity"""
        tol = pybamm.settings.tolerances["D_e__c_e"]
        c_e = pybamm.maximum(c_e, tol)
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

    def dead_lithium_decay_rate_dimensional(self, L_sei):
        """Dimensional dead lithium decay rate [s-1]"""
        inputs = {"Total SEI thickness [m]": L_sei}
        return pybamm.FunctionParameter("Dead lithium decay rate [s-1]", inputs)

    def _set_scales(self):
        """Define the scales used in the non-dimensionalisation scheme"""
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
            self.c_max = self.p.prim.c_max
        else:
            self.c_max = self.n.prim.c_max
        self.tau_discharge = self.F * self.c_max * self.L_x / self.i_typ

        # Electrolyte diffusion timescale
        self.D_e_typ = self.D_e_dimensional(self.c_e_typ, self.T_ref)
        self.tau_diffusion_e = self.L_x**2 / self.D_e_typ

        # Thermal diffusion timescale
        self.tau_th_yz = self.therm.tau_th_yz

        # Choose discharge timescale
        if self.options["timescale"] == "default":
            self.timescale = self.tau_discharge
        else:
            self.timescale = pybamm.Scalar(self.options["timescale"])

        for domain in self.domain_params.values():
            domain._set_scales()

    def _set_dimensionless_parameters(self):
        """Defines the dimensionless parameters"""
        # Timescale ratios
        self.C_e = self.tau_diffusion_e / self.timescale
        self.C_th = self.tau_th_yz / self.timescale

        # Concentration ratios
        self.gamma_e = (self.tau_discharge / self.timescale) * self.c_e_typ / self.c_max

        # Macroscale Geometry
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

        for domain in self.domain_params.values():
            domain._set_dimensionless_parameters()

        # Electrolyte Properties
        self.beta_surf = pybamm.Scalar(0)

        # Electrical
        self.voltage_low_cut = (
            self.voltage_low_cut_dimensional - self.ocv_ref
        ) / self.potential_scale
        self.voltage_high_cut = (
            self.voltage_high_cut_dimensional - self.ocv_ref
        ) / self.potential_scale

        # Thermal
        self.Theta = self.therm.Theta
        self.T_amb_dim = self.therm.T_amb_dim
        self.T_amb = self.therm.T_amb

        self.h_edge = self.therm.h_edge
        self.h_total = self.therm.h_total
        self.rho = self.therm.rho

        self.B = (
            self.i_typ
            * self.R
            * self.T_ref
            * self.tau_th_yz
            / (self.therm.rho_eff_dim_ref * self.F * self.Delta_T * self.L_x)
        )

        # lithium plating parameters
        self.c_plated_Li_0 = self.c_plated_Li_0_dim / self.c_Li_typ

        self.alpha_plating = pybamm.Parameter("Lithium plating transfer coefficient")
        self.alpha_stripping = 1 - self.alpha_plating

        # ratio of lithium plating reaction scaled to intercalation reaction
        self.Gamma_plating = (
            self.n.prim.a_typ * self.n.prim.j_scale * self.timescale
        ) / (self.F * self.c_Li_typ)

        # Initial conditions
        self.c_e_init = self.c_e_init_dimensional / self.c_e_typ
        self.ocv_init = (self.ocv_init_dim - self.ocv_ref) / self.potential_scale

        # Dimensionless mechanical parameters
        self.t0_cr = 3600 / (self.C_rate * self.timescale)

    def chi(self, c_e, T):
        """
        Thermodynamic factor:
            (1-2*t_plus) is for Nernst-Planck,
            2*(1-t_plus) for Stefan-Maxwell,
        see Bizeray et al (2016) "Resolving a discrepancy ...".
        """
        c_e_dimensional = c_e * self.c_e_typ
        T_dim = self.Delta_T * T + self.T_ref
        return self.chi_dimensional(c_e_dimensional, T_dim)

    def chiRT_over_Fc(self, c_e, T):
        """
        chi * (1 + Theta * T) / c,
        as it appears in the electrolyte potential equation
        """
        tol = pybamm.settings.tolerances["chi__c_e"]
        c_e = pybamm.maximum(c_e, tol)
        return self.chi(c_e, T) * (1 + self.Theta * T) / c_e

    def t_plus(self, c_e, T):
        """Cation transference number (dimensionless)"""
        c_e_dimensional = c_e * self.c_e_typ
        T_dim = self.Delta_T * T + self.T_ref
        return self.t_plus_dimensional(c_e_dimensional, T_dim)

    def D_e(self, c_e, T):
        """Dimensionless electrolyte diffusivity"""
        c_e_dimensional = c_e * self.c_e_typ
        T_dim = self.Delta_T * T + self.T_ref
        return self.D_e_dimensional(c_e_dimensional, T_dim) / self.D_e_typ

    def kappa_e(self, c_e, T):
        """Dimensionless electrolyte conductivity"""
        c_e_dimensional = c_e * self.c_e_typ
        kappa_scale = self.F**2 * self.D_e_typ * self.c_e_typ / (self.R * self.T_ref)
        T_dim = self.Delta_T * T + self.T_ref
        return self.kappa_e_dimensional(c_e_dimensional, T_dim) / kappa_scale

    def j0_stripping(self, c_e, c_Li, T):
        """Dimensionless exchange-current density for stripping"""
        c_e_dim = c_e * self.c_e_typ
        c_Li_dim = c_Li * self.c_Li_typ
        T_dim = self.Delta_T * T + self.T_ref

        return (
            self.j0_stripping_dimensional(c_e_dim, c_Li_dim, T_dim)
            / self.n.prim.j_scale
        )

    def j0_plating(self, c_e, c_Li, T):
        """Dimensionless reverse plating current"""
        c_e_dim = c_e * self.c_e_typ
        c_Li_dim = c_Li * self.c_Li_typ
        T_dim = self.Delta_T * T + self.T_ref

        return (
            self.j0_plating_dimensional(c_e_dim, c_Li_dim, T_dim) / self.n.prim.j_scale
        )

    def dead_lithium_decay_rate(self, L_sei):
        """Dimensionless exchange-current density for stripping"""
        L_sei_dim = L_sei * self.n.prim.L_sei_0_dim

        return self.dead_lithium_decay_rate_dimensional(L_sei_dim) * self.timescale

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


class DomainLithiumIonParameters(BaseParameters):
    def __init__(self, domain, main_param):
        self.domain = domain
        self.main_param = main_param

        self.geo = getattr(main_param.geo, domain[0])
        self.therm = getattr(main_param.therm, domain[0])

        if domain != "separator":
            self.prim = ParticleLithiumIonParameters("primary", self)
            phases_option = int(getattr(main_param.options, domain)["particle phases"])
            if phases_option >= 2:
                self.sec = ParticleLithiumIonParameters("secondary", self)
            else:
                self.sec = NullParameters()
        else:
            self.prim = NullParameters()
            self.sec = NullParameters()

        self.phase_params = {"primary": self.prim, "secondary": self.sec}

    def _set_dimensional_parameters(self):
        main = self.main_param
        domain, Domain = self.domain_Domain

        if domain == "separator":
            x = pybamm.standard_spatial_vars.x_s * main.L_x
            self.epsilon_init = pybamm.FunctionParameter(
                "Separator porosity", {"Through-cell distance (x) [m]": x}
            )
            self.epsilon_inactive = 1 - self.epsilon_init
            self.b_e = self.geo.b_e
            self.L = self.geo.L
            return

        x = (
            pybamm.SpatialVariable(
                f"x_{domain[0]}",
                domain=[f"{domain} electrode"],
                auxiliary_domains={"secondary": "current collector"},
                coord_sys="cartesian",
            )
            * main.L_x
        )

        # Macroscale geometry
        self.L_cc = self.geo.L_cc
        self.L = self.geo.L

        for phase in self.phase_params.values():
            phase._set_dimensional_parameters()

        # Tab geometry (for pouch cells)
        self.L_tab = self.geo.L_tab
        self.Centre_y_tab = self.geo.Centre_y_tab
        self.Centre_z_tab = self.geo.Centre_z_tab
        self.A_tab = self.geo.A_tab

        # Particle properties
        self.sigma_cc_dimensional = pybamm.Parameter(
            f"{Domain} current collector conductivity [S.m-1]"
        )
        if main.options.electrode_types[domain] == "porous":
            self.epsilon_init = pybamm.FunctionParameter(
                f"{Domain} electrode porosity", {"Through-cell distance (x) [m]": x}
            )
            epsilon_s_tot = sum(phase.epsilon_s for phase in self.phase_params.values())
            self.epsilon_inactive = 1 - self.epsilon_init - epsilon_s_tot

            self.Q_init = sum(phase.Q_init for phase in self.phase_params.values())
            # Use primary phase to set the reference potential
            self.U_ref = self.prim.U_dimensional(self.prim.c_init_av, main.T_ref)
        else:
            self.U_ref = pybamm.Scalar(0)

        self.n_Li_init = sum(phase.n_Li_init for phase in self.phase_params.values())
        self.Q_Li_init = sum(phase.Q_Li_init for phase in self.phase_params.values())

        # Tortuosity parameters
        self.b_e = self.geo.b_e
        self.b_s = self.geo.b_s

        self.C_dl_dimensional = pybamm.Parameter(
            f"{Domain} electrode double-layer capacity [F.m-2]"
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
        """Dimensional electrical conductivity in electrode"""
        inputs = {"Temperature [K]": T}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} electrode conductivity [S.m-1]", inputs
        )

    def _set_scales(self):
        """Define the scales used in the non-dimensionalisation scheme"""
        for phase in self.phase_params.values():
            phase._set_scales()

        if self.domain == "separator":
            return

    def _set_dimensionless_parameters(self):
        for phase in self.phase_params.values():
            phase._set_dimensionless_parameters()

        main = self.main_param

        if self.domain == "separator":
            self.l = self.geo.l
            self.rho = self.therm.rho
            self.lambda_ = self.therm.lambda_
            return

        # Macroscale Geometry
        self.l_cc = self.geo.l_cc
        self.l = self.geo.l

        # Thermal
        self.rho_cc = self.therm.rho_cc
        self.rho = self.therm.rho
        self.lambda_cc = self.therm.lambda_cc
        self.lambda_ = self.therm.lambda_
        self.h_tab = self.therm.h_tab
        self.h_cc = self.therm.h_cc

        # Tab geometry (for pouch cells)
        self.l_tab = self.geo.l_tab
        self.centre_y_tab = self.geo.centre_y_tab
        self.centre_z_tab = self.geo.centre_z_tab

        # Electrochemical Reactions
        self.C_dl = (
            self.C_dl_dimensional
            * main.potential_scale
            / self.prim.j_scale
            / main.timescale
        )
        # Electrode Properties
        self.sigma_cc = (
            self.sigma_cc_dimensional * main.potential_scale / main.i_typ / main.L_x
        )
        self.sigma_cc_prime = self.sigma_cc * main.delta**2
        self.sigma_cc_dbl_prime = self.sigma_cc_prime * main.delta

        # Electrolyte Properties
        self.beta_surf = pybamm.Scalar(0)

        # Utilisation factors
        self.beta_utilisation = (
            self.beta_utilisation_dimensional
            * self.prim.a_typ
            * self.prim.j_scale
            * main.timescale
        ) / main.F

        if main.options.electrode_types[self.domain] == "planar":
            return

        # Dimensionless mechanical parameters
        self.rho_cr = self.rho_cr_dim * self.l_cr_0 * self.w_cr
        self.theta = self.theta_dim * self.prim.c_max / main.T_ref
        self.c_0 = self.c_0_dim / self.prim.c_max
        self.beta_LAM = self.beta_LAM_dimensional * main.timescale
        # normalised typical time for one cycle
        self.stress_critical = self.stress_critical_dim / self.E
        # Reaction-driven LAM parameters
        self.beta_LAM_sei = (
            self.beta_LAM_sei_dimensional
            * self.prim.a_typ
            * self.prim.j_scale
            * main.timescale
        ) / main.F

    def sigma(self, T):
        """Dimensionless electrode electrical conductivity"""
        main = self.main_param
        T_dim = self.main_param.Delta_T * T + self.main_param.T_ref
        return (
            self.sigma_dimensional(T_dim) * main.potential_scale / main.i_typ / main.L_x
        )

    def sigma_prime(self, T):
        """Rescaled dimensionless electrode electrical conductivity"""
        return self.sigma(T) * self.main_param.delta

    def k_cr(self, T):
        """
        Dimensionless cracking rate for the electrode;
        """
        Domain = self.domain.capitalize()
        T_dim = self.main_param.Delta_T * T + self.main_param.T_ref
        delta_k_cr = self.E**self.m_cr * self.l_cr_0 ** (self.m_cr / 2 - 1)
        return (
            pybamm.FunctionParameter(
                f"{Domain} electrode cracking rate", {"Temperature [K]": T_dim}
            )
            * delta_k_cr
        )


class ParticleLithiumIonParameters(BaseParameters):
    def __init__(self, phase, domain_param):
        self.domain_param = domain_param
        self.domain = domain_param.domain
        self.main_param = domain_param.main_param
        self.phase = phase
        self.set_phase_name()
        if self.phase == "primary":
            self.geo = domain_param.geo.prim
        elif self.phase == "secondary":
            self.geo = domain_param.geo.sec

    def _set_dimensional_parameters(self):
        main = self.main_param
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        pref = self.phase_prefactor

        # Electrochemical reactions
        self.ne = pybamm.Parameter(f"{pref}{Domain} electrode electrons in reaction")

        # Intercalation kinetics
        self.mhc_lambda_dimensional = pybamm.Parameter(
            f"{pref}{Domain} electrode reorganization energy [eV]"
        )
        self.alpha_bv = pybamm.Parameter(
            f"{pref}{Domain} electrode Butler-Volmer transfer coefficient"
        )

        if self.domain == "negative":
            # SEI parameters
            self.V_bar_inner_dimensional = pybamm.Parameter(
                f"{pref}Inner SEI partial molar volume [m3.mol-1]"
            )
            self.V_bar_outer_dimensional = pybamm.Parameter(
                f"{pref}Outer SEI partial molar volume [m3.mol-1]"
            )

            self.m_sei_dimensional = pybamm.Parameter(
                f"{pref}SEI reaction exchange current density [A.m-2]"
            )

            self.R_sei_dimensional = pybamm.Parameter(f"{pref}SEI resistivity [Ohm.m]")
            self.D_sol_dimensional = pybamm.Parameter(
                f"{pref}Outer SEI solvent diffusivity [m2.s-1]"
            )
            self.c_sol_dimensional = pybamm.Parameter(
                f"{pref}Bulk solvent concentration [mol.m-3]"
            )
            self.U_inner_dimensional = pybamm.Parameter(
                f"{pref}Inner SEI open-circuit potential [V]"
            )
            self.U_outer_dimensional = pybamm.Parameter(
                f"{pref}Outer SEI open-circuit potential [V]"
            )
            self.kappa_inner_dimensional = pybamm.Parameter(
                f"{pref}Inner SEI electron conductivity [S.m-1]"
            )
            self.D_li_dimensional = pybamm.Parameter(
                f"{pref}Inner SEI lithium interstitial diffusivity [m2.s-1]"
            )
            self.c_li_0_dimensional = pybamm.Parameter(
                f"{pref}Lithium interstitial reference concentration [mol.m-3]"
            )
            self.L_inner_0_dim = pybamm.Parameter(
                f"{pref}Initial inner SEI thickness [m]"
            )
            self.L_outer_0_dim = pybamm.Parameter(
                f"{pref}Initial outer SEI thickness [m]"
            )
            self.L_sei_0_dim = self.L_inner_0_dim + self.L_outer_0_dim
            self.E_sei_dimensional = pybamm.Parameter(
                f"{pref}SEI growth activation energy [J.mol-1]"
            )
            self.alpha_SEI = pybamm.Parameter(f"{pref}SEI growth transfer coefficient")

            # EC reaction
            self.c_ec_0_dim = pybamm.Parameter(
                f"{pref}EC initial concentration in electrolyte [mol.m-3]"
            )
            self.D_ec_dim = pybamm.Parameter(f"{pref}EC diffusivity [m2.s-1]")
            self.k_sei_dim = pybamm.Parameter(
                f"{pref}SEI kinetic rate constant [m.s-1]"
            )
            self.U_sei_dim = pybamm.Parameter(f"{pref}SEI open-circuit potential [V]")

        if main.options.electrode_types[domain] == "planar":
            self.n_Li_init = pybamm.Scalar(0)
            self.Q_Li_init = pybamm.Scalar(0)
            self.U_init_dim = pybamm.Scalar(0)
            return

        x = (
            pybamm.SpatialVariable(
                f"x_{domain[0]}",
                domain=[f"{domain} electrode"],
                auxiliary_domains={"secondary": "current collector"},
                coord_sys="cartesian",
            )
            * main.L_x
        )
        r = (
            pybamm.SpatialVariable(
                f"r_{domain[0]}",
                domain=[f"{domain} {self.phase_name}particle"],
                auxiliary_domains={
                    "secondary": f"{domain} electrode",
                    "tertiary": "current collector",
                },
                coord_sys="spherical polar",
            )
            * self.geo.R_typ
        )

        # Macroscale geometry
        # Note: the surface area to volume ratio is defined later with the function
        # parameters. The particle size as a function of through-cell position is
        # already defined in geometric_parameters.py
        self.R_dimensional = self.geo.R_dimensional

        # Particle properties
        self.c_max = pybamm.Parameter(
            f"{pref}Maximum concentration in {domain} electrode [mol.m-3]"
        )

        # Particle-size distribution parameters
        self.R_min_dim = self.geo.R_min_dim
        self.R_max_dim = self.geo.R_max_dim
        self.sd_a_dim = self.geo.sd_a_dim
        self.f_a_dist_dimensional = self.geo.f_a_dist_dimensional

        self.epsilon_s = pybamm.FunctionParameter(
            f"{pref}{Domain} electrode active material volume fraction",
            {"Through-cell distance (x) [m]": x},
        )
        self.c_init_dimensional = pybamm.FunctionParameter(
            f"{pref}Initial concentration in {domain} electrode [mol.m-3]",
            {
                "Radial distance (r) [m]": r,
                "Through-cell distance (x) [m]": pybamm.PrimaryBroadcast(
                    x, f"{domain} {phase_name}particle"
                ),
            },
        )
        self.c_init = self.c_init_dimensional / self.c_max
        self.c_init_av = pybamm.xyz_average(pybamm.r_average(self.c_init))
        eps_c_init_av = pybamm.xyz_average(
            self.epsilon_s * pybamm.r_average(self.c_init)
        )
        self.n_Li_init = eps_c_init_av * self.c_max * self.domain_param.L * main.A_cc
        self.Q_Li_init = self.n_Li_init * main.F / 3600

        eps_s_av = pybamm.xyz_average(self.epsilon_s)
        self.elec_loading = eps_s_av * self.domain_param.L * self.c_max * main.F / 3600
        self.Q_init = self.elec_loading * main.A_cc

        self.U_init_dim = self.U_dimensional(self.c_init_av, main.T_init_dim)

    def D_dimensional(self, sto, T):
        """Dimensional diffusivity in particle. Note this is defined as a
        function of stochiometry"""
        Domain = self.domain.capitalize()
        inputs = {
            f"{self.phase_prefactor}{Domain} particle stoichiometry": sto,
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            f"{self.phase_prefactor}{Domain} electrode diffusivity [m2.s-1]",
            inputs,
        )

    def j0_dimensional(self, c_e, c_s_surf, T):
        """Dimensional exchange-current density [A.m-2]"""
        domain, Domain = self.domain_Domain
        inputs = {
            "Electrolyte concentration [mol.m-3]": c_e,
            f"{Domain} particle surface concentration [mol.m-3]": c_s_surf,
            f"{self.phase_prefactor}Maximum {domain} particle "
            "surface concentration [mol.m-3]": self.c_max,
            "Temperature [K]": T,
            f"{self.phase_prefactor}Maximum {domain} particle "
            "surface concentration [mol.m-3]": self.c_max,
        }
        return pybamm.FunctionParameter(
            f"{self.phase_prefactor}{Domain} electrode "
            "exchange-current density [A.m-2]",
            inputs,
        )

    def U_dimensional(self, sto, T, lithiation=None):
        """Dimensional open-circuit potential [V]"""
        # bound stoichiometry between tol and 1-tol. Adding 1/sto + 1/(sto-1) later
        # will ensure that ocp goes to +- infinity if sto goes into that region
        # anyway
        Domain = self.domain.capitalize()
        tol = pybamm.settings.tolerances["U__c_s"]
        sto = pybamm.maximum(pybamm.minimum(sto, 1 - tol), tol)
        if lithiation is None:
            lithiation = ""
        else:
            lithiation = lithiation + " "
        inputs = {f"{self.phase_prefactor}{Domain} particle stoichiometry": sto}
        u_ref = pybamm.FunctionParameter(
            f"{self.phase_prefactor}{Domain} electrode {lithiation}OCP [V]", inputs
        )
        # add a term to ensure that the OCP goes to infinity at 0 and -infinity at 1
        # this will not affect the OCP for most values of sto
        # see #1435
        u_ref = u_ref + 1e-6 * (1 / sto + 1 / (sto - 1))
        dudt_dim_func = self.dUdT_dimensional(sto)
        d = self.domain[0]
        dudt_dim_func.print_name = r"\frac{dU_{" + d + r"}}{dT}"
        return u_ref + (T - self.main_param.T_ref) * dudt_dim_func

    def dUdT_dimensional(self, sto):
        """
        Dimensional entropic change of the open-circuit potential [V.K-1]
        """
        domain, Domain = self.domain_Domain
        inputs = {
            f"{Domain} particle stoichiometry": sto,
            f"{self.phase_prefactor}Maximum {domain} particle "
            "surface concentration [mol.m-3]": self.c_max,
        }
        return pybamm.FunctionParameter(
            f"{self.phase_prefactor}{Domain} electrode OCP entropic change [V.K-1]",
            inputs,
        )

    def _set_scales(self):
        """Define the scales used in the non-dimensionalisation scheme"""
        domain = self.domain
        main = self.main_param

        # Scale for interfacial current density in A/m2
        if main.options.electrode_types[domain] == "planar":
            # planar electrode (boundary condition between negative and separator)
            self.a_typ = 1
            self.j_scale = main.i_typ
            return

        # Microscale
        self.R_typ = self.geo.R_typ

        if main.options["particle shape"] == "spherical":
            self.a_typ = 3 * pybamm.xyz_average(self.epsilon_s) / self.R_typ

        # porous electrode
        self.j_scale = main.i_typ / (self.a_typ * main.L_x)

        # Concentration
        self.particle_concentration_scale = self.c_max

        # Reference exchange-current density
        self.j0_ref_dimensional = (
            self.j0_dimensional(main.c_e_typ, self.c_max / 2, main.T_ref) * 2
        )

        # Reaction timescales
        self.tau_r = main.F * self.c_max / (self.j0_ref_dimensional * self.a_typ)
        # Particle diffusion timescales
        self.D_typ_dim = self.D_dimensional(pybamm.Scalar(1), main.T_ref)
        self.tau_diffusion = self.R_typ**2 / self.D_typ_dim

    def _set_dimensionless_parameters(self):
        main = self.main_param
        domain_param = self.domain_param
        pref = self.phase_prefactor

        # Intercalation kinetics
        self.mhc_lambda = self.mhc_lambda_dimensional / main.potential_scale_eV

        if self.domain == "negative":
            # SEI parameters
            self.inner_sei_proportion = pybamm.Parameter(
                f"{pref}Inner SEI reaction proportion"
            )

            self.z_sei = pybamm.Parameter(f"{pref}Ratio of lithium moles to SEI moles")

            self.E_over_RT_sei = self.E_sei_dimensional / main.R / main.T_ref

            self.C_sei_reaction = (self.j_scale / self.m_sei_dimensional) * pybamm.exp(
                -(main.F * domain_param.U_ref / (2 * main.R * main.T_ref))
            )

            self.C_sei_solvent = (
                self.j_scale
                * self.L_sei_0_dim
                / (self.c_sol_dimensional * main.F * self.D_sol_dimensional)
            )

            self.C_sei_electron = (
                self.j_scale
                * main.F
                * self.L_sei_0_dim
                / (self.kappa_inner_dimensional * main.R * main.T_ref)
            )

            self.C_sei_inter = (
                self.j_scale
                * self.L_sei_0_dim
                / (self.D_li_dimensional * self.c_li_0_dimensional * main.F)
            )

            self.U_inner_electron = (
                main.F * self.U_inner_dimensional / main.R / main.T_ref
            )

            self.R_sei = (
                main.F
                * self.j_scale
                * self.R_sei_dimensional
                * self.L_sei_0_dim
                / main.R
                / main.T_ref
            )

            self.v_bar = self.V_bar_outer_dimensional / self.V_bar_inner_dimensional
            self.c_sei_scale = (
                self.L_sei_0_dim * self.a_typ / self.V_bar_inner_dimensional
            )
            self.c_sei_outer_scale = (
                self.L_sei_0_dim * self.a_typ / self.V_bar_outer_dimensional
            )

            self.L_inner_0 = self.L_inner_0_dim / self.L_sei_0_dim
            self.L_outer_0 = self.L_outer_0_dim / self.L_sei_0_dim

            # Dividing by 10000 makes initial condition effectively zero
            # without triggering division by zero errors
            self.L_inner_crack_0 = self.L_inner_0 / 10000
            self.L_outer_crack_0 = self.L_outer_0 / 10000

            # ratio of SEI reaction scale to intercalation reaction
            self.Gamma_SEI = (
                self.V_bar_inner_dimensional * self.j_scale * main.timescale
            ) / (main.F * self.z_sei * self.L_sei_0_dim)

            # EC reaction
            self.C_ec = (
                self.L_sei_0_dim
                * self.j_scale
                / (main.F * self.c_ec_0_dim * self.D_ec_dim)
            )
            self.C_sei_ec = (
                main.F
                * self.k_sei_dim
                * self.c_ec_0_dim
                / self.j_scale
                * (
                    pybamm.exp(
                        -(
                            main.F
                            * (domain_param.U_ref - self.U_sei_dim)
                            / (2 * main.R * main.T_ref)
                        )
                    )
                )
            )
            self.c_sei_init = self.c_ec_0_dim / self.c_sei_outer_scale

        # Initial conditions
        if main.options.electrode_types[self.domain] == "planar":
            self.U_init = pybamm.Scalar(0)
            return
        else:
            self.U_init = (
                self.U_init_dim - self.domain_param.U_ref
            ) / main.potential_scale

        # Timescale ratios
        self.C_diff = self.tau_diffusion / main.timescale
        self.C_r = self.tau_r / main.timescale

        # Microscale geometry
        self.R = self.geo.R
        self.a_R = self.a_typ * self.R_typ

        # Particle-size distribution geometry
        self.R_min = self.geo.R_min
        self.R_max = self.geo.R_max
        self.sd_a = self.geo.sd_a
        self.f_a_dist = self.geo.f_a_dist

        # Concentration ratios
        # In most cases gamma_n will be equal to 1
        self.gamma = (main.tau_discharge / main.timescale) * self.c_max / main.c_max

        # Electrolyte Properties
        self.beta_surf = pybamm.Scalar(0)

    def D(self, c_s, T):
        """Dimensionless particle diffusivity"""
        sto = c_s
        T_dim = self.main_param.Delta_T * T + self.main_param.T_ref
        return self.D_dimensional(sto, T_dim) / self.D_typ_dim

    def j0(self, c_e, c_s_surf, T):
        """Dimensionless exchange-current density"""
        tol = pybamm.settings.tolerances["j0__c_e"]
        c_e = pybamm.maximum(c_e, tol)
        tol = pybamm.settings.tolerances["j0__c_s"]
        c_s_surf = pybamm.maximum(pybamm.minimum(c_s_surf, 1 - tol), tol)
        c_e_dim = c_e * self.main_param.c_e_typ
        c_s_surf_dim = c_s_surf * self.c_max
        T_dim = self.main_param.Delta_T * T + self.main_param.T_ref

        return self.j0_dimensional(c_e_dim, c_s_surf_dim, T_dim) / self.j_scale

    def U(self, c_s, T, lithiation=None):
        """Dimensionless open-circuit potential in the electrode"""
        main = self.main_param
        sto = c_s
        T_dim = self.main_param.Delta_T * T + self.main_param.T_ref
        return (
            self.U_dimensional(sto, T_dim, lithiation) - self.domain_param.U_ref
        ) / main.potential_scale

    def dUdT(self, c_s):
        """Dimensionless entropic change in open-circuit potential"""
        main = self.main_param
        sto = c_s
        return self.dUdT_dimensional(sto) * main.Delta_T / main.potential_scale

    def t_change(self, sto):
        """
        Dimensionless volume change for the electrode;
        sto should be R-averaged
        """
        domain, Domain = self.domain_Domain
        return pybamm.FunctionParameter(
            f"{Domain} electrode volume change",
            {
                "Particle stoichiometry": sto,
                f"{self.phase_prefactor}Maximum {domain} particle "
                "surface concentration [mol.m-3]": self.c_max,
            },
        )
