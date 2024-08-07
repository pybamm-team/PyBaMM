#
# Standard parameters for lithium-ion battery models
#
import pybamm
from .base_parameters import BaseParameters, NullParameters


class LithiumIonParameters(BaseParameters):
    """
    Standard parameters for lithium-ion battery models

    Parameters
    ----------

    options : dict, optional
        A dictionary of options to be passed to the parameters, see
        :class:`pybamm.BatteryModelOptions`.
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

        # Set parameters
        self._set_parameters()

    def _set_parameters(self):
        """Defines the dimensional parameters"""
        # Physical constants
        self.R = pybamm.Parameter("Ideal gas constant [J.K-1.mol-1]")
        self.F = pybamm.Parameter("Faraday constant [C.mol-1]")
        self.k_b = pybamm.Parameter("Boltzmann constant [J.K-1]")
        self.q_e = pybamm.Parameter("Elementary charge [C]")

        # Thermal parameters
        self.T_ref = self.therm.T_ref
        self.T_init = self.therm.T_init
        self.T_amb = self.therm.T_amb
        self.T_amb_av = self.therm.T_amb_av
        self.h_edge = self.therm.h_edge
        self.h_total = self.therm.h_total
        self.rho_c_p_eff = self.therm.rho_c_p_eff
        self.lambda_eff = self.therm.lambda_eff

        # Macroscale geometry
        self.L_x = self.geo.L_x
        self.L = self.geo.L
        self.L_y = self.geo.L_y
        self.L_z = self.geo.L_z
        self.r_inner = self.geo.r_inner
        self.r_outer = self.geo.r_outer
        self.A_cc = self.geo.A_cc
        self.A_cooling = self.geo.A_cooling
        self.V_cell = self.geo.V_cell

        # Electrical
        self.current_with_time = self.elec.current_with_time
        self.current_density_with_time = self.elec.current_density_with_time
        self.Q = self.elec.Q
        self.R_contact = self.elec.R_contact
        self.n_electrodes_parallel = self.elec.n_electrodes_parallel
        self.n_cells = self.elec.n_cells
        self.voltage_low_cut = self.elec.voltage_low_cut
        self.voltage_high_cut = self.elec.voltage_high_cut
        self.ocp_soc_0 = self.elec.ocp_soc_0
        self.ocp_soc_100 = self.elec.ocp_soc_100

        # Domain parameters
        for domain in self.domain_params.values():
            domain._set_parameters()

        # Electrolyte properties
        self.epsilon_init = pybamm.concatenation(
            *[
                self.domain_params[domain.split()[0]].epsilon_init
                for domain in self.options.whole_cell_domains
            ]
        )

        # Required by lithium plating and lithium metal plating reactions
        self.V_bar_Li = pybamm.Parameter(
            "Lithium metal partial molar volume [m3.mol-1]"
        )

        # Initial conditions
        # Note: the initial concentration in the electrodes can be set as a function
        # of through-cell position, so is defined later as a function
        self.c_e_init = pybamm.Parameter(
            "Initial concentration in electrolyte [mol.m-3]"
        )
        self.c_e_init_av = pybamm.xyz_average(self.c_e_init)
        self.c_e_init_av.print_name = "c_e_init"

        self.alpha_T_cell = pybamm.Parameter(
            "Cell thermal expansion coefficient [m.K-1]"
        )

        # Total lithium
        # Electrolyte
        c_e_av_init = pybamm.xyz_average(self.epsilon_init) * self.c_e_init
        self.n_Li_e_init = c_e_av_init * self.L_x * self.A_cc

        self.n_Li_particles_init = self.n.n_Li_init + self.p.n_Li_init
        self.n_Li_init = self.n_Li_particles_init + self.n_Li_e_init
        self.Q_Li_particles_init = self.n_Li_particles_init * self.F / 3600
        self.Q_Li_init = self.n_Li_init * self.F / 3600

        # Reference OCP based on initial concentration
        self.ocv_init = self.p.prim.U_init - self.n.prim.U_init

        # Some scales
        self.thermal_voltage = self.R * self.T_ref / self.F
        self.I_typ = self.Q / (self.A_cc * self.n_electrodes_parallel)
        self.a_j_scale = self.I_typ / self.L_x

    def chi(self, c_e, T):
        """
        Thermodynamic factor:
            (1-2*t_plus) is for Nernst-Planck,
            2*(1-t_plus) for Stefan-Maxwell,
        see Bizeray et al (2016) "Resolving a discrepancy ...".
        """
        return (2 * (1 - self.t_plus(c_e, T))) * (self.thermodynamic_factor(c_e, T))

    def chiRT_over_Fc(self, c_e, T):
        """
        chi * (1 + Theta * T) / c,
        as it appears in the electrolyte potential equation
        """
        tol = pybamm.settings.tolerances["chi__c_e"]
        c_e = pybamm.maximum(c_e, tol)
        return (self.R * T / self.F) * self.chi(c_e, T) / c_e

    def t_plus(self, c_e, T):
        """Cation transference number"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        return pybamm.FunctionParameter("Cation transference number", inputs)

    def thermodynamic_factor(self, c_e, T):
        """Thermodynamic factor"""
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        return pybamm.FunctionParameter("Thermodynamic factor", inputs)

    def D_e(self, c_e, T):
        """Dimensional diffusivity in electrolyte"""
        tol = pybamm.settings.tolerances["D_e__c_e"]
        c_e = pybamm.maximum(c_e, tol)
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        return pybamm.FunctionParameter("Electrolyte diffusivity [m2.s-1]", inputs)

    def kappa_e(self, c_e, T):
        """Dimensional electrolyte conductivity"""
        tol = pybamm.settings.tolerances["kappa_e__c_e"]
        c_e = pybamm.maximum(c_e, tol)
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        return pybamm.FunctionParameter("Electrolyte conductivity [S.m-1]", inputs)

    def j0_Li_metal(self, c_e, c_Li, T):
        """Dimensional exchange-current density for lithium metal electrode [A.m-2]"""
        inputs = {
            "Electrolyte concentration [mol.m-3]": c_e,
            "Lithium metal concentration [mol.m-3]": c_Li,
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Exchange-current density for lithium metal electrode [A.m-2]", inputs
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

    def _set_parameters(self):
        main = self.main_param
        domain, Domain = self.domain_Domain

        # Parameters that appear in the separator
        self.b_e = self.geo.b_e
        self.tau_e = self.geo.tau_e
        self.L = self.geo.L

        # Thermal
        self.rho_c_p = self.therm.rho_c_p
        self.lambda_ = self.therm.lambda_
        self.h_cc = self.therm.h_cc
        self.h_tab = self.therm.h_tab

        if domain == "separator":
            x = pybamm.standard_spatial_vars.x_s
            self.epsilon_init = pybamm.FunctionParameter(
                "Separator porosity", {"Through-cell distance (x) [m]": x}
            )
            self.epsilon_inactive = 1 - self.epsilon_init
            return

        self.rho_c_p_cc = self.therm.rho_c_p_cc
        self.lambda_cc = self.therm.lambda_cc

        x = pybamm.SpatialVariable(
            f"x_{domain[0]}",
            domain=[f"{domain} electrode"],
            auxiliary_domains={"secondary": "current collector"},
            coord_sys="cartesian",
        )

        # Macroscale geometry
        self.L_cc = self.geo.L_cc

        for phase in self.phase_params.values():
            phase._set_parameters()

        # Tab geometry (for pouch cells)
        self.L_tab = self.geo.L_tab
        self.centre_y_tab = self.geo.centre_y_tab
        self.centre_z_tab = self.geo.centre_z_tab
        self.A_tab = self.geo.A_tab

        # Particle properties
        self.sigma_cc = pybamm.Parameter(
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

        self.n_Li_init = sum(phase.n_Li_init for phase in self.phase_params.values())
        self.Q_Li_init = sum(phase.Q_Li_init for phase in self.phase_params.values())

        # Tortuosity parameters
        self.b_s = self.geo.b_s
        self.tau_s = self.geo.tau_s

        # Mechanical parameters
        self.nu = pybamm.Parameter(f"{Domain} electrode Poisson's ratio")
        self.c_0 = pybamm.Parameter(
            f"{Domain} electrode reference concentration for free of deformation "
            "[mol.m-3]"
        )

        self.l_cr_0 = pybamm.Parameter(f"{Domain} electrode initial crack length [m]")
        self.w_cr = pybamm.Parameter(f"{Domain} electrode initial crack width [m]")
        self.rho_cr = pybamm.Parameter(
            f"{Domain} electrode number of cracks per unit area [m-2]"
        )
        self.b_cr = pybamm.Parameter(f"{Domain} electrode Paris' law constant b")
        self.m_cr = pybamm.Parameter(f"{Domain} electrode Paris' law constant m")

        # Loss of active material parameters
        self.m_LAM = pybamm.Parameter(
            f"{Domain} electrode LAM constant exponential term"
        )
        self.beta_LAM = pybamm.Parameter(
            f"{Domain} electrode LAM constant proportional term [s-1]"
        )
        self.stress_critical = pybamm.Parameter(
            f"{Domain} electrode critical stress [Pa]"
        )
        self.beta_LAM_sei = pybamm.Parameter(
            f"{Domain} electrode reaction-driven LAM factor [m3.mol-1]"
        )

        # Utilisation parameters
        self.u_init = pybamm.Parameter(
            f"Initial {domain} electrode interface utilisation"
        )
        self.beta_utilisation = pybamm.Parameter(
            f"{Domain} electrode current-driven interface utilisation factor [m3.mol-1]"
        )

    def C_dl(self, T):
        """Dimensional double-layer capacity [F.m-2]"""
        inputs = {"Temperature [K]": T}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} electrode double-layer capacity [F.m-2]", inputs
        )

    def Omega(self, sto, T):
        """Dimensional partial molar volume of Li in solid solution [m3.mol-1]"""
        Domain = self.domain.capitalize()
        inputs = {f"{Domain} particle stoichiometry": sto, "Temperature [K]": T}
        return pybamm.FunctionParameter(
            f"{Domain} electrode partial molar volume [m3.mol-1]", inputs
        )

    def E(self, sto, T):
        """Dimensional Young's modulus"""
        Domain = self.domain.capitalize()
        inputs = {f"{Domain} particle stoichiometry": sto, "Temperature [K]": T}
        return pybamm.FunctionParameter(
            f"{Domain} electrode Young's modulus [Pa]", inputs
        )

    def sigma(self, T):
        """Dimensional electrical conductivity in electrode"""
        inputs = {"Temperature [K]": T}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} electrode conductivity [S.m-1]", inputs
        )

    def k_cr(self, T):
        """
        Cracking rate for the electrode;
        """
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} electrode cracking rate", {"Temperature [K]": T}
        )

    def LAM_rate_current(self, i, T):
        """
        Dimensional rate of loss of active material as a function of applied current
        density
        """
        Domain = self.domain.capitalize()
        inputs = {"Total current density [A.m-2]": i, "Temperature [K]": T}
        return pybamm.FunctionParameter(
            f"{Domain} electrode current-driven LAM rate", inputs
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
        self.options = getattr(self.main_param.options, self.domain)

    def _set_parameters(self):
        main = self.main_param
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        pref = self.phase_prefactor

        # Electrochemical reactions
        self.ne = pybamm.Scalar(1)

        # Intercalation kinetics
        self.mhc_lambda = pybamm.Parameter(
            f"{pref}{Domain} electrode reorganization energy [eV]"
        )
        self.alpha_bv = pybamm.Parameter(
            f"{pref}{Domain} electrode Butler-Volmer transfer coefficient"
        )

        # SEI parameters
        self.V_bar_inner = pybamm.Parameter(
            f"{pref}Inner SEI partial molar volume [m3.mol-1]"
        )
        self.V_bar_outer = pybamm.Parameter(
            f"{pref}Outer SEI partial molar volume [m3.mol-1]"
        )

        self.j0_sei = pybamm.Parameter(
            f"{pref}SEI reaction exchange current density [A.m-2]"
        )

        self.R_sei = pybamm.Parameter(f"{pref}SEI resistivity [Ohm.m]")
        self.D_sol = pybamm.Parameter(f"{pref}Outer SEI solvent diffusivity [m2.s-1]")
        self.c_sol = pybamm.Parameter(f"{pref}Bulk solvent concentration [mol.m-3]")
        self.U_inner = pybamm.Parameter(f"{pref}Inner SEI open-circuit potential [V]")
        self.U_outer = pybamm.Parameter(f"{pref}Outer SEI open-circuit potential [V]")
        self.kappa_inner = pybamm.Parameter(
            f"{pref}Inner SEI electron conductivity [S.m-1]"
        )
        self.D_li = pybamm.Parameter(
            f"{pref}Inner SEI lithium interstitial diffusivity [m2.s-1]"
        )
        self.c_li_0 = pybamm.Parameter(
            f"{pref}Lithium interstitial reference concentration [mol.m-3]"
        )
        self.L_inner_0 = pybamm.Parameter(f"{pref}Initial inner SEI thickness [m]")
        self.L_outer_0 = pybamm.Parameter(f"{pref}Initial outer SEI thickness [m]")
        self.L_inner_crack_0 = pybamm.Parameter(
            f"{pref}Initial inner SEI on cracks thickness [m]"
        )
        self.L_outer_crack_0 = pybamm.Parameter(
            f"{pref}Initial outer SEI on cracks thickness [m]"
        )

        self.L_sei_0 = self.L_inner_0 + self.L_outer_0
        self.E_sei = pybamm.Parameter(f"{pref}SEI growth activation energy [J.mol-1]")
        self.alpha_SEI = pybamm.Parameter(f"{pref}SEI growth transfer coefficient")
        self.inner_sei_proportion = pybamm.Parameter(
            f"{pref}Inner SEI reaction proportion"
        )
        self.z_sei = pybamm.Parameter(f"{pref}Ratio of lithium moles to SEI moles")

        # EC reaction
        self.c_ec_0 = pybamm.Parameter(
            f"{pref}EC initial concentration in electrolyte [mol.m-3]"
        )
        self.D_ec = pybamm.Parameter(f"{pref}EC diffusivity [m2.s-1]")
        self.k_sei = pybamm.Parameter(f"{pref}SEI kinetic rate constant [m.s-1]")
        self.U_sei = pybamm.Parameter(f"{pref}SEI open-circuit potential [V]")

        # Lithium plating parameters
        self.c_Li_typ = pybamm.Parameter(
            f"{pref}Typical plated lithium concentration [mol.m-3]"
        )
        self.c_plated_Li_0 = pybamm.Parameter(
            f"{pref}Initial plated lithium concentration [mol.m-3]"
        )
        self.alpha_plating = pybamm.Parameter(
            f"{pref}Lithium plating transfer coefficient"
        )
        self.alpha_stripping = 1 - self.alpha_plating

        if main.options.electrode_types[domain] == "planar":
            self.n_Li_init = pybamm.Scalar(0)
            self.Q_Li_init = pybamm.Scalar(0)
            self.U_init = pybamm.Scalar(0)
            return

        # Spatial variables for parameters that depend on position within the cell
        # and/or particle
        x = pybamm.SpatialVariable(
            f"x_{domain[0]}",
            domain=[f"{domain} electrode"],
            auxiliary_domains={"secondary": "current collector"},
            coord_sys="cartesian",
        )
        r = pybamm.SpatialVariable(
            f"r_{domain[0]}",
            domain=[f"{domain} {self.phase_name}particle"],
            auxiliary_domains={
                "secondary": f"{domain} electrode",
                "tertiary": "current collector",
            },
            coord_sys="spherical polar",
        )

        # Microscale geometry
        # Note: the surface area to volume ratio is defined later with the function
        # parameters. The particle size as a function of through-cell position is
        # already defined in geometric_parameters.py
        self.R = self.geo.R
        self.R_typ = self.geo.R_typ
        # Particle-size distribution parameters
        self.R_min = self.geo.R_min
        self.R_max = self.geo.R_max
        self.f_a_dist = self.geo.f_a_dist

        # Particle properties
        self.epsilon_s = pybamm.FunctionParameter(
            f"{pref}{Domain} electrode active material volume fraction",
            {"Through-cell distance (x) [m]": x},
        )
        self.epsilon_s_av = pybamm.xyz_average(self.epsilon_s)
        self.c_max = pybamm.Parameter(
            f"{pref}Maximum concentration in {domain} electrode [mol.m-3]"
        )
        if self.options["open-circuit potential"] == "MSMR":
            self.U_init = pybamm.Parameter(
                f"{pref}Initial voltage in {domain} electrode [V]",
            )
            self.c_init = self.x(self.U_init, main.T_init) * self.c_max
        else:
            self.c_init = pybamm.FunctionParameter(
                f"{pref}Initial concentration in {domain} electrode [mol.m-3]",
                {
                    "Radial distance (r) [m]": r,
                    "Through-cell distance (x) [m]": pybamm.PrimaryBroadcast(
                        x, f"{domain} {phase_name}particle"
                    ),
                },
            )
        self.c_init_av = pybamm.xyz_average(pybamm.r_average(self.c_init))
        self.sto_init_av = self.c_init_av / self.c_max
        eps_c_init_av = pybamm.xyz_average(
            self.epsilon_s * pybamm.r_average(self.c_init)
        )
        # if self.options['open-circuit potential'] == 'Plett':
        self.hysteresis_decay = pybamm.Parameter(
            f"{pref}{Domain} particle hysteresis decay rate"
        )
        self.hysteresis_switch = pybamm.Parameter(
            f"{pref}{Domain} particle hysteresis switching factor"
        )
        self.h_init = pybamm.Scalar(0)

        if self.options["open-circuit potential"] != "MSMR":
            self.U_init = self.U(self.sto_init_av, main.T_init)

        # Electrode loading and capacity
        self.elec_loading = (
            self.epsilon_s_av * self.domain_param.L * self.c_max * main.F / 3600
        )
        self.n_Li_init = eps_c_init_av * self.domain_param.L * main.A_cc
        self.Q_Li_init = self.n_Li_init * main.F / 3600
        self.Q_init = self.elec_loading * main.A_cc

        if self.options["particle shape"] == "spherical":
            self.a_typ = 3 * pybamm.xyz_average(self.epsilon_s) / self.R_typ

    def D(self, c_s, T, lithiation=None):
        """
        Dimensional diffusivity in particle. In the parameter sets this is defined as
        a function of stoichiometry (dimensionless), but in the models we use it as a
        function of concentration (mol/m3). We convert from concentration to
        stoichiometry by dividing by the maximum concentration.
        """
        Domain = self.domain.capitalize()
        sto = c_s / self.c_max
        tol = pybamm.settings.tolerances["D__c_s"]
        sto = pybamm.maximum(pybamm.minimum(sto, 1 - tol), tol)
        if lithiation is None:
            lithiation = ""
        else:
            lithiation = lithiation + " "
        inputs = {
            f"{self.phase_prefactor}{Domain} particle stoichiometry": sto,
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            f"{self.phase_prefactor}{Domain} particle {lithiation}"
            "diffusivity [m2.s-1]",
            inputs,
        )

    def j0(self, c_e, c_s_surf, T, lithiation=None):
        """Dimensional exchange-current density [A.m-2]"""
        tol = pybamm.settings.tolerances["j0__c_e"]
        c_e = pybamm.maximum(c_e, tol)
        tol = pybamm.settings.tolerances["j0__c_s"]
        c_s_surf = pybamm.maximum(
            pybamm.minimum(c_s_surf, (1 - tol) * self.c_max), tol * self.c_max
        )
        domain, Domain = self.domain_Domain
        if lithiation is None:
            lithiation = ""
        else:
            lithiation = lithiation + " "
        inputs = {
            "Electrolyte concentration [mol.m-3]": c_e,
            f"{Domain} particle surface concentration [mol.m-3]": c_s_surf,
            f"{self.phase_prefactor}Maximum {domain} particle "
            "surface concentration [mol.m-3]": self.c_max,
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            f"{self.phase_prefactor}{Domain} electrode {lithiation}"
            "exchange-current density [A.m-2]",
            inputs,
        )

    def j0_stripping(self, c_e, c_Li, T):
        """Dimensional exchange-current density for stripping [A.m-2]"""
        Domain = self.domain.capitalize()
        inputs = {
            f"{Domain} electrolyte concentration [mol.m-3]": c_e,
            f"{Domain} plated lithium concentration [mol.m-3]": c_Li,
            f"{Domain} temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            f"{self.phase_prefactor}Exchange-current density for stripping [A.m-2]",
            inputs,
        )

    def j0_plating(self, c_e, c_Li, T):
        """Dimensional exchange-current density for plating [A.m-2]"""
        Domain = self.domain.capitalize()
        inputs = {
            f"{Domain} electrolyte concentration [mol.m-3]": c_e,
            f"{Domain} plated lithium concentration [mol.m-3]": c_Li,
            f"{Domain} temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            f"{self.phase_prefactor}Exchange-current density for plating [A.m-2]",
            inputs,
        )

    def dead_lithium_decay_rate(self, L_sei):
        """Dimensional dead lithium decay rate [s-1]"""
        Domain = self.domain.capitalize()
        inputs = {f"{Domain} total {self.phase_name}SEI thickness [m]": L_sei}
        return pybamm.FunctionParameter(
            f"{self.phase_prefactor}Dead lithium decay rate [s-1]", inputs
        )

    def U(self, sto, T, lithiation=None):
        """
        Dimensional open-circuit potential [V], calculated as
        U(x,T) = U_ref(x) + dUdT(x) * (T - T_ref). See the documentation for
        dUdT for more details.
        """
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

        dudt_func = self.dUdT(sto)
        u_ref = u_ref + (T - self.main_param.T_ref) * dudt_func

        # add a term to ensure that the OCP goes to infinity at 0 and -infinity at 1
        # this will not affect the OCP for most values of sto
        # see #1435
        out = u_ref + 1e-6 * (1 / sto + 1 / (sto - 1))

        if self.domain == "negative":
            out.print_name = r"U_\mathrm{n}(c^\mathrm{surf}_\mathrm{s,n}, T)"
        elif self.domain == "positive":
            out.print_name = r"U_\mathrm{p}(c^\mathrm{surf}_\mathrm{s,p}, T)"
        return out

    def dUdT(self, sto):
        """
        Dimensional entropic change of the open-circuit potential [V.K-1].

        Note: in the "classical" formulation, the open-circuit potential is defined
        as U(x,T) = U_ref(x) + dUdT(x) * (T - T_ref). The user provides U_ref and
        dUdT, and the model uses these to calculate U. dUdT is also used to calculate
        the reversible heat generation term in the thermal model. However, in the
        "MSMR" formulation, stoichiometry is explicitly defined as a function of U and
        T, and dUdT is only used to calculate the reversible heat generation term.
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

    def X_j(self, T, index):
        "Available host sites indexed by reaction j"
        inputs = {"Temperature [K]": T}
        domain = self.domain
        d = domain[0]
        Xj = pybamm.FunctionParameter(f"X_{d}_{index}", inputs)
        return Xj

    def U0_j(self, T, index):
        "Equilibrium potential indexed by reaction j"
        inputs = {"Temperature [K]": T}
        domain = self.domain
        d = domain[0]
        U0j = pybamm.FunctionParameter(f"U0_{d}_{index}", inputs)
        return U0j

    def w_j(self, T, index):
        "Order parameter indexed by reaction j"
        inputs = {"Temperature [K]": T}
        domain = self.domain
        d = domain[0]
        wj = pybamm.FunctionParameter(f"w_{d}_{index}", inputs)
        return wj

    def alpha_bv_j(self, T, index):
        "Dimensional Butler-Volmer exchange-current density indexed by reaction j"
        inputs = {"Temperature [K]": T}
        domain = self.domain
        d = domain[0]
        alpha_bv_j = pybamm.FunctionParameter(f"a_{d}_{index}", inputs)
        return alpha_bv_j

    def x_j(self, U, T, index):
        "Fractional occupancy of site j as a function of potential"
        f = self.main_param.F / (self.main_param.R * T)
        U0j = self.U0_j(T, index)
        wj = self.w_j(T, index)
        Xj = self.X_j(T, index)
        # Equation 5, Baker et al 2018
        xj = Xj / (1 + pybamm.exp(f * (U - U0j) / wj))
        return xj

    def dxdU_j(self, U, T, index):
        "Derivative of fractional occupancy of site j as a function of potential [V-1]"
        f = self.main_param.F / (self.main_param.R * T)
        U0j = self.U0_j(T, index)
        wj = self.w_j(T, index)
        Xj = self.X_j(T, index)
        e = pybamm.exp(f * (U - U0j) / wj)
        # Equation 25, Baker et al 2018
        dxjdU = -(f / wj) * (Xj * e) / (1 + e) ** 2
        return dxjdU

    def j0_j(self, c_e, U, T, index):
        "Exchange-current density index by reaction j [A.m-2]"
        domain = self.domain
        d = domain[0]

        tol = pybamm.settings.tolerances["j0__c_e"]
        c_e = pybamm.maximum(c_e, tol)
        c_e_ref = self.main_param.c_e_init
        xj = self.x_j(U, T, index)
        # xj = pybamm.maximum(pybamm.minimum(xj, (1 - tol)), tol)

        f = self.main_param.F / (self.main_param.R * T)
        wj = self.w_j(T, index)
        self.X_j(T, index)
        aj = self.alpha_bv_j(T, index)
        j0_ref_j = pybamm.FunctionParameter(
            f"j0_ref_{d}_{index}", {"Temperature [K]": T}
        )

        # Equation 16, Baker et al 2018. The original formulation would be implemented
        # as:
        # j0_j = (
        #    j0_ref_j
        #    * xj ** (wj * aj)
        #    * (Xj - xj) ** (wj * (1 - aj))
        #    * (c_e / c_e_ref) ** (1 - aj)
        # )
        # However, we reformulate in terms of potential to avoid singularity as x_j
        # approaches X_j
        j0_j = (
            j0_ref_j
            * xj**wj
            * pybamm.exp(f * (1 - aj) * (U - self.U0_j(T, index)))
            * (c_e / c_e_ref) ** (1 - aj)
        )
        return j0_j

    def x(self, U, T):
        "Stoichiometry as a function of potential (for use with MSMR models)"
        N = int(self.options["number of MSMR reactions"])
        # Equation 6, Baker et al 2018
        x = 0
        for i in range(N):
            x += self.x_j(U, T, i)
        return x

    def dxdU(self, U, T):
        """
        Differential stoichiometry as a function of potential (for use with MSMR models)
        """
        N = int(self.options["number of MSMR reactions"])
        # Equation 25, Baker et al 2018
        dxdU = 0
        for i in range(N):
            dxdU += self.dxdU_j(U, T, i)
        return dxdU

    def t_change(self, sto):
        """
        Volume change for the electrode; sto should be R-averaged
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
