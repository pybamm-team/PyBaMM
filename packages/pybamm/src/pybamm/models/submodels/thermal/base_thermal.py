#
# Base class for thermal effects
#
import numpy as np

import pybamm


class BaseThermal(pybamm.BaseSubModel):
    """
    Base class for thermal effects

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    x_average : bool, optional
        Whether the heat of mixing uses the x-averaged particle variables. Default is
        False.
    x_average_particles : bool, optional
        Whether the particle submodels solve for x-averaged concentrations. Default is
        ``x_average``.
    full_electrolyte_conductivity : bool, optional
        Whether the electrolyte potential is resolved through the cell (full rather
        than reduced-order conductivity). Default is False.
    """

    def __init__(
        self,
        param,
        options=None,
        x_average=False,
        x_average_particles=None,
        full_electrolyte_conductivity=False,
    ):
        super().__init__(param, options=options)
        self.use_lumped_thermal_capacity = self.options.get(
            "use lumped thermal capacity", "false"
        )
        self.x_average = x_average
        self.x_average_particles = (
            x_average if x_average_particles is None else x_average_particles
        )
        self.full_electrolyte_conductivity = full_electrolyte_conductivity

        if self.options["heat of mixing"] == "true":
            pybamm.citations.register("Richardson2021")

    def _get_x_uniform_temperatures(self, T_x_av, T_vol_av):
        """Temperatures of a cell whose temperature does not vary through x."""
        T_dict = {
            "negative current collector": T_x_av,
            "positive current collector": T_x_av,
            "x-averaged cell": T_x_av,
            "volume-averaged cell": T_vol_av,
        }
        for domain in ["negative electrode", "separator", "positive electrode"]:
            T_dict[domain] = pybamm.PrimaryBroadcast(T_x_av, domain)
        return T_dict

    def _get_standard_fundamental_variables(self, T_dict):
        """
        Temperature variables from the temperature in each domain. The averages
        are passed in explicitly since they differ between submodels.
        """

        # The variable T is the concatenation of the temperature in the middle domains
        # (e.g. negative electrode, separator and positive electrode for a full cell),
        # excluding current collectors, for use in the electrochemical models
        T_mid = [T_dict[k] for k in self.options.whole_cell_domains]
        T = pybamm.concatenation(*T_mid)

        # Get the ambient temperature, which can be specified as a function of space
        # (y, z) only and time
        y = pybamm.standard_spatial_vars.y
        z = pybamm.standard_spatial_vars.z
        T_amb = self.param.T_amb(y, z, pybamm.t)
        T_amb_av = self._yz_average(T_amb)

        variables = {
            "Ambient temperature [K]": T_amb,
            "Volume-averaged ambient temperature [K]": T_amb_av,
            "Cell temperature [K]": T,
        }
        for name, var in T_dict.items():
            Name = name.capitalize()
            variables[f"{Name} temperature [K]"] = var
            if name in ["negative electrode", "separator", "positive electrode"]:
                variables[f"X-averaged {name} temperature [K]"] = pybamm.x_average(var)

        # Calculate temperatures in Celsius
        variables_Kelvin = variables.copy()
        for name_K, var in variables_Kelvin.items():
            name_C = name_K.replace("[K]", "[C]")
            variables.update({name_C: var - 273.15})

        return variables

    def _get_heat_source_variables(self, T_dict):
        """
        Heat sources, their averages and the effective heat capacity, given the
        temperature in each domain.
        """
        Q_ohm, Q_ohm_s_cn, Q_ohm_s_cp = self._ohmic_heating(T_dict)
        Q_rxn, Q_rev, Q_hys = self._electrochemical_heating(T_dict)
        Q_mix = pybamm.concatenation(*self._heat_of_mixing(T_dict))
        Q_cr = self._contact_resistance_heating(T_dict["volume-averaged cell"])
        Q_cr = pybamm.FullBroadcast(Q_cr, Q_ohm.domain, "current collector")
        Q = Q_ohm + Q_rxn + Q_rev + Q_mix + Q_hys + Q_cr

        # Only Ohmic heating has a current collector contribution
        sources = {
            "Ohmic heating": (Q_ohm, Q_ohm_s_cn, Q_ohm_s_cp),
            "irreversible electrochemical heating": (Q_rxn, 0, 0),
            "reversible heating": (Q_rev, 0, 0),
            "heat of mixing": (Q_mix, 0, 0),
            "hysteresis electrochemical heating": (Q_hys, 0, 0),
            "contact resistance heating": (Q_cr, 0, 0),
            "total heating": (Q, Q_ohm_s_cn, Q_ohm_s_cp),
        }
        variables = {}
        for name, (Q_k, Q_k_cn, Q_k_cp) in sources.items():
            variables.update(
                self._get_heat_source_average_variables(name, Q_k, Q_k_cn, Q_k_cp)
            )
        # Misspelt name kept for backwards compatibility
        variables["Volume-averaged heating of mixing [W.m-3]"] = variables[
            "Volume-averaged heat of mixing [W.m-3]"
        ]

        if self.use_lumped_thermal_capacity == "true":
            rho_c_p_eff_av = self.param.cell_heat_capacity
        else:
            rho_c_p_eff_av = self.param.rho_c_p_eff(T_dict["volume-averaged cell"])

        variables.update(
            {
                "Negative current collector Ohmic heating [W.m-3]": Q_ohm_s_cn,
                "Positive current collector Ohmic heating [W.m-3]": Q_ohm_s_cp,
                "Volume-averaged effective heat capacity [J.K-1.m-3]": rho_c_p_eff_av,
                "Cell thermal volume [m3]": self.param.V_cell,
            }
        )
        return variables

    def _get_heat_source_average_variables(self, name, Q, Q_cn, Q_cp):
        """
        Averages of a heat source ``Q`` in the cell, with ``Q_cn`` and ``Q_cp`` in
        the current collectors.
        """
        Name = name[0].upper() + name[1:]
        # X-average over the entire cell including current collectors (can still
        # depend on y and z in pouch cell models)
        Q_av = self._x_average(Q, Q_cn, Q_cp)
        # Integrated heat source per unit simulated electrode-pair area
        Q_Wm2 = Q_av * self.param.L
        # Total over the cell: Q_Wm2 is per electrode pair and per *modelled* area
        n_elec = self.param.n_electrodes_parallel
        A = self.param.L_y * self.param.L_z
        Q_W = self._yz_average(Q_Wm2) * n_elec * A
        return {
            f"{Name} [W.m-3]": Q,
            f"X-averaged {name} [W.m-3]": Q_av,
            # Averaged over the *actual* cell volume
            f"Volume-averaged {name} [W.m-3]": Q_W / self.param.V_cell,
            f"{Name} per unit electrode-pair area [W.m-2]": Q_Wm2,
            f"{Name} [W]": Q_W,
        }

    def _coupled(self, name, domain=None):
        """
        Placeholder for a variable of another submodel over ``domain``, with the
        current collector as secondary domain when ``domain`` is in the cell.
        """
        if domain is None or domain == "current collector":
            return self.coupled_variable(name, domain=domain)
        return self.coupled_variable(
            name, domain=domain, auxiliary_domains={"secondary": "current collector"}
        )

    def _coupled_surface_temperature(self):
        """Placeholder for the surface temperature."""
        if self.options["surface temperature"] == "lumped":
            return self._coupled("Surface temperature [K]")
        return self._coupled("Surface temperature [K]", "current collector")

    def _ohmic_heating(self, T_dict):
        """Ohmic heating in the cell and in the two current collectors."""
        Q_ohm_s_cn, Q_ohm_s_cp = self._current_collector_heating()
        Q_ohm_s = self._solid_ohmic_heating(T_dict["negative electrode"])
        Q_ohm_e = self._electrolyte_ohmic_heating(T_dict)
        return Q_ohm_s + Q_ohm_e, Q_ohm_s_cn, Q_ohm_s_cp

    def _solid_ohmic_heating(self, T_n):
        """Ohmic heating in the electrode solid phase."""
        if self.options.electrode_types["negative"] == "planar":
            i_boundary_cc = self._coupled(
                "Current collector current density [A.m-2]", "current collector"
            )
            Q_ohm_s_n = i_boundary_cc**2 / self.param.n.sigma(None, T_n)
        else:
            Q_ohm_s_n = self._solid_ohmic_heating_in_electrode("negative")
        Q_ohm_s_s = pybamm.FullBroadcast(0, ["separator"], "current collector")
        Q_ohm_s_p = self._solid_ohmic_heating_in_electrode("positive")
        return pybamm.concatenation(Q_ohm_s_n, Q_ohm_s_s, Q_ohm_s_p)

    def _solid_ohmic_heating_in_electrode(self, domain):
        Domain = domain.capitalize()
        electrode = f"{domain} electrode"
        i_s = self._coupled(f"{Domain} electrode current density [A.m-2]", electrode)
        phi_s = self._coupled(f"{Domain} electrode potential [V]", electrode)
        return -pybamm.inner(i_s, pybamm.grad(phi_s))

    def _electrolyte_ohmic_heating(self, T_dict):
        """
        Ohmic heating in the electrolyte. Reduced-order and surface-form
        conductivities define the current in each domain, so the heating is computed
        by domain; full conductivity defines it over the whole cell.
        """
        planar = self.options.electrode_types["negative"] == "planar"
        if self.full_electrolyte_conductivity:
            if self.options["surface form"] == "false" and not planar:
                whole_cell = self.options.whole_cell_domains
                i_e = self._coupled("Electrolyte current density [A.m-2]", whole_cell)
                phi_e = self._coupled("Electrolyte potential [V]", whole_cell)
                return -pybamm.inner(i_e, pybamm.grad(phi_e))
            if self.options["surface form"] == "false":
                i_e_s, i_e_p = self._get_half_cell_electrolyte_current(T_dict)
            else:
                i_e_s = self._coupled(
                    "Separator electrolyte current density [A.m-2]", "separator"
                )
                i_e_p = self._coupled(
                    "Positive electrolyte current density [A.m-2]", "positive electrode"
                )
        else:
            # all the current is carried by the electrolyte in the separator
            i_boundary_cc = self._coupled(
                "Current collector current density [A.m-2]", "current collector"
            )
            i_e_s = pybamm.PrimaryBroadcast(i_boundary_cc, "separator")
            i_e_p = self._coupled(
                "Positive electrolyte current density [A.m-2]", "positive electrode"
            )

        phi_e_s = self._coupled("Separator electrolyte potential [V]", "separator")
        phi_e_p = self._coupled(
            "Positive electrolyte potential [V]", "positive electrode"
        )
        if planar:
            Q_ohm_e_n = pybamm.FullBroadcast(
                0, ["negative electrode"], "current collector"
            )
        else:
            i_e_n = self._coupled(
                "Negative electrolyte current density [A.m-2]", "negative electrode"
            )
            phi_e_n = self._coupled(
                "Negative electrolyte potential [V]", "negative electrode"
            )
            Q_ohm_e_n = -pybamm.inner(i_e_n, pybamm.grad(phi_e_n))
        Q_ohm_e_s = -pybamm.inner(i_e_s, pybamm.grad(phi_e_s))
        Q_ohm_e_p = -pybamm.inner(i_e_p, pybamm.grad(phi_e_p))
        return pybamm.concatenation(Q_ohm_e_n, Q_ohm_e_s, Q_ohm_e_p)

    def _get_half_cell_electrolyte_current(self, T_dict):
        """
        Electrolyte current in the separator and positive electrode of a half cell,
        which full conductivity does not define by domain.
        """
        currents = []
        for domain in ["separator", "positive electrode"]:
            name = domain.split()[0]
            Name = name.capitalize()
            c_e = self._coupled(f"{Name} electrolyte concentration [mol.m-3]", domain)
            grad_phi_e = self._coupled(
                f"Gradient of {name} electrolyte potential [V.m-1]", domain
            )
            tor = self._coupled(f"{Name} electrolyte transport efficiency", domain)
            T = T_dict[domain]
            currents.append(
                (self.param.kappa_e(c_e, T) * tor)
                * (self.param.chiRT_over_Fc(c_e, T) * pybamm.grad(c_e) - grad_phi_e)
            )
        return currents

    def _current_collector_heating(self):
        """Compute Ohmic heating in current collectors."""
        cc_dimension = self.options["dimensionality"]

        # Compute the Ohmic heating for 0D current collectors
        if cc_dimension == 0:
            i_boundary_cc = self._coupled(
                "Current collector current density [A.m-2]", "current collector"
            )
            Q_s_cn = i_boundary_cc**2 / self.param.n.sigma_cc
            Q_s_cp = i_boundary_cc**2 / self.param.p.sigma_cc
        # Otherwise the current flows in the y,z directions in the current collectors
        elif cc_dimension in [1, 2]:
            phi_s_cn = self._coupled(
                "Negative current collector potential [V]", "current collector"
            )
            phi_s_cp = self._coupled(
                "Positive current collector potential [V]", "current collector"
            )
            # TODO: implement grad_squared in other spatial methods so that the
            # if statement can be removed
            if cc_dimension == 1:
                Q_s_cn = self.param.n.sigma_cc * pybamm.inner(
                    pybamm.grad(phi_s_cn), pybamm.grad(phi_s_cn)
                )
                Q_s_cp = self.param.p.sigma_cc * pybamm.inner(
                    pybamm.grad(phi_s_cp), pybamm.grad(phi_s_cp)
                )
            elif cc_dimension == 2:
                # Inner not implemented in 2D -- have to call grad_squared directly
                Q_s_cn = self.param.n.sigma_cc * pybamm.grad_squared(phi_s_cn)
                Q_s_cp = self.param.p.sigma_cc * pybamm.grad_squared(phi_s_cp)
        return Q_s_cn, Q_s_cp

    def _electrochemical_heating(self, T_dict):
        """Irreversible, reversible and hysteresis electrochemical heating."""
        if self.options.electrode_types["negative"] == "planar":
            heating_n = self._lithium_metal_heating()
        else:
            heating_n = self._electrode_heating(
                "negative", T_dict["negative electrode"]
            )
        heating_p = self._electrode_heating("positive", T_dict["positive electrode"])
        zero_s = pybamm.FullBroadcast(0, "separator", "current collector")
        return tuple(
            pybamm.concatenation(Q_n, zero_s, Q_p)
            for Q_n, Q_p in zip(heating_n, heating_p, strict=True)
        )

    def _electrode_heating(self, domain, T):
        """
        Irreversible, reversible and hysteresis heating in a porous electrode,
        summed over particle phases.
        """
        Domain = domain.capitalize()
        electrode = f"{domain} electrode"
        phases = self.options.phases[domain]
        phase_names = [""] if len(phases) == 1 else [f"{phase} " for phase in phases]
        ocp_options = getattr(self.options, domain)["open-circuit potential"]
        if isinstance(ocp_options, str):
            # a single option applies to every phase
            ocp_options = (ocp_options,) * len(phase_names)

        Q_rxn, Q_rev, Q_hys = 0, 0, 0
        for phase_name, ocp_option in zip(phase_names, ocp_options, strict=False):
            prefix = f"{Domain} electrode {phase_name}"
            a_j = self._coupled(
                f"{prefix}volumetric interfacial current density [A.m-3]", electrode
            )
            eta_r = self._coupled(f"{prefix}reaction overpotential [V]", electrode)
            eta_sei = self._coupled(f"{prefix}SEI film overpotential [V]", electrode)
            # Irreversible heating. eta_r is the kinetic overpotential net of the
            # film drop, so eta_r - eta_sei adds the resistive loss in the SEI film
            Q_rxn += a_j * (eta_r - eta_sei)
            Q_rxn += self._side_reaction_heating(domain, phase_name)
            # Reversible heating
            ocp_domain = self._open_circuit_domain(domain, ocp_option)
            if isinstance(self.param, pybamm.LeadAcidParameters):
                # the lead-acid open-circuit potential does not depend on temperature
                dUdT = self._coupled(f"{prefix}entropic change [V.K-1]")
            else:
                dUdT = self._coupled(f"{prefix}entropic change [V.K-1]", ocp_domain)
            Q_rev += a_j * T * dUdT
            # Hysteresis heating
            if ocp_option == "single":
                V_hys = 0
            else:
                pybamm.citations.register("Wycisk2023")
                U = self._coupled(f"{prefix}open-circuit potential [V]", electrode)
                U_eq = self._coupled(
                    f"{prefix}equilibrium open-circuit potential [V]", ocp_domain
                )
                V_hys = U - U_eq
            Q_hys += a_j * V_hys
        return Q_rxn, Q_rev, Q_hys

    def _open_circuit_domain(self, domain, ocp_option):
        """
        Domain of the entropic change and equilibrium open-circuit potential of a
        lithium-ion electrode. Except for MSMR, these are x-averaged when both the
        particles and the temperature are uniform in x.
        """
        if (
            ocp_option != "MSMR"
            and self.x_average_particles
            and self.options["thermal"] != "x-full"
        ):
            return "current collector"
        return f"{domain} electrode"

    def _lithium_metal_heating(self):
        """
        Irreversible, reversible and hysteresis heating at a lithium metal negative
        electrode, spread over the negative electrode.
        """
        cc = "current collector"
        # The total current splits between lithium plating and SEI growth
        j_pl = self._coupled("Lithium metal plating current density [A.m-2]", cc)
        eta_r = self._coupled("Lithium metal interface reaction overpotential [V]", cc)
        # Inverted kinetics without a film resistance leave no SEI film overpotential
        inverse_kinetics = (
            self.options["SEI"] in ["none", "constant"]
            and self.options["intercalation kinetics"]
            in ["linear", "symmetric Butler-Volmer"]
            and self.options["surface form"] == "false"
        )
        if inverse_kinetics and self.options["SEI film resistance"] == "none":
            eta_sei = self._coupled("Negative electrode SEI film overpotential [V]")
        else:
            eta_sei = self._coupled("Negative electrode SEI film overpotential [V]", cc)
        Q_rxn = j_pl * (eta_r - eta_sei)
        if self.options.negative["SEI"] not in ["none", "constant"]:
            j_sei = self._coupled(
                "Negative electrode SEI interfacial current density [A.m-2]", cc
            )
            delta_phi = self._coupled(
                "Lithium metal interface surface potential difference [V]", cc
            )
            Q_rxn += j_sei * (delta_phi - self.param.n.prim.U_sei)
        Q_rxn = pybamm.PrimaryBroadcast(Q_rxn / self.param.n.L, "negative electrode")
        zero = pybamm.FullBroadcast(0, ["negative electrode"], cc)
        return Q_rxn, zero, zero

    def _side_reaction_heating(self, domain, phase_name):
        """
        Irreversible heating from SEI growth and lithium plating in a porous
        electrode, as each side-reaction current times ``delta_phi - U`` for that
        reaction, which includes any SEI film drop.
        """
        Domain = domain.capitalize()
        electrode = f"{domain} electrode"
        phase = phase_name.strip() or "primary"
        domain_options = getattr(self.options, domain)
        phase_options = getattr(domain_options, phase)
        prefix = f"{Domain} electrode {phase_name}"
        suffix = "volumetric interfacial current density [A.m-3]"

        Q = 0
        if phase_options["SEI"] not in ["none", "constant"]:
            delta_phi = self._surface_potential_difference(domain)
            U_sei = self.param.domain_params[domain].phase_params[phase].U_sei
            reactions = ["SEI"]
            if domain_options["SEI on cracks"] == "true":
                reactions.append("SEI on cracks")
            for reaction in reactions:
                a_j_sei = self._coupled(f"{prefix}{reaction} {suffix}", electrode)
                Q += a_j_sei * (delta_phi - U_sei)
        if phase_options["lithium plating"] != "none":
            # Lithium plating has zero open-circuit potential
            delta_phi = self._surface_potential_difference(domain)
            a_j_pl = self._coupled(f"{prefix}lithium plating {suffix}", electrode)
            Q += a_j_pl * delta_phi
        return Q

    def _surface_potential_difference(self, domain):
        return self._coupled(
            f"{domain.capitalize()} electrode surface potential difference [V]",
            f"{domain} electrode",
        )

    def _phase_heat_of_mixing(self, domain, phase_name, phase_param, T_electrode):
        """
        Compute the heat of mixing source term for a single particle phase.

        The OCP is the phase's equilibrium OCP, the same one the hysteresis heat is
        referenced to, not the hysteresis-weighted OCP that sets the voltage. Energy
        only balances if both heat sources share that reference.
        """
        phase_options = getattr(getattr(self.options, domain), phase_param.phase)
        if phase_options["particle"] in ["uniform profile", "MSMR"]:
            # uniform: no concentration gradient to mix; MSMR: not implemented
            return pybamm.FullBroadcast(0, [f"{domain} electrode"], "current collector")
        Domain = domain.capitalize()
        F = pybamm.constants.F.value
        electrode = f"{domain} electrode"
        a = self._coupled(
            f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]",
            electrode,
        )
        R = self._coupled(f"{Domain} {phase_name}particle radius [m]", electrode)
        N = a / (4 * np.pi * R**2)

        particle = f"{domain} {phase_name}particle"
        if self.x_average:
            prefix = f"X-averaged {domain}"
            T = pybamm.x_average(T_electrode)
            particle_domains = {"primary": particle, "secondary": "current collector"}
        else:
            prefix = Domain
            T = T_electrode
            particle_domains = {
                "primary": particle,
                "secondary": electrode,
                "tertiary": "current collector",
            }

        def coupled_particle(name):
            return self.coupled_variable(
                f"{prefix} {phase_name}particle {name}", domains=particle_domains
            )

        c = coupled_particle("concentration [mol.m-3]")
        # the diffusivity the particle flux actually uses, including any current
        # sigmoid or stress-induced diffusion, so the dissipation matches the flux
        D = coupled_particle("effective diffusivity [m2.s-1]")
        T_part = pybamm.PrimaryBroadcast(T, [particle])
        sto = c / phase_param.c_max
        U_eq = phase_param.U(sto, T_part)
        if phase_options["particle"] in ["quadratic profile", "quartic profile"]:
            # c is a polynomial in r whose closed-form flux is N_s = -D dc/dr, so
            # D (dc/dr)^2 = N_s^2 / D; a finite-volume gradient doesn't apply
            N_s = coupled_particle("flux [mol.m-2.s-1]")
            # differentiate by the whole stoichiometry, which simplification keeps
            # intact once the polynomial for c is substituted
            dUdc = U_eq.diff(sto) / phase_param.c_max
            integrand_r = N_s**2 / D * dUdc
        else:
            dc_dr2 = pybamm.inner(pybamm.grad(c), pybamm.grad(c))
            integrand_r = D * dc_dr2 * U_eq.diff(c)
        integration_variable_r = [
            pybamm.SpatialVariable("r", domain=integrand_r.domain)
        ]
        integral_r = pybamm.Integral(integrand_r, integration_variable_r)
        return -F * N * integral_r

    def _electrode_heat_of_mixing(self, domain, T_electrode):
        """Sum the heat of mixing over the particle phases of one electrode."""
        electrode_param = self.param.domain_params[domain]
        phases = self.options.phases[domain]
        names = [""] if len(phases) == 1 else [f"{phase} " for phase in phases]
        return sum(
            self._phase_heat_of_mixing(
                domain, name, electrode_param.phase_params[phase], T_electrode
            )
            for name, phase in zip(names, phases, strict=True)
        )

    def _heat_of_mixing(self, T_dict):
        """Heat of mixing in the negative electrode, separator and positive electrode."""
        Q_mix = {
            domain: pybamm.FullBroadcast(0, [domain], "current collector")
            for domain in ["negative electrode", "separator", "positive electrode"]
        }
        if self.options["heat of mixing"] == "true":
            electrodes = ["positive"]
            if self.options.electrode_types["negative"] == "porous":
                electrodes.append("negative")
            for domain in electrodes:
                Q_mix[f"{domain} electrode"] = self._electrode_heat_of_mixing(
                    domain, T_dict[f"{domain} electrode"]
                )
        return Q_mix.values()

    def _contact_resistance_heating(self, T_vol_av):
        """
        Heating from the contact resistance, which lumps unresolved series
        resistances so has no location and is spread uniformly over the stack.
        """
        if self.options["contact resistance"] == "false":
            return pybamm.Scalar(0)
        I = self._coupled("Current [A]")
        V_stack = (
            self.param.n_electrodes_parallel
            * self.param.L_y
            * self.param.L_z
            * self.param.L_x
        )
        return I**2 * self.param.R_contact(T_vol_av) / V_stack

    def _x_average(self, var, var_cn, var_cp):
        """
        Computes the X-average over the whole cell (including current collectors)
        from the variable in the cell (negative electrode, separator,
        positive electrode), negative current collector, and positive current
        collector.
        Note: we do this as we cannot create a single variable which is
        the concatenation [var_cn, var, var_cp] since var_cn and var_cp share the
        same domain. (In the N+1D formulation the current collector variables are
        assumed independent of x, so we do not make the distinction between negative
        and positive current collectors in the geometry).
        """
        out = (
            self.param.n.L_cc * var_cn
            + self.param.L_x * pybamm.x_average(var)
            + self.param.p.L_cc * var_cp
        ) / self.param.L
        return out

    def _yz_average(self, var):
        """Computes the y-z average."""
        # TODO: change the behaviour of z_average and yz_average so the if statement
        # can be removed
        if self.options["dimensionality"] in [0, 1]:
            return pybamm.z_average(var)
        elif self.options["dimensionality"] == 2:
            return pybamm.yz_average(var)
