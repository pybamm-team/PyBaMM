#
# Base interface class
#

import pybamm


class BaseInterface(pybamm.BaseSubModel):
    """
    Base class for interfacial currents

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model. See
        :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")
    """

    def __init__(self, param, domain, reaction, options, phase="primary"):
        super().__init__(param, domain, options=options, phase=phase)
        if reaction in ["lithium-ion main", "lithium metal plating"]:
            self.reaction_name = ""
        elif reaction == "lead-acid main":
            self.reaction_name = ""  # empty reaction name for the main reaction
        elif reaction == "lead-acid oxygen":
            self.reaction_name = "oxygen "
        elif reaction in ["SEI", "SEI on cracks", "lithium plating"]:
            self.reaction_name = reaction + " "

        if reaction in [
            "lithium-ion main",
            "lithium metal plating",
            "lithium plating",
            "SEI",
            "SEI on cracks",
        ]:
            # phase_name can be "" or "primary " or "secondary "
            self.reaction_name = self.phase_name + self.reaction_name

        self.reaction = reaction

    def _get_exchange_current_density(self, variables):
        """
        A private function to obtain the exchange current density

        Parameters
        ----------
        variables: dict
            The variables in the full model.

        Returns
        -------
        j0 : :class: `pybamm.Symbol`
            The exchange current density.
        """
        param = self.param
        phase_param = self.phase_param
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        domain_options = getattr(self.options, domain)

        c_e = variables[f"{Domain} electrolyte concentration [mol.m-3]"]
        T = variables[f"{Domain} electrode temperature [K]"]

        if self.reaction == "lithium-ion main":
            # For "particle-size distribution" submodels, take distribution version
            # of c_s_surf that depends on particle size.
            domain_options = getattr(self.options, domain)
            if domain_options["particle size"] == "distribution":
                c_s_surf = variables[
                    f"{Domain} {phase_name}particle surface "
                    "concentration distribution [mol.m-3]"
                ]
                # If all variables were broadcast (in "x"), take only the orphans,
                # then re-broadcast c_e
                if (
                    isinstance(c_s_surf, pybamm.Broadcast)
                    and isinstance(c_e, pybamm.Broadcast)
                    and isinstance(T, pybamm.Broadcast)
                ):
                    c_s_surf = c_s_surf.orphans[0]
                    c_e = c_e.orphans[0]
                    T = T.orphans[0]

                    # as c_e must now be a scalar, re-broadcast to
                    # "current collector"
                    c_e = pybamm.PrimaryBroadcast(c_e, ["current collector"])
                # broadcast c_e, T onto "particle size"
                c_e = pybamm.PrimaryBroadcast(c_e, [f"{domain} particle size"])
                T = pybamm.PrimaryBroadcast(T, [f"{domain} particle size"])

            else:
                c_s_surf = variables[
                    f"{Domain} {phase_name}particle surface concentration [mol.m-3]"
                ]
                # If all variables were broadcast, take only the orphans
                if (
                    isinstance(c_s_surf, pybamm.Broadcast)
                    and isinstance(c_e, pybamm.Broadcast)
                    and isinstance(T, pybamm.Broadcast)
                ):
                    c_s_surf = c_s_surf.orphans[0]
                    c_e = c_e.orphans[0]
                    T = T.orphans[0]
            # Get main reaction exchange-current density (may have empirical hysteresis)
            j0_option = getattr(domain_options, self.phase)["exchange-current density"]
            if j0_option == "single":
                j0 = phase_param.j0(c_e, c_s_surf, T)
            elif j0_option == "current sigmoid":
                current = variables["Total current density [A.m-2]"]
                k = 100
                if Domain == "Positive":
                    lithiation_current = current
                elif Domain == "Negative":
                    lithiation_current = -current
                m_lith = pybamm.sigmoid(
                    0, lithiation_current, k
                )  # lithiation_current > 0
                m_delith = 1 - m_lith  # lithiation_current < 0
                j0_lith = phase_param.j0(c_e, c_s_surf, T, "lithiation")
                j0_delith = phase_param.j0(c_e, c_s_surf, T, "delithiation")
                j0 = m_lith * j0_lith + m_delith * j0_delith

        elif self.reaction == "lithium metal plating":
            # compute T on the surface of the anode (interface with separator)
            T = pybamm.boundary_value(T, "right")
            c_Li_metal = 1 / param.V_bar_Li
            j0 = param.j0_Li_metal(c_e, c_Li_metal, T)

        elif self.reaction == "lead-acid main":
            # If variable was broadcast, take only the orphan
            if isinstance(c_e, pybamm.Broadcast) and isinstance(T, pybamm.Broadcast):
                c_e = c_e.orphans[0]
                T = T.orphans[0]
            j0 = phase_param.j0(c_e, T)

        elif self.reaction == "lead-acid oxygen":
            # If variable was broadcast, take only the orphan
            if isinstance(c_e, pybamm.Broadcast) and isinstance(T, pybamm.Broadcast):
                c_e = c_e.orphans[0]
                T = T.orphans[0]
            if self.domain == "negative":
                j0 = pybamm.Scalar(0)
            elif self.domain == "positive":
                j0 = param.p.prim.j0_Ox(c_e, T)

        return j0

    def _get_number_of_electrons_in_reaction(self):
        """Returns the number of electrons in the reaction."""
        if self.reaction in [
            "lithium-ion main",
            "lithium metal plating",
        ]:
            return self.phase_param.ne
        elif self.reaction == "lead-acid main":
            return self.phase_param.ne
        elif self.reaction == "lead-acid oxygen":
            return self.param.ne_Ox

    def _get_average_total_interfacial_current_density(self, variables):
        """
        Method to obtain the average total interfacial current density.

        Note: for lithium-ion models this is only exact if all the particles have
        the same radius. For the current set of models implemeted in pybamm,
        having the radius as a function of through-cell distance only makes sense
        for the DFN model. In the DFN, the correct average interfacial current density
        is computed in 'base_kinetics.py' by averaging the actual interfacial current
        density. The approximation here is only used to get the approximate constant
        additional resistance term for the "average" SEI film resistance model
        (if using), where only negligible errors will be introduced.

        For "leading-order" and "composite" submodels (as used in the SPM and SPMe)
        there is only a single particle radius, so this method returns correct result.
        """
        domain = self.domain
        i_boundary_cc = variables["Current collector current density [A.m-2]"]

        if self.options.electrode_types[domain] == "planar":
            # In a half-cell the total interfacial current density is the current
            # collector current density, not divided by electrode thickness
            i_boundary_cc = variables["Current collector current density [A.m-2]"]
            j_total_average = i_boundary_cc
            a_j_total_average = i_boundary_cc
        else:
            a_av = variables[
                f"X-averaged {domain} electrode {self.phase_name}"
                "surface area to volume ratio [m-1]"
            ]
            sgn = 1 if self.domain == "negative" else -1

            a_j_total_average = sgn * i_boundary_cc / (self.domain_param.L)
            j_total_average = a_j_total_average / a_av

        return j_total_average, a_j_total_average

    def _get_standard_interfacial_current_variables(self, j):
        domain, Domain = self.domain_Domain
        reaction_name = self.reaction_name

        j.print_name = f"j_{domain[0]}"

        if self.reaction == "lithium metal plating":
            # Half-cell domain, j should not be broadcast
            variables = {"Lithium metal plating current density [A.m-2]": j}
            return variables

        # Size average. For j variables that depend on particle size, see
        # "_get_standard_size_distribution_interfacial_current_variables"
        if j.domain in [["negative particle size"], ["positive particle size"]]:
            j = pybamm.size_average(j)
        # Average, and broadcast if necessary
        j_av = pybamm.x_average(j)
        if j.domain == []:
            j = pybamm.FullBroadcast(j, f"{domain} electrode", "current collector")
        elif j.domain == ["current collector"]:
            j = pybamm.PrimaryBroadcast(j, f"{domain} electrode")

        variables = {
            f"{Domain} electrode {reaction_name}"
            "interfacial current density [A.m-2]": j,
            f"X-averaged {domain} electrode {reaction_name}"
            "interfacial current density [A.m-2]": j_av,
        }

        return variables

    def _get_standard_total_interfacial_current_variables(self, j_tot_av, a_j_tot_av):
        domain = self.domain

        if self.options.electrode_types[domain] == "planar":
            variables = {
                "Lithium metal total interfacial current density [A.m-2]": j_tot_av,
            }
        else:
            variables = {
                f"X-averaged {domain} electrode total interfacial "
                "current density [A.m-2]": j_tot_av,
                f"X-averaged {domain} electrode total volumetric interfacial "
                "current density [A.m-3]": a_j_tot_av,
            }

        return variables

    def _get_standard_exchange_current_variables(self, j0):
        domain, Domain = self.domain_Domain
        reaction_name = self.reaction_name

        if self.reaction == "lithium metal plating":
            # half-cell domain
            variables = {
                "Lithium metal interface exchange current density [A.m-2]": j0,
            }
            return variables

        # Size average. For j0 variables that depend on particle size, see
        # "_get_standard_size_distribution_exchange_current_variables"
        if j0.domain in [["negative particle size"], ["positive particle size"]]:
            j0 = pybamm.size_average(j0)
        # Average, and broadcast if necessary
        j0_av = pybamm.x_average(j0)

        # X-average, and broadcast if necessary
        if j0.domain == []:
            j0 = pybamm.FullBroadcast(j0, f"{domain} electrode", "current collector")
        elif j0.domain == ["current collector"]:
            j0 = pybamm.PrimaryBroadcast(j0, f"{domain} electrode")

        variables = {
            f"{Domain} electrode {reaction_name}"
            "exchange current density [A.m-2]": j0,
            f"X-averaged {domain} electrode {reaction_name}"
            "exchange current density [A.m-2]": j0_av,
        }

        return variables

    def _get_standard_volumetric_current_density_variables(self, variables):
        domain, Domain = self.domain_Domain

        if self.options.electrode_types[domain] == "planar":
            return variables

        reaction_name = self.reaction_name
        phase_name = self.phase_name

        if isinstance(self, pybamm.kinetics.NoReaction):
            a = 0
        else:
            a = variables[
                f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]"
            ]

        j = variables[
            f"{Domain} electrode {reaction_name}interfacial current density [A.m-2]"
        ]
        a_j = a * j
        a_j_av = pybamm.x_average(a_j)

        if reaction_name == "SEI on cracks ":
            roughness = variables[f"{Domain} electrode roughness ratio"] - 1
            roughness_av = (
                variables[f"X-averaged {domain} electrode roughness ratio"] - 1
            )
        else:
            roughness = 1
            roughness_av = 1

        variables.update(
            {
                f"{Domain} electrode {reaction_name}volumetric "
                "interfacial current density [A.m-3]": a_j * roughness,
                f"X-averaged {domain} electrode {reaction_name}volumetric interfacial "
                "current density [A.m-3]": a_j_av * roughness_av,
            }
        )
        return variables

    def _get_standard_overpotential_variables(self, eta_r):
        domain, Domain = self.domain_Domain
        reaction_name = self.reaction_name

        if self.reaction == "lithium metal plating":
            # half-cell domain
            variables = {"Lithium metal interface reaction overpotential [V]": eta_r}
            return variables

        # Size average. For eta_r variables that depend on particle size, see
        # "_get_standard_size_distribution_overpotential_variables"
        if eta_r.domain in [["negative particle size"], ["positive particle size"]]:
            eta_r = pybamm.size_average(eta_r)

        # X-average, and broadcast if necessary
        eta_r_av = pybamm.x_average(eta_r)
        if eta_r.domain == ["current collector"]:
            eta_r = pybamm.PrimaryBroadcast(eta_r, f"{domain} electrode")

        variables = {
            f"{Domain} electrode {reaction_name}reaction overpotential [V]": eta_r,
            f"X-averaged {domain} electrode {reaction_name}reaction "
            "overpotential [V]": eta_r_av,
        }

        return variables

    def _get_standard_sei_film_overpotential_variables(self, eta_sei):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        Phase_name = phase_name.capitalize()

        if self.options.electrode_types[domain] == "planar":
            # half-cell domain
            variables = {
                f"{Domain} electrode {Phase_name}SEI film overpotential [V]": eta_sei,
            }
            return variables

        # Average, and broadcast if necessary
        eta_sei_av = pybamm.x_average(eta_sei)
        if eta_sei.domain == []:
            eta_sei = pybamm.FullBroadcast(
                eta_sei, f"{domain} electrode", "current collector"
            )
        elif eta_sei.domain == ["current collector"]:
            eta_sei = pybamm.PrimaryBroadcast(eta_sei, f"{domain} electrode")

        variables = {
            f"{Domain} electrode {phase_name}SEI film overpotential [V]": eta_sei,
            f"X-averaged {domain} electrode {phase_name}SEI"
            " film overpotential [V]": eta_sei_av,
        }

        return variables

    def _get_standard_average_surface_potential_difference_variables(
        self, delta_phi_av
    ):
        domain = self.domain

        if self.options.electrode_types[domain] == "planar":
            variables = {
                "Lithium metal interface surface potential difference [V]"
                "": delta_phi_av,
            }
        else:
            variables = {
                f"X-averaged {domain} electrode "
                "surface potential difference [V]": delta_phi_av,
            }

        return variables

    def _get_standard_size_distribution_interfacial_current_variables(self, j):
        """
        Interfacial current density variables that depend on particle size R,
        relevant if "particle size" option is "distribution".
        """
        domain, Domain = self.domain_Domain
        reaction_name = self.reaction_name

        # X-average and broadcast if necessary
        if j.domains["secondary"] == [f"{domain} electrode"]:
            # x-average
            j_xav = pybamm.x_average(j)
        else:
            j_xav = j
            j = pybamm.SecondaryBroadcast(j_xav, [f"{domain} electrode"])

        variables = {
            f"{Domain} electrode {reaction_name}"
            "interfacial current density distribution [A.m-2]": j,
            f"X-averaged {domain} electrode {reaction_name}"
            "interfacial current density distribution [A.m-2]": j_xav,
        }

        return variables

    def _get_standard_size_distribution_exchange_current_variables(self, j0):
        """
        Exchange current variables that depend on particle size.
        """
        domain, Domain = self.domain_Domain
        reaction_name = self.reaction_name

        # X-average or broadcast to electrode if necessary
        if j0.domains["secondary"] != [f"{domain} electrode"]:
            j0_av = j0
            j0 = pybamm.SecondaryBroadcast(j0, f"{domain} electrode")
        else:
            j0_av = pybamm.x_average(j0)

        variables = {
            f"{Domain} electrode {reaction_name}"
            "exchange current density distribution [A.m-2]": j0,
            f"X-averaged {domain} electrode {reaction_name}"
            "exchange current density distribution [A.m-2]": j0_av,
        }

        return variables

    def _get_standard_size_distribution_overpotential_variables(self, eta_r):
        """
        Overpotential variables that depend on particle size.
        """
        domain, Domain = self.domain_Domain
        reaction_name = self.reaction_name

        # X-average or broadcast to electrode if necessary
        if eta_r.domains["secondary"] != [f"{domain} electrode"]:
            eta_r_av = eta_r
            eta_r = pybamm.SecondaryBroadcast(eta_r, f"{domain} electrode")
        else:
            eta_r_av = pybamm.x_average(eta_r)

        domain_reaction = f"{Domain} electrode {reaction_name}reaction overpotential"

        variables = {
            f"{domain_reaction} [V]": eta_r,
            f"X-averaged {domain_reaction.lower()} distribution [V]": eta_r_av,
        }

        return variables
