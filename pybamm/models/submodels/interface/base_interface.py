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
    phase : str
        Phase of the particle

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain, reaction, options=None, phase="primary"):
        super().__init__(param, domain, options=options, phase=phase)
        if reaction in ["lithium-ion main", "lithium metal plating"]:
            self.reaction_name = self.phase_name
            # can be "" or "primary " or "secondary "
        elif reaction == "lead-acid main":
            self.reaction_name = ""  # empty reaction name for the main reaction
        elif reaction == "lead-acid oxygen":
            self.reaction_name = "oxygen "
        elif reaction == "lithium-ion oxygen":
            self.reaction_name = "oxygen "
        elif reaction == "SEI": # Jason-SEI for "primary" or "secondary"?
            self.reaction_name = "SEI "
            # print("Jason-1")
            # self.reaction_name = f"{self.phase_name}SEI " # primary SEI or secondary SEI
        elif reaction == "lithium plating":
            self.reaction_name = "lithium plating "

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
        Domain = self.domain
        domain = Domain.lower()
        phase_name = self.phase_name

        c_e = variables[f"{Domain} electrolyte concentration"]
        T = variables[f"{Domain} electrode temperature"]

        if isinstance(self, pybamm.kinetics.NoReaction):
            return pybamm.Scalar(0)
        elif self.reaction == "lithium-ion main":
            # For "particle-size distribution" submodels, take distribution version
            # of c_s_surf that depends on particle size.
            if self.options["particle size"] == "distribution":
                c_s_surf = variables[
                    f"{Domain} {phase_name}particle surface concentration distribution"
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
                    f"{Domain} {phase_name}particle surface concentration"
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

            tol = 1e-8
            c_e = pybamm.maximum(tol, c_e)
            c_s_surf = pybamm.maximum(tol, pybamm.minimum(c_s_surf, 1 - tol))

            j0 = phase_param.gamma * phase_param.j0(c_e, c_s_surf, T) / phase_param.C_r

        elif self.reaction == "lithium metal plating":
            j0 = param.j0_plating(c_e, 1, T)

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
            if self.domain == "Negative":
                j0 = pybamm.Scalar(0)
            elif self.domain == "Positive":
                j0 = param.p.prim.j0_Ox(c_e, T)
        else:
            j0 = pybamm.Scalar(0)

        return j0

    def _get_open_circuit_potential(self, variables):
        """
        A private function to obtain the open circuit potential and entropic change

        Parameters
        ----------
        variables: dict
            The variables in the full model.
        Returns
        -------
        ocp : :class:`pybamm.Symbol`
            The open-circuit potential
        dUdT : :class:`pybamm.Symbol`
            The entropic change in open-circuit potential due to temperature

        """
        Domain = self.domain
        domain = Domain.lower()
        phase_name = self.phase_name

        if isinstance(self, pybamm.kinetics.NoReaction):
            ocp = pybamm.Scalar(0)
            dUdT = pybamm.Scalar(0)
        elif self.reaction == "lithium-ion main":

            T = variables[f"{Domain} electrode temperature"]
            # For "particle-size distribution" models, take distribution version
            # of c_s_surf that depends on particle size.
            if self.options["particle size"] == "distribution":
                c_s_surf = variables[
                    f"{Domain} {phase_name}particle surface concentration distribution"
                ]
                # If variable was broadcast, take only the orphan
                if isinstance(c_s_surf, pybamm.Broadcast) and isinstance(
                    T, pybamm.Broadcast
                ):
                    c_s_surf = c_s_surf.orphans[0]
                    T = T.orphans[0]
                T = pybamm.PrimaryBroadcast(T, [f"{domain} particle size"])
            else:
                c_s_surf = variables[
                    f"{Domain} {phase_name}particle surface concentration"
                ]
                # If variable was broadcast, take only the orphan
                if isinstance(c_s_surf, pybamm.Broadcast) and isinstance(
                    T, pybamm.Broadcast
                ):
                    c_s_surf = c_s_surf.orphans[0]
                    T = T.orphans[0]

            ocp = self.phase_param.U(c_s_surf, T)
            dUdT = self.phase_param.dUdT(c_s_surf)
        elif self.reaction == "lithium metal plating":
            T = variables[f"{Domain} electrode temperature"]
            ocp = self.param.n.prim.U_ref
            dUdT = 0 * T
        elif self.reaction == "lead-acid main":
            c_e = variables[f"{Domain} electrolyte concentration"]
            # If c_e was broadcast, take only the orphan
            if isinstance(c_e, pybamm.Broadcast):
                c_e = c_e.orphans[0]
            ocp = self.phase_param.U(c_e, self.param.T_init)
            dUdT = pybamm.Scalar(0)

        elif self.reaction == "lead-acid oxygen":
            ocp = self.domain_param.U_Ox
            dUdT = pybamm.Scalar(0)

        else:
            ocp = pybamm.Scalar(0)
            dUdT = pybamm.Scalar(0)

        return ocp, dUdT

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
        else:
            return pybamm.Scalar(0)

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
        domain = self.domain.lower()
        i_boundary_cc = variables["Current collector current density"]

        if self.half_cell and self.domain == "Negative":
            # In a half-cell the total interfacial current density is the current
            # collector current density, not divided by electrode thickness
            i_boundary_cc = variables["Current collector current density"]
            j_total_average = i_boundary_cc
        else:
            a_av = variables[
                f"X-averaged {domain} electrode {self.phase_name}"
                "surface area to volume ratio"
            ]
            sgn = 1 if self.domain == "Negative" else -1

            j_total_average = sgn * i_boundary_cc / (a_av * self.domain_param.l)

        return j_total_average

    def _get_standard_interfacial_current_variables(self, j):
        Domain = self.domain
        domain = Domain.lower()
        reaction_name = self.reaction_name
        param = self.param
        j_scale = self.phase_param.j_scale

        if self.reaction == "lithium metal plating":
            # Half-cell domain, j should not be broadcast
            variables = {
                "Lithium metal plating current density": j,
                "Lithium metal plating current density [A.m-2]": j_scale * j,
            }
            return variables

        i_typ = param.i_typ
        L_x = param.L_x

        # Size average. For j variables that depend on particle size, see
        # "_get_standard_size_distribution_interfacial_current_variables"
        if j.domain in [["negative particle size"], ["positive particle size"]]:
            j = pybamm.size_average(j)
        # Average, and broadcast if necessary
        j_av = pybamm.x_average(j)
        if j.domain == []:
            j = pybamm.FullBroadcast(j, self.domain_for_broadcast, "current collector")
        elif j.domain == ["current collector"]:
            j = pybamm.PrimaryBroadcast(j, self.domain_for_broadcast)

        variables = {
            f"{Domain} electrode {reaction_name}interfacial current density": j,
            f"X-averaged {domain} electrode {reaction_name}"
            "interfacial current density": j_av,
            f"{Domain} electrode {reaction_name}"
            "interfacial current density [A.m-2]": j_scale * j,
            f"X-averaged {domain} electrode {reaction_name}"
            "interfacial current density [A.m-2]": j_scale * j_av,
            f"{Domain} electrode {reaction_name}"
            "interfacial current density per volume [A.m-3]": i_typ / L_x * j,
            f"X-averaged {domain} electrode {reaction_name}"
            "interfacial current density per volume [A.m-3]": i_typ / L_x * j_av,
        } # Jason - reaction_name should be expanded with "primary SEI" and "secondary SEI"
        # print(f"{reaction_name}")
        return variables

    def _get_standard_total_interfacial_current_variables(self, j_tot_av):
        domain = self.domain.lower()

        i_typ = self.param.i_typ
        L_x = self.param.L_x
        j_scale = self.phase_param.j_scale

        if self.half_cell and self.domain == "Negative":
            variables = {
                "Lithium metal total interfacial current density": j_tot_av,
                "Lithium metal total interfacial current density [A.m-2]": j_scale
                * j_tot_av,
            }
        else:
            variables = {
                f"X-averaged {domain} electrode total interfacial "
                "current density": j_tot_av,
                f"X-averaged {domain} electrode total interfacial "
                "current density [A.m-2]": j_scale * j_tot_av,
                f"X-averaged {domain} electrode total interfacial "
                "current density per volume [A.m-3]": i_typ / L_x * j_tot_av,
            }

        return variables

    def _get_standard_exchange_current_variables(self, j0):
        Domain = self.domain
        domain = Domain.lower()
        reaction_name = self.reaction_name
        param = self.param
        j_scale = self.phase_param.j_scale

        if self.reaction == "lithium metal plating":
            # half-cell domain
            variables = {
                "Lithium metal interface exchange current density": j0,
                "Lithium metal interface exchange current density [A.m-2]": j_scale
                * j0,
            }
            return variables

        i_typ = param.i_typ
        L_x = param.L_x
        # Size average. For j0 variables that depend on particle size, see
        # "_get_standard_size_distribution_exchange_current_variables"
        if j0.domain in [["negative particle size"], ["positive particle size"]]:
            j0 = pybamm.size_average(j0)
        # Average, and broadcast if necessary
        j0_av = pybamm.x_average(j0)

        # X-average, and broadcast if necessary
        if j0.domain == []:
            j0 = pybamm.FullBroadcast(
                j0, self.domain_for_broadcast, "current collector"
            )
        elif j0.domain == ["current collector"]:
            j0 = pybamm.PrimaryBroadcast(j0, self.domain_for_broadcast)

        variables = {
            f"{Domain} electrode {reaction_name}" "exchange current density": j0,
            f"X-averaged {domain} electrode {reaction_name}"
            "exchange current density": j0_av,
            f"{Domain} electrode {reaction_name}"
            "exchange current density [A.m-2]": j_scale * j0,
            f"X-averaged {domain} electrode {reaction_name}"
            "exchange current density [A.m-2]": j_scale * j0_av,
            f"{Domain} electrode {reaction_name}"
            "exchange current density per volume [A.m-3]": i_typ / L_x * j0,
            f"X-averaged {domain} electrode {reaction_name}"
            "exchange current density per volume [A.m-3]": i_typ / L_x * j0_av,
        }
        # print(f"Jason - X-averaged {domain} electrode {reaction_name}exchange current density per volume [A.m-3]")
        return variables

    def _get_standard_overpotential_variables(self, eta_r):
        Domain = self.domain
        reaction_name = self.reaction_name
        pot_scale = self.param.potential_scale

        if self.reaction == "lithium metal plating":
            # half-cell domain
            variables = {
                "Lithium metal interface reaction overpotential": eta_r,
                "Lithium metal interface reaction overpotential [V]": pot_scale * eta_r,
            }
            return variables

        # Size average. For eta_r variables that depend on particle size, see
        # "_get_standard_size_distribution_overpotential_variables"
        if eta_r.domain in [["negative particle size"], ["positive particle size"]]:
            eta_r = pybamm.size_average(eta_r)

        # X-average, and broadcast if necessary
        eta_r_av = pybamm.x_average(eta_r)
        if eta_r.domain == ["current collector"]:
            eta_r = pybamm.PrimaryBroadcast(eta_r, self.domain_for_broadcast)

        domain_reaction = f"{Domain} electrode {reaction_name}reaction overpotential"

        variables = {
            domain_reaction: eta_r,
            f"X-averaged {domain_reaction.lower()}": eta_r_av,
            f"{domain_reaction} [V]": eta_r * pot_scale,
            f"X-averaged {domain_reaction.lower()} [V]": eta_r_av * pot_scale,
        }

        return variables

    def _get_standard_sei_film_overpotential_variables(self, eta_sei):

        phase_name = self.phase_name
        pot_scale = self.param.potential_scale
        pref = phase_name.capitalize()

        if self.half_cell:
            # half-cell domain
            variables = {
                f"{pref}SEI film overpotential": eta_sei,
                f"{pref}SEI film overpotential [V]": eta_sei * pot_scale,
                # f"{phase_name}SEI film overpotential": eta_sei,
                # f"{phase_name}SEI film overpotential [V]": eta_sei * pot_scale,
            }
            return variables # Jason-does these variables modified properly

        # Average, and broadcast if necessary
        eta_sei_av = pybamm.x_average(eta_sei)
        if eta_sei.domain == []:
            eta_sei = pybamm.FullBroadcast(
                eta_sei, self.domain_for_broadcast, "current collector"
            )
        elif eta_sei.domain == ["current collector"]:
            eta_sei = pybamm.PrimaryBroadcast(eta_sei, self.domain_for_broadcast)

        variables = {
            f"{pref}SEI film overpotential": eta_sei,
            # f"{phase_name}SEI film overpotential": eta_sei,
            f"X-averaged {phase_name}SEI film overpotential": eta_sei_av,
            f"{pref}SEI film overpotential [V]": eta_sei * pot_scale,
            # f"{phase_name}SEI film overpotential [V]": eta_sei * pot_scale,
            f"X-averaged {phase_name}SEI film overpotential [V]": eta_sei_av * pot_scale,
        }

        return variables

    def _get_standard_average_surface_potential_difference_variables(
        self, delta_phi_av
    ):
        domain = self.domain.lower()

        ocp_ref = self.phase_param.U_ref

        delta_phi_av_dim = ocp_ref + delta_phi_av * self.param.potential_scale

        if self.half_cell and self.domain == "Negative":
            variables = {
                "Lithium metal interface surface potential difference": delta_phi_av,
                "Lithium metal interface surface potential difference [V]"
                "": delta_phi_av_dim,
            }
        else:
            variables = {
                f"X-averaged {domain} electrode "
                "surface potential difference": delta_phi_av,
                f"X-averaged {domain} electrode "
                "surface potential difference [V]": delta_phi_av_dim,
            } # Jason - need {phase_name} here?

        return variables

    def _get_standard_surface_potential_difference_variables(self, delta_phi):

        ocp_ref = self.phase_param.U_ref

        # Broadcast if necessary
        delta_phi_dim = ocp_ref + delta_phi * self.param.potential_scale
        if delta_phi.domain == ["current collector"]:
            delta_phi = pybamm.PrimaryBroadcast(delta_phi, self.domain_for_broadcast)
            delta_phi_dim = pybamm.PrimaryBroadcast(
                delta_phi_dim, self.domain_for_broadcast
            )

        variables = {
            f"{self.domain} electrode surface potential difference": delta_phi,
            f"{self.domain} electrode surface potential difference [V]": delta_phi_dim,
        }

        return variables

    def _get_standard_ocp_variables(self, ocp, dUdT):
        """
        A private function to obtain the open circuit potential and
        related standard variables.

        Parameters
        ----------
        ocp : :class:`pybamm.Symbol`
            The open-circuit potential
        dUdT : :class:`pybamm.Symbol`
            The entropic change in ocp

        Returns
        -------
        variables : dict
            The variables dictionary including the open circuit potentials
            and related standard variables.
        """
        Domain = self.domain
        domain = Domain.lower()
        reaction_name = self.reaction_name

        # Size average. For ocp variables that depend on particle size, see
        # "_get_standard_size_distribution_ocp_variables"
        if ocp.domain in [["negative particle size"], ["positive particle size"]]:
            ocp = pybamm.size_average(ocp)
        if dUdT.domain in [["negative particle size"], ["positive particle size"]]:
            dUdT = pybamm.size_average(dUdT)

        # Average, and broadcast if necessary
        dUdT_av = pybamm.x_average(dUdT)
        ocp_av = pybamm.x_average(ocp)
        if self.half_cell and self.domain == "Negative":
            # Half-cell domain, ocp should not be broadcast
            pass
        elif ocp.domain == []:
            ocp = pybamm.FullBroadcast(
                ocp, self.domain_for_broadcast, "current collector"
            )
        elif ocp.domain == ["current collector"]:
            ocp = pybamm.PrimaryBroadcast(ocp, self.domain_for_broadcast)

        pot_scale = self.param.potential_scale
        ocp_dim = self.phase_param.U_ref + pot_scale * ocp
        ocp_av_dim = self.phase_param.U_ref + pot_scale * ocp_av

        variables = {
            f"{Domain} electrode {reaction_name}open circuit potential": ocp,
            f"{Domain} electrode {reaction_name}" "open circuit potential [V]": ocp_dim,
            f"X-averaged {domain} electrode {reaction_name}"
            "open circuit potential": ocp_av,
            f"X-averaged {domain} electrode {reaction_name}"
            "open circuit potential [V]": ocp_av_dim,
        }
        if self.reaction in ["lithium-ion main", "lead-acid main"]:
            variables.update(
                {
                    f"{Domain} electrode entropic change": dUdT,
                    f"{Domain} electrode entropic change [V.K-1]"
                    "": pot_scale * dUdT / self.param.Delta_T,
                    f"X-averaged {domain} electrode entropic change": dUdT_av,
                    f"X-averaged {domain} electrode entropic change [V.K-1]"
                    "": pot_scale * dUdT_av / self.param.Delta_T,
                }
            )# Jason - need {phase_name} here?

        return variables

    def _get_standard_size_distribution_interfacial_current_variables(self, j):
        """
        Interfacial current density variables that depend on particle size R,
        relevant if "particle size" option is "distribution".
        """
        Domain = self.domain
        domain = Domain.lower()
        reaction_name = self.reaction_name

        # X-average and broadcast if necessary
        if j.domains["secondary"] == [f"{domain} electrode"]:
            # x-average
            j_xav = pybamm.x_average(j)
        else:
            j_xav = j
            j = pybamm.SecondaryBroadcast(j_xav, [f"{domain} electrode"]) # Jason-what does j means here? phase_name?

        # j scale
        i_typ = self.param.i_typ
        L_x = self.param.L_x
        j_scale = i_typ / (self.phase_param.a_typ * L_x)

        variables = {
            f"{Domain} electrode {reaction_name}"
            "interfacial current density distribution": j,
            f"X-averaged {domain} electrode {reaction_name}"
            "interfacial current density distribution": j_xav,
            f"{Domain} electrode {reaction_name}"
            "interfacial current density distribution [A.m-2]": j_scale * j,
            f"X-averaged {domain} electrode {reaction_name}"
            "interfacial current density distribution [A.m-2]": j_scale * j_xav,
        }

        return variables

    def _get_standard_size_distribution_exchange_current_variables(self, j0):
        """
        Exchange current variables that depend on particle size.
        """
        Domain = self.domain
        domain = Domain.lower()
        reaction_name = self.reaction_name
        i_typ = self.param.i_typ
        L_x = self.param.L_x
        j_scale = i_typ / (self.phase_param.a_typ * L_x)

        # X-average or broadcast to electrode if necessary
        if j0.domains["secondary"] != [f"{domain} electrode"]:
            j0_av = j0
            j0 = pybamm.SecondaryBroadcast(j0, self.domain_for_broadcast)
        else:
            j0_av = pybamm.x_average(j0)

        variables = {
            f"{Domain} electrode {reaction_name}"
            "exchange current density distribution": j0,
            f"X-averaged {domain} electrode {reaction_name}"
            "exchange current density distribution": j0_av,
            f"{Domain} electrode {reaction_name}"
            "exchange current density distribution [A.m-2]": j_scale * j0,
            f"X-averaged {domain} electrode {reaction_name}"
            "exchange current density distribution [A.m-2]": j_scale * j0_av,
            f"{Domain} electrode {reaction_name}"
            "exchange current density distribution"
            + " per volume [A.m-3]": i_typ / L_x * j0,
            f"X-averaged {domain} electrode {reaction_name}"
            "exchange current density distribution"
            + " per volume [A.m-3]": i_typ / L_x * j0_av,
        }

        return variables

    def _get_standard_size_distribution_overpotential_variables(self, eta_r):
        """
        Overpotential variables that depend on particle size.
        """
        pot_scale = self.param.potential_scale
        Domain = self.domain
        domain = Domain.lower()
        reaction_name = self.reaction_name

        # X-average or broadcast to electrode if necessary
        if eta_r.domains["secondary"] != [f"{domain} electrode"]:
            eta_r_av = eta_r
            eta_r = pybamm.SecondaryBroadcast(eta_r, self.domain_for_broadcast)
        else:
            eta_r_av = pybamm.x_average(eta_r)

        domain_reaction = f"{Domain} electrode {reaction_name}reaction overpotential"

        variables = {
            domain_reaction: eta_r,
            f"X-averaged {domain_reaction.lower()} distribution": eta_r_av,
            f"{domain_reaction} [V]": eta_r * pot_scale,
            f"X-averaged {domain_reaction.lower()} distribution [V]": eta_r_av
            * pot_scale,
        }

        return variables

    def _get_standard_size_distribution_ocp_variables(self, ocp, dUdT):
        """
        A private function to obtain the open circuit potential and
        related standard variables when there is a distribution of particle sizes.
        """
        Domain = self.domain
        domain = Domain.lower()
        reaction_name = self.reaction_name

        # X-average or broadcast to electrode if necessary
        if ocp.domains["secondary"] != [f"{domain} electrode"]:
            ocp_av = ocp
            ocp = pybamm.SecondaryBroadcast(ocp, self.domain_for_broadcast)
        else:
            ocp_av = pybamm.x_average(ocp)

        if dUdT.domains["secondary"] != [f"{domain} electrode"]:
            dUdT_av = dUdT
            dUdT = pybamm.SecondaryBroadcast(dUdT, self.domain_for_broadcast)
        else:
            dUdT_av = pybamm.x_average(dUdT)

        pot_scale = self.param.potential_scale
        ocp_dim = self.phase_param.U_ref + pot_scale * ocp
        ocp_av_dim = self.phase_param.U_ref + pot_scale * ocp_av

        variables = {
            f"{Domain} electrode {reaction_name}"
            "open circuit potential distribution": ocp,
            f"{Domain} electrode {reaction_name}"
            "open circuit potential distribution [V]": ocp_dim,
            f"X-averaged {domain} electrode {reaction_name}"
            "open circuit potential distribution": ocp_av,
            f"X-averaged {domain} electrode {reaction_name}"
            "open circuit potential distribution [V]": ocp_av_dim,
        }
        if self.reaction_name == "":
            variables.update(
                {
                    f"{Domain} electrode entropic change (size-dependent)": dUdT,
                    f"{Domain} electrode entropic change "
                    "(size-dependent) [V.K-1]": pot_scale * dUdT / self.param.Delta_T,
                    f"X-averaged {domain} electrode entropic change "
                    "(size-dependent)": dUdT_av,
                    f"X-averaged {domain} electrode entropic change "
                    "(size-dependent) [V.K-1]": pot_scale
                    * dUdT_av
                    / self.param.Delta_T,
                }
            )

        return variables
