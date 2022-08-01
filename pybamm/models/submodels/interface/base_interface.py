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

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain, reaction, options=None):
        super().__init__(param, domain, options=options)
        if reaction in ["lithium-ion main", "lithium metal plating"]:
            self.reaction_name = ""  # empty reaction name for the main reaction
            self.Reaction_icd = "Interfacial current density"
        elif reaction == "lead-acid main":
            self.reaction_name = ""  # empty reaction name for the main reaction
            self.Reaction_icd = "Interfacial current density"
        elif reaction == "lead-acid oxygen":
            self.reaction_name = " oxygen"
            self.Reaction_icd = "Oxygen interfacial current density"
        elif reaction == "SEI":
            self.reaction_name = " SEI"
            self.Reaction_icd = "SEI interfacial current density"
        elif reaction == "SEI on cracks":
            self.reaction_name = " SEI on cracks"
            self.Reaction_icd = "SEI on cracks interfacial current density"
        elif reaction == "lithium plating":
            self.reaction_name = " lithium plating"
            self.Reaction_icd = "Lithium plating interfacial current density"
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
        domain_param = self.domain_param

        c_e = variables[self.domain + " electrolyte concentration"]
        T = variables[self.domain + " electrode temperature"]

        if self.reaction == "lithium-ion main":
            # For "particle-size distribution" submodels, take distribution version
            # of c_s_surf that depends on particle size.
            if self.options["particle size"] == "distribution":
                c_s_surf = variables[
                    self.domain + " particle surface concentration distribution"
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
                c_e = pybamm.PrimaryBroadcast(
                    c_e, [self.domain.lower() + " particle size"]
                )
                T = pybamm.PrimaryBroadcast(T, [self.domain.lower() + " particle size"])

            else:
                c_s_surf = variables[self.domain + " particle surface concentration"]

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

            j0 = (
                domain_param.gamma
                * domain_param.j0(c_e, c_s_surf, T)
                / domain_param.C_r
            )

        elif self.reaction == "lithium metal plating":
            j0 = param.j0_plating(c_e, 1, T)

        elif self.reaction == "lead-acid main":
            # If variable was broadcast, take only the orphan
            if isinstance(c_e, pybamm.Broadcast) and isinstance(T, pybamm.Broadcast):
                c_e = c_e.orphans[0]
                T = T.orphans[0]
            j0 = domain_param.j0(c_e, T)

        elif self.reaction == "lead-acid oxygen":
            # If variable was broadcast, take only the orphan
            if isinstance(c_e, pybamm.Broadcast) and isinstance(T, pybamm.Broadcast):
                c_e = c_e.orphans[0]
                T = T.orphans[0]
            if self.domain == "Negative":
                j0 = pybamm.Scalar(0)
            elif self.domain == "Positive":
                j0 = param.p.j0_Ox(c_e, T)

        return j0

    def _get_number_of_electrons_in_reaction(self):
        """Returns the number of electrons in the reaction."""
        if self.reaction in [
            "lead-acid main",
            "lithium-ion main",
            "lithium metal plating",
        ]:
            return self.domain_param.ne
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

        i_boundary_cc = variables["Current collector current density"]

        if self.half_cell and self.domain == "Negative":
            # In a half-cell the total interfacial current density is the current
            # collector current density, not divided by electrode thickness
            i_boundary_cc = variables["Current collector current density"]
            j_total_average = i_boundary_cc
        else:
            a_av = variables[
                "X-averaged "
                + self.domain.lower()
                + " electrode surface area to volume ratio"
            ]
            sgn = 1 if self.domain == "Negative" else -1

            j_total_average = sgn * i_boundary_cc / (a_av * self.domain_param.l)

        return j_total_average

    def _get_standard_interfacial_current_variables(self, j):
        j_scale = self.domain_param.j_scale

        if self.reaction == "lithium metal plating":
            # Half-cell domain, j should not be broadcast
            variables = {
                "Lithium metal plating current density": j,
                "Lithium metal plating current density [A.m-2]": j_scale * j,
            }
            return variables

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
            self.domain
            + " electrode"
            + self.reaction_name
            + " interfacial current density": j,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " interfacial current density": j_av,
            self.domain
            + " electrode"
            + self.reaction_name
            + " interfacial current density [A.m-2]": j_scale * j,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " interfacial current density [A.m-2]": j_scale * j_av,
        }

        return variables

    def _get_standard_total_interfacial_current_variables(self, j_tot_av):

        j_scale = self.domain_param.j_scale

        if self.half_cell and self.domain == "Negative":
            variables = {
                "Lithium metal total interfacial current density": j_tot_av,
                "Lithium metal total interfacial current density [A.m-2]": j_scale
                * j_tot_av,
            }
        else:
            variables = {
                "X-averaged "
                + self.domain.lower()
                + " electrode total interfacial current density": j_tot_av,
                "X-averaged "
                + self.domain.lower()
                + " electrode total interfacial current density [A.m-2]": j_scale
                * j_tot_av,
            }

        return variables

    def _get_standard_exchange_current_variables(self, j0):
        j_scale = self.domain_param.j_scale

        if self.reaction == "lithium metal plating":
            # half-cell domain
            variables = {
                "Lithium metal interface exchange current density": j0,
                "Lithium metal interface exchange current density [A.m-2]": j_scale
                * j0,
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
            j0 = pybamm.FullBroadcast(
                j0, self.domain_for_broadcast, "current collector"
            )
        elif j0.domain == ["current collector"]:
            j0 = pybamm.PrimaryBroadcast(j0, self.domain_for_broadcast)

        variables = {
            self.domain
            + " electrode"
            + self.reaction_name
            + " exchange current density": j0,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " exchange current density": j0_av,
            self.domain
            + " electrode"
            + self.reaction_name
            + " exchange current density [A.m-2]": j_scale * j0,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " exchange current density [A.m-2]": j_scale * j0_av,
        }

        return variables

    def _get_standard_volumetric_current_density_variables(self, variables):
        if self.half_cell and self.domain == "Negative":
            return variables

        Domain = self.domain
        domain = Domain.lower()
        reaction_name = self.reaction_name

        a = variables[f"{Domain} electrode surface area to volume ratio"]
        a_av = variables[f"X-averaged {domain} electrode surface area to volume ratio"]
        j = variables[f"{Domain} electrode{reaction_name} interfacial current density"]
        j_av = variables[
            f"X-averaged {domain} electrode{reaction_name} interfacial current density"
        ]
        scale = self.param.i_typ / self.param.L_x

        variables.update(
            {
                f"{Domain} electrode{reaction_name} volumetric "
                "interfacial current density": a * j,
                f"X-averaged {domain} electrode{reaction_name} volumetric "
                "interfacial current density": a_av * j_av,
                f"{Domain} electrode{reaction_name} volumetric "
                "interfacial current density [A.m-3]": scale * a * j,
                f"X-averaged {domain} electrode{reaction_name} volumetric "
                "interfacial current density [A.m-3]": scale * a_av * j_av,
            }
        )
        return variables

    def _get_standard_overpotential_variables(self, eta_r):

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

        domain_reaction = (
            self.domain + " electrode" + self.reaction_name + " reaction overpotential"
        )

        variables = {
            domain_reaction: eta_r,
            "X-averaged " + domain_reaction.lower(): eta_r_av,
            domain_reaction + " [V]": eta_r * pot_scale,
            "X-averaged " + domain_reaction.lower() + " [V]": eta_r_av * pot_scale,
        }

        return variables

    def _get_standard_sei_film_overpotential_variables(self, eta_sei):

        pot_scale = self.param.potential_scale

        if self.half_cell:
            # half-cell domain
            variables = {
                "SEI film overpotential": eta_sei,
                "SEI film overpotential [V]": eta_sei * pot_scale,
            }
            return variables

        # Average, and broadcast if necessary
        eta_sei_av = pybamm.x_average(eta_sei)
        if eta_sei.domain == []:
            eta_sei = pybamm.FullBroadcast(
                eta_sei, self.domain_for_broadcast, "current collector"
            )
        elif eta_sei.domain == ["current collector"]:
            eta_sei = pybamm.PrimaryBroadcast(eta_sei, self.domain_for_broadcast)

        variables = {
            "SEI film overpotential": eta_sei,
            "X-averaged SEI film overpotential": eta_sei_av,
            "SEI film overpotential [V]": eta_sei * pot_scale,
            "X-averaged SEI film overpotential [V]": eta_sei_av * pot_scale,
        }

        return variables

    def _get_standard_average_surface_potential_difference_variables(
        self, delta_phi_av
    ):
        ocp_ref = self.domain_param.U_ref

        delta_phi_av_dim = ocp_ref + delta_phi_av * self.param.potential_scale

        if self.half_cell and self.domain == "Negative":
            variables = {
                "Lithium metal interface surface potential difference": delta_phi_av,
                "Lithium metal interface surface potential difference [V]"
                "": delta_phi_av_dim,
            }
        else:
            variables = {
                "X-averaged "
                + self.domain.lower()
                + " electrode surface potential difference": delta_phi_av,
                "X-averaged "
                + self.domain.lower()
                + " electrode surface potential difference [V]": delta_phi_av_dim,
            }

        return variables

    def _get_standard_surface_potential_difference_variables(self, delta_phi):

        ocp_ref = self.domain_param.U_ref
        pot_scale = self.param.potential_scale

        # Broadcast if necessary
        delta_phi_dim = ocp_ref + delta_phi * pot_scale
        if delta_phi.domain == ["current collector"]:
            delta_phi = pybamm.PrimaryBroadcast(delta_phi, self.domain_for_broadcast)
            delta_phi_dim = pybamm.PrimaryBroadcast(
                delta_phi_dim, self.domain_for_broadcast
            )

        variables = {
            self.domain + " electrode surface potential difference": delta_phi,
            self.domain + " electrode surface potential difference [V]": delta_phi_dim,
        }

        return variables

    def _get_standard_size_distribution_interfacial_current_variables(self, j):
        """
        Interfacial current density variables that depend on particle size R,
        relevant if "particle size" option is "distribution".
        """
        # X-average and broadcast if necessary
        if j.domains["secondary"] == [self.domain.lower() + " electrode"]:
            # x-average
            j_xav = pybamm.x_average(j)
        else:
            j_xav = j
            j = pybamm.SecondaryBroadcast(j_xav, [self.domain.lower() + " electrode"])

        # j scale
        j_scale = self.domain_param.j_scale

        variables = {
            self.domain
            + " electrode"
            + self.reaction_name
            + " interfacial current density distribution": j,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " interfacial current density distribution": j_xav,
            self.domain
            + " electrode"
            + self.reaction_name
            + " interfacial current density"
            + " distribution [A.m-2]": j_scale * j,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " interfacial current density"
            + " distribution [A.m-2]": j_scale * j_xav,
        }

        return variables

    def _get_standard_size_distribution_exchange_current_variables(self, j0):
        """
        Exchange current variables that depend on particle size.
        """
        j_scale = self.domain_param.j_scale

        # X-average or broadcast to electrode if necessary
        if j0.domains["secondary"] != [self.domain.lower() + " electrode"]:
            j0_av = j0
            j0 = pybamm.SecondaryBroadcast(j0, self.domain_for_broadcast)
        else:
            j0_av = pybamm.x_average(j0)

        variables = {
            self.domain
            + " electrode"
            + self.reaction_name
            + " exchange current density distribution": j0,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " exchange current density distribution": j0_av,
            self.domain
            + " electrode"
            + self.reaction_name
            + " exchange current density distribution [A.m-2]": j_scale * j0,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " exchange current density distribution [A.m-2]": j_scale * j0_av,
        }

        return variables

    def _get_standard_size_distribution_overpotential_variables(self, eta_r):
        """
        Overpotential variables that depend on particle size.
        """
        pot_scale = self.param.potential_scale

        # X-average or broadcast to electrode if necessary
        if eta_r.domains["secondary"] != [self.domain.lower() + " electrode"]:
            eta_r_av = eta_r
            eta_r = pybamm.SecondaryBroadcast(eta_r, self.domain_for_broadcast)
        else:
            eta_r_av = pybamm.x_average(eta_r)

        domain_reaction = (
            self.domain + " electrode" + self.reaction_name + " reaction overpotential"
        )

        variables = {
            domain_reaction: eta_r,
            "X-averaged " + domain_reaction.lower() + " distribution": eta_r_av,
            domain_reaction + " [V]": eta_r * pot_scale,
            "X-averaged "
            + domain_reaction.lower()
            + " distribution [V]": eta_r_av * pot_scale,
        }

        return variables
