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
        elif reaction == "lithium-ion oxygen":
            self.reaction_name = " oxygen"
            self.Reaction_icd = "Oxygen interfacial current density"
        elif reaction == "SEI":
            self.reaction_name = " SEI"
            self.Reaction_icd = "SEI interfacial current density"
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

            if self.domain == "Negative":
                j0 = param.gamma_n * param.j0_n(c_e, c_s_surf, T) / param.C_r_n
            elif self.domain == "Positive":
                j0 = param.gamma_p * param.j0_p(c_e, c_s_surf, T) / param.C_r_p

        elif self.reaction == "lithium metal plating":
            j0 = param.j0_plating(c_e, 1, T)

        elif self.reaction == "lead-acid main":
            # If variable was broadcast, take only the orphan
            if isinstance(c_e, pybamm.Broadcast) and isinstance(T, pybamm.Broadcast):
                c_e = c_e.orphans[0]
                T = T.orphans[0]
            if self.domain == "Negative":
                j0 = param.j0_n(c_e, T)
            elif self.domain == "Positive":
                j0 = param.j0_p(c_e, T)

        elif self.reaction == "lead-acid oxygen":
            # If variable was broadcast, take only the orphan
            if isinstance(c_e, pybamm.Broadcast) and isinstance(T, pybamm.Broadcast):
                c_e = c_e.orphans[0]
                T = T.orphans[0]
            if self.domain == "Negative":
                j0 = pybamm.Scalar(0)
            elif self.domain == "Positive":
                j0 = param.j0_p_Ox(c_e, T)
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

        if self.reaction == "lithium-ion main":
            T = variables[self.domain + " electrode temperature"]
            # For "particle-size distribution" models, take distribution version
            # of c_s_surf that depends on particle size.
            if self.options["particle size"] == "distribution":
                c_s_surf = variables[
                    self.domain + " particle surface concentration distribution"
                ]
                # If variable was broadcast, take only the orphan
                if isinstance(c_s_surf, pybamm.Broadcast) and isinstance(
                    T, pybamm.Broadcast
                ):
                    c_s_surf = c_s_surf.orphans[0]
                    T = T.orphans[0]
                T = pybamm.PrimaryBroadcast(T, [self.domain.lower() + " particle size"])
            else:
                c_s_surf = variables[self.domain + " particle surface concentration"]

                # If variable was broadcast, take only the orphan
                if isinstance(c_s_surf, pybamm.Broadcast) and isinstance(
                    T, pybamm.Broadcast
                ):
                    c_s_surf = c_s_surf.orphans[0]
                    T = T.orphans[0]

            if self.domain == "Negative":
                ocp = self.param.U_n(c_s_surf, T)
                dUdT = self.param.dUdT_n(c_s_surf)
            elif self.domain == "Positive":
                ocp = self.param.U_p(c_s_surf, T)
                dUdT = self.param.dUdT_p(c_s_surf)
        elif self.reaction == "lithium metal plating":
            T = variables[self.domain + " electrode temperature"]
            ocp = self.param.U_n_ref
            dUdT = 0 * T
        elif self.reaction == "lead-acid main":
            c_e = variables[self.domain + " electrolyte concentration"]
            # If c_e was broadcast, take only the orphan
            if isinstance(c_e, pybamm.Broadcast):
                c_e = c_e.orphans[0]
            if self.domain == "Negative":
                ocp = self.param.U_n(c_e, self.param.T_init)
            elif self.domain == "Positive":
                ocp = self.param.U_p(c_e, self.param.T_init)
            dUdT = pybamm.Scalar(0)

        elif self.reaction == "lead-acid oxygen":
            if self.domain == "Negative":
                ocp = self.param.U_n_Ox
            elif self.domain == "Positive":
                ocp = self.param.U_p_Ox
            dUdT = pybamm.Scalar(0)

        else:
            ocp = pybamm.Scalar(0)
            dUdT = pybamm.Scalar(0)

        return ocp, dUdT

    def _get_number_of_electrons_in_reaction(self):
        """Returns the number of electrons in the reaction."""
        if self.reaction in [
            "lead-acid main",
            "lithium-ion main",
            "lithium metal plating",
        ]:
            if self.domain == "Negative":
                return self.param.ne_n
            elif self.domain == "Positive":
                return self.param.ne_p
        elif self.reaction == "lead-acid oxygen":
            return self.param.ne_Ox
        else:
            return pybamm.Scalar(0)

    def _get_electrolyte_reaction_signed_stoichiometry(self):
        """Returns the number of electrons in the reaction."""
        if self.reaction in [
            "lithium-ion main",
            "SEI",
            "lithium plating",
            "lithium metal plating",
        ]:
            # Both the main reaction current contribute to the electrolyte reaction
            # current
            return pybamm.Scalar(1), pybamm.Scalar(1)
        elif self.reaction == "lead-acid main":
            return self.param.s_plus_n_S, self.param.s_plus_p_S
        elif self.reaction == "lead-acid oxygen":
            return self.param.s_plus_Ox, self.param.s_plus_Ox
        else:
            return pybamm.Scalar(0), pybamm.Scalar(0)

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

            if self.domain == "Negative":
                j_total_average = i_boundary_cc / (a_av * self.param.l_n)

            elif self.domain == "Positive":
                j_total_average = -i_boundary_cc / (a_av * self.param.l_p)

        return j_total_average

    def _get_standard_interfacial_current_variables(self, j):
        param = self.param
        if self.domain == "Negative":
            j_scale = param.j_scale_n
        elif self.domain == "Positive":
            j_scale = param.j_scale_p

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
            self.domain
            + " electrode"
            + self.reaction_name
            + " interfacial current density per volume [A.m-3]": i_typ / L_x * j,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " interfacial current density per volume [A.m-3]": i_typ / L_x * j_av,
        }

        return variables

    def _get_standard_total_interfacial_current_variables(self, j_tot_av):

        i_typ = self.param.i_typ
        L_x = self.param.L_x
        if self.domain == "Negative":
            j_scale = self.param.j_scale_n
        elif self.domain == "Positive":
            j_scale = self.param.j_scale_p

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
                "X-averaged " + self.domain.lower() + " electrode total interfacial "
                "current density per volume [A.m-3]": i_typ / L_x * j_tot_av,
            }

        return variables

    def _get_standard_whole_cell_interfacial_current_variables(self, variables):
        """
        Get variables associated with interfacial current over the whole cell domain
        This function also automatically increments the "total source term" variables
        """
        param = self.param

        i_typ = param.i_typ
        L_x = param.L_x
        j_n_scale = param.j_scale_n
        j_p_scale = param.j_scale_p

        j_p_av = variables[
            "X-averaged positive electrode"
            + self.reaction_name
            + " interfacial current density"
        ]

        zero_s = pybamm.FullBroadcast(0, "separator", "current collector")
        j_p = variables[
            "Positive electrode" + self.reaction_name + " interfacial current density"
        ]
        if self.half_cell:
            j = pybamm.concatenation(zero_s, j_p)
            j_dim = pybamm.concatenation(zero_s, j_p_scale * j_p)
        else:
            j_n_av = variables[
                "X-averaged negative electrode"
                + self.reaction_name
                + " interfacial current density"
            ]
            j_n = variables[
                "Negative electrode"
                + self.reaction_name
                + " interfacial current density"
            ]
            j = pybamm.concatenation(j_n, zero_s, j_p)
            j_dim = pybamm.concatenation(j_n_scale * j_n, zero_s, j_p_scale * j_p)

        # Create separate 'new_variables' so that variables only get updated once
        # everything is computed
        new_variables = variables.copy()
        if self.reaction not in ["SEI", "lithium plating"]:
            new_variables.update(
                {
                    self.Reaction_icd: j,
                    self.Reaction_icd + " [A.m-2]": j_dim,
                    self.Reaction_icd + " per volume [A.m-3]": i_typ / L_x * j,
                }
            )

        a_p = new_variables["Positive electrode surface area to volume ratio"]

        s_n, s_p = self._get_electrolyte_reaction_signed_stoichiometry()
        if self.half_cell:
            a_n = pybamm.Scalar(1)
            a = pybamm.concatenation(zero_s, a_p)
            s = pybamm.concatenation(
                zero_s,
                pybamm.FullBroadcast(s_p, "positive electrode", "current collector"),
            )
        else:
            a_n = new_variables["Negative electrode surface area to volume ratio"]
            a = pybamm.concatenation(a_n, zero_s, a_p)
            s = pybamm.concatenation(
                pybamm.FullBroadcast(s_n, "negative electrode", "current collector"),
                zero_s,
                pybamm.FullBroadcast(s_p, "positive electrode", "current collector"),
            )

        # Override print_name
        j.print_name = "J"
        a.print_name = "a"
        j_p.print_name = "j_p"

        new_variables["Sum of electrolyte reaction source terms"] += a * s * j
        new_variables[
            "Sum of positive electrode electrolyte reaction source terms"
        ] += (a_p * s_p * j_p)
        new_variables[
            "Sum of x-averaged positive electrode electrolyte reaction source terms"
        ] += pybamm.x_average(a_p * s_p * j_p)

        new_variables["Sum of interfacial current densities"] += j
        new_variables["Sum of positive electrode interfacial current densities"] += j_p
        new_variables[
            "Sum of x-averaged positive electrode interfacial current densities"
        ] += j_p_av

        if not self.half_cell:
            j_n.print_name = "j_n"
            new_variables[
                "Sum of negative electrode electrolyte reaction source terms"
            ] += (a_n * s_n * j_n)
            new_variables[
                "Sum of x-averaged negative electrode electrolyte reaction source terms"
            ] += pybamm.x_average(a_n * s_n * j_n)
            new_variables[
                "Sum of negative electrode interfacial current densities"
            ] += j_n
            new_variables[
                "Sum of x-averaged negative electrode interfacial current densities"
            ] += j_n_av

        variables.update(new_variables)
        return variables

    def _get_standard_exchange_current_variables(self, j0):
        param = self.param
        if self.domain == "Negative":
            j_scale = param.j_scale_n
        elif self.domain == "Positive":
            j_scale = param.j_scale_p

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
            self.domain
            + " electrode"
            + self.reaction_name
            + " exchange current density per volume [A.m-3]": i_typ / L_x * j0,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " exchange current density per volume [A.m-3]": i_typ / L_x * j0_av,
        }

        return variables

    def _get_standard_whole_cell_exchange_current_variables(self, variables):
        param = self.param
        i_typ = param.i_typ
        L_x = param.L_x
        j_n_scale = param.j_scale_n
        j_p_scale = param.j_scale_p

        zero_s = pybamm.FullBroadcast(0, "separator", "current collector")
        j0_p = variables[
            "Positive electrode" + self.reaction_name + " exchange current density"
        ]
        if self.half_cell:
            j0 = pybamm.concatenation(zero_s, j0_p)
            j0_dim = pybamm.concatenation(zero_s, j_p_scale * j0_p)
        else:
            j0_n = variables[
                "Negative electrode" + self.reaction_name + " exchange current density"
            ]
            j0 = pybamm.concatenation(j0_n, zero_s, j0_p)
            j0_dim = pybamm.concatenation(j_n_scale * j0_n, zero_s, j_p_scale * j0_p)

        if self.reaction_name == "":
            variables = {
                "Exchange current density": j0,
                "Exchange current density [A.m-2]": j0_dim,
                "Exchange current density per volume [A.m-3]": i_typ / L_x * j0,
            }
        else:
            reaction_name = self.reaction_name[1:].capitalize()
            variables = {
                reaction_name + " exchange current density": j0,
                reaction_name + " exchange current density [A.m-2]": j0_dim,
                reaction_name
                + " exchange current density per volume [A.m-3]": i_typ / L_x * j0,
            }

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
        if self.domain == "Negative":
            ocp_ref = self.param.U_n_ref
        elif self.domain == "Positive":
            ocp_ref = self.param.U_p_ref

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

        if self.domain == "Negative":
            ocp_ref = self.param.U_n_ref
        elif self.domain == "Positive":
            ocp_ref = self.param.U_p_ref
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
        if self.domain == "Negative":
            ocp_dim = self.param.U_n_ref + pot_scale * ocp
            ocp_av_dim = self.param.U_n_ref + pot_scale * ocp_av
        elif self.domain == "Positive":
            ocp_dim = self.param.U_p_ref + pot_scale * ocp
            ocp_av_dim = self.param.U_p_ref + pot_scale * ocp_av

        variables = {
            self.domain
            + " electrode"
            + self.reaction_name
            + " open circuit potential": ocp,
            self.domain
            + " electrode"
            + self.reaction_name
            + " open circuit potential [V]": ocp_dim,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " open circuit potential": ocp_av,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " open circuit potential [V]": ocp_av_dim,
        }
        if self.reaction in ["lithium-ion main", "lead-acid main"]:
            variables.update(
                {
                    self.domain + " electrode entropic change": dUdT,
                    self.domain
                    + " electrode entropic change [V.K-1]": pot_scale
                    * dUdT
                    / self.param.Delta_T,
                    "X-averaged "
                    + self.domain.lower()
                    + " electrode entropic change": dUdT_av,
                    "X-averaged "
                    + self.domain.lower()
                    + " electrode entropic change [V.K-1]": pot_scale
                    * dUdT_av
                    / self.param.Delta_T,
                }
            )

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
        i_typ = self.param.i_typ
        L_x = self.param.L_x
        if self.domain == "Negative":
            j_scale = i_typ / (self.param.a_n_typ * L_x)
        elif self.domain == "Positive":
            j_scale = i_typ / (self.param.a_p_typ * L_x)

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
        i_typ = self.param.i_typ
        L_x = self.param.L_x
        if self.domain == "Negative":
            j_scale = i_typ / (self.param.a_n_typ * L_x)
        elif self.domain == "Positive":
            j_scale = i_typ / (self.param.a_p_typ * L_x)

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
            self.domain
            + " electrode"
            + self.reaction_name
            + " exchange current density distribution"
            + " per volume [A.m-3]": i_typ / L_x * j0,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " exchange current density distribution"
            + " per volume [A.m-3]": i_typ / L_x * j0_av,
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

    def _get_standard_size_distribution_ocp_variables(self, ocp, dUdT):
        """
        A private function to obtain the open circuit potential and
        related standard variables when there is a distribution of particle sizes.
        """

        # X-average or broadcast to electrode if necessary
        if ocp.domains["secondary"] != [self.domain.lower() + " electrode"]:
            ocp_av = ocp
            ocp = pybamm.SecondaryBroadcast(ocp, self.domain_for_broadcast)
        else:
            ocp_av = pybamm.x_average(ocp)

        if dUdT.domains["secondary"] != [self.domain.lower() + " electrode"]:
            dUdT_av = dUdT
            dUdT = pybamm.SecondaryBroadcast(dUdT, self.domain_for_broadcast)
        else:
            dUdT_av = pybamm.x_average(dUdT)

        pot_scale = self.param.potential_scale
        if self.domain == "Negative":
            ocp_dim = self.param.U_n_ref + pot_scale * ocp
            ocp_av_dim = self.param.U_n_ref + pot_scale * ocp_av
        elif self.domain == "Positive":
            ocp_dim = self.param.U_p_ref + pot_scale * ocp
            ocp_av_dim = self.param.U_p_ref + pot_scale * ocp_av

        variables = {
            self.domain
            + " electrode"
            + self.reaction_name
            + " open circuit potential distribution": ocp,
            self.domain
            + " electrode"
            + self.reaction_name
            + " open circuit potential distribution [V]": ocp_dim,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " open circuit potential distribution": ocp_av,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " open circuit potential distribution [V]": ocp_av_dim,
        }
        if self.reaction_name == "":
            variables.update(
                {
                    self.domain
                    + " electrode entropic change"
                    + " (size-dependent)": dUdT,
                    self.domain
                    + " electrode entropic change"
                    + " (size-dependent) [V.K-1]": pot_scale
                    * dUdT
                    / self.param.Delta_T,
                    "X-averaged "
                    + self.domain.lower()
                    + " electrode entropic change"
                    + " (size-dependent)": dUdT_av,
                    "X-averaged "
                    + self.domain.lower()
                    + " electrode entropic change"
                    + " (size-dependent) [V.K-1]": pot_scale
                    * dUdT_av
                    / self.param.Delta_T,
                }
            )

        return variables
