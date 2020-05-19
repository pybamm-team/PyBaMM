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

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain, reaction):
        super().__init__(param, domain)
        if reaction == "lithium-ion main":
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
        elif reaction == "sei":
            self.reaction_name = " sei"
            self.Reaction_icd = "Sei interfacial current density"
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
        c_e = variables[self.domain + " electrolyte concentration"]
        T = variables[self.domain + " electrode temperature"]

        if self.reaction == "lithium-ion main":
            c_s_surf = variables[self.domain + " particle surface concentration"]

            # If variable was broadcast, take only the orphan
            if (
                isinstance(c_s_surf, pybamm.Broadcast)
                and isinstance(c_e, pybamm.Broadcast)
                and isinstance(T, pybamm.Broadcast)
            ):
                c_s_surf = c_s_surf.orphans[0]
                c_e = c_e.orphans[0]
                T = T.orphans[0]
            if self.domain == "Negative":
                j0 = self.param.j0_n(c_e, c_s_surf, T) / self.param.C_r_n
            elif self.domain == "Positive":
                j0 = (
                    self.param.gamma_p
                    * self.param.j0_p(c_e, c_s_surf, T)
                    / self.param.C_r_p
                )

        elif self.reaction == "lead-acid main":
            # If variable was broadcast, take only the orphan
            if isinstance(c_e, pybamm.Broadcast) and isinstance(T, pybamm.Broadcast):
                c_e = c_e.orphans[0]
                T = T.orphans[0]
            if self.domain == "Negative":
                j0 = self.param.j0_n(c_e, T)
            elif self.domain == "Positive":
                j0 = self.param.j0_p(c_e, T)

        elif self.reaction == "lead-acid oxygen":
            # If variable was broadcast, take only the orphan
            if isinstance(c_e, pybamm.Broadcast) and isinstance(T, pybamm.Broadcast):
                c_e = c_e.orphans[0]
                T = T.orphans[0]
            if self.domain == "Negative":
                j0 = pybamm.Scalar(0)
            elif self.domain == "Positive":
                j0 = self.param.j0_p_Ox(c_e, T)
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
            c_s_surf = variables[self.domain + " particle surface concentration"]
            T = variables[self.domain + " electrode temperature"]

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
        "Returns the number of electrons in the reaction"
        if self.reaction in ["lead-acid main", "lithium-ion main"]:
            if self.domain == "Negative":
                return self.param.ne_n
            elif self.domain == "Positive":
                return self.param.ne_p
        elif self.reaction == "lead-acid oxygen":
            return self.param.ne_Ox
        else:
            return pybamm.Scalar(0)

    def _get_electrolyte_reaction_signed_stoichiometry(self):
        "Returns the number of electrons in the reaction"
        if self.reaction in ["lithium-ion main", "sei"]:
            # Both the main reaction current contribute to the electrolyte reaction
            # current
            return pybamm.Scalar(1), pybamm.Scalar(1)
        elif self.reaction == "lead-acid main":
            return self.param.s_plus_n_S, self.param.s_plus_p_S
        elif self.reaction == "lead-acid oxygen":
            return self.param.s_plus_Ox, self.param.s_plus_Ox
        else:
            return pybamm.Scalar(0), pybamm.Scalar(0)

    def _get_delta_phi(self, variables):
        "Calculate delta_phi, and derived variables, using phi_s and phi_e"
        phi_s = variables[self.domain + " electrode potential"]
        phi_e = variables[self.domain + " electrolyte potential"]
        delta_phi = phi_s - phi_e
        variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi)
        )
        return variables

    def _get_average_total_interfacial_current_density(self, variables):
        """
        Method to obtain the average total interfacial current density.
        """

        i_boundary_cc = variables["Current collector current density"]

        if self.domain == "Negative":
            j_total_average = i_boundary_cc / pybamm.geometric_parameters.l_n

        elif self.domain == "Positive":
            j_total_average = -i_boundary_cc / pybamm.geometric_parameters.l_p

        return j_total_average

    def _get_standard_interfacial_current_variables(self, j):

        i_typ = self.param.i_typ
        L_x = self.param.L_x
        if self.domain == "Negative":
            j_scale = i_typ / (self.param.a_n_dim * L_x)
        elif self.domain == "Positive":
            j_scale = i_typ / (self.param.a_p_dim * L_x)

        # Average, and broadcast if necessary
        if j.domain == []:
            j_av = j
            j = pybamm.FullBroadcast(j, self.domain_for_broadcast, "current collector")
        elif j.domain == ["current collector"]:
            j_av = j
            j = pybamm.PrimaryBroadcast(j, self.domain_for_broadcast)
        else:
            j_av = pybamm.x_average(j)

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
            j_scale = i_typ / (self.param.a_n_dim * L_x)
        elif self.domain == "Positive":
            j_scale = i_typ / (self.param.a_p_dim * L_x)

        variables = {
            "X-averaged "
            + self.domain.lower()
            + " electrode total interfacial current density": j_tot_av,
            "X-averaged "
            + self.domain.lower()
            + " electrode total interfacial current density [A.m-2]": j_scale
            * j_tot_av,
            "X-averaged "
            + self.domain.lower()
            + " electrode total interfacial current density per volume [A.m-3]": i_typ
            / L_x
            * j_tot_av,
        }

        return variables

    def _get_standard_whole_cell_interfacial_current_variables(self, variables):
        """
        Get variables associated with interfacial current over the whole cell domain
        This function also automatically increments the "total source term" variables
        """
        i_typ = self.param.i_typ
        L_x = self.param.L_x
        j_n_scale = i_typ / (self.param.a_n_dim * L_x)
        j_p_scale = i_typ / (self.param.a_p_dim * L_x)

        j_n_av = variables[
            "X-averaged negative electrode"
            + self.reaction_name
            + " interfacial current density"
        ]
        j_p_av = variables[
            "X-averaged positive electrode"
            + self.reaction_name
            + " interfacial current density"
        ]

        j_n = variables[
            "Negative electrode" + self.reaction_name + " interfacial current density"
        ]
        j_s = pybamm.FullBroadcast(0, "separator", "current collector")
        j_p = variables[
            "Positive electrode" + self.reaction_name + " interfacial current density"
        ]
        j = pybamm.Concatenation(j_n, j_s, j_p)
        j_dim = pybamm.Concatenation(j_n_scale * j_n, j_s, j_p_scale * j_p)

        variables.update(
            {
                self.Reaction_icd: j,
                self.Reaction_icd + " [A.m-2]": j_dim,
                self.Reaction_icd + " per volume [A.m-3]": i_typ / L_x * j,
            }
        )

        s_n, s_p = self._get_electrolyte_reaction_signed_stoichiometry()
        s = pybamm.Concatenation(
            pybamm.FullBroadcast(s_n, "negative electrode", "current collector"),
            pybamm.FullBroadcast(0, "separator", "current collector"),
            pybamm.FullBroadcast(s_p, "positive electrode", "current collector"),
        )
        variables["Sum of electrolyte reaction source terms"] += s * j
        variables["Sum of negative electrode electrolyte reaction source terms"] += (
            s_n * j_n
        )
        variables["Sum of positive electrode electrolyte reaction source terms"] += (
            s_p * j_p
        )
        variables[
            "Sum of x-averaged negative electrode electrolyte reaction source terms"
        ] += (s_n * j_n_av)
        variables[
            "Sum of x-averaged positive electrode electrolyte reaction source terms"
        ] += (s_p * j_p_av)

        variables["Sum of interfacial current densities"] += j
        variables["Sum of negative electrode interfacial current densities"] += j_n
        variables["Sum of positive electrode interfacial current densities"] += j_p
        variables[
            "Sum of x-averaged negative electrode interfacial current densities"
        ] += j_n_av
        variables[
            "Sum of x-averaged positive electrode interfacial current densities"
        ] += j_p_av

        return variables

    def _get_standard_exchange_current_variables(self, j0):

        i_typ = self.param.i_typ
        L_x = self.param.L_x
        if self.domain == "Negative":
            j_scale = i_typ / (self.param.a_n_dim * L_x)
        elif self.domain == "Positive":
            j_scale = i_typ / (self.param.a_p_dim * L_x)

        # Average, and broadcast if necessary
        if j0.domain == []:
            j0_av = j0
            j0 = pybamm.FullBroadcast(
                j0, self.domain_for_broadcast, "current collector"
            )
        elif j0.domain == ["current collector"]:
            j0_av = j0
            j0 = pybamm.PrimaryBroadcast(j0, self.domain_for_broadcast)
        else:
            j0_av = pybamm.x_average(j0)

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

        i_typ = self.param.i_typ
        L_x = self.param.L_x
        j_n_scale = i_typ / (self.param.a_n_dim * L_x)
        j_p_scale = i_typ / (self.param.a_p_dim * L_x)

        j0_n = variables[
            "Negative electrode" + self.reaction_name + " exchange current density"
        ]
        j0_s = pybamm.FullBroadcast(0, "separator", "current collector")
        j0_p = variables[
            "Positive electrode" + self.reaction_name + " exchange current density"
        ]
        j0 = pybamm.Concatenation(j0_n, j0_s, j0_p)
        j0_dim = pybamm.Concatenation(j_n_scale * j0_n, j0_s, j_p_scale * j0_p)

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
        # Average, and broadcast if necessary
        eta_r_av = pybamm.x_average(eta_r)
        if eta_r.domain == []:
            eta_r = pybamm.FullBroadcast(
                eta_r, self.domain_for_broadcast, "current collector"
            )
        elif eta_r.domain == ["current collector"]:
            eta_r = pybamm.PrimaryBroadcast(eta_r, self.domain_for_broadcast)

        variables = {
            self.domain
            + " electrode"
            + self.reaction_name
            + " reaction overpotential": eta_r,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " reaction overpotential": eta_r_av,
            self.domain
            + " electrode"
            + self.reaction_name
            + " reaction overpotential [V]": eta_r * pot_scale,
            "X-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " reaction overpotential [V]": eta_r_av * pot_scale,
        }

        return variables

    def _get_standard_sei_film_overpotential_variables(self, eta_sei):

        pot_scale = self.param.potential_scale
        # Average, and broadcast if necessary
        eta_sei_av = pybamm.x_average(eta_sei)
        if eta_sei.domain == []:
            eta_sei = pybamm.FullBroadcast(
                eta_sei, self.domain_for_broadcast, "current collector"
            )
        elif eta_sei.domain == ["current collector"]:
            eta_sei = pybamm.PrimaryBroadcast(eta_sei, self.domain_for_broadcast)

        domain = self.domain.lower() + " electrode"
        variables = {
            self.domain + " electrode sei film overpotential": eta_sei,
            "X-averaged " + domain + " sei film overpotential": eta_sei_av,
            self.domain + " electrode sei film overpotential [V]": eta_sei * pot_scale,
            "X-averaged "
            + domain
            + " sei film overpotential [V]": eta_sei_av * pot_scale,
        }

        return variables

    def _get_standard_surface_potential_difference_variables(self, delta_phi):

        if self.domain == "Negative":
            ocp_ref = self.param.U_n_ref
        elif self.domain == "Positive":
            ocp_ref = self.param.U_p_ref
        pot_scale = self.param.potential_scale

        # Average, and broadcast if necessary
        if delta_phi.domain == []:
            delta_phi_av = delta_phi
            delta_phi = pybamm.FullBroadcast(
                delta_phi, self.domain_for_broadcast, "current collector"
            )
        elif delta_phi.domain == ["current collector"]:
            delta_phi_av = delta_phi
            delta_phi = pybamm.PrimaryBroadcast(delta_phi, self.domain_for_broadcast)
        else:
            delta_phi_av = pybamm.x_average(delta_phi)

        variables = {
            self.domain + " electrode surface potential difference": delta_phi,
            "X-averaged "
            + self.domain.lower()
            + " electrode surface potential difference": delta_phi_av,
            self.domain
            + " electrode surface potential difference [V]": ocp_ref
            + delta_phi * pot_scale,
            "X-averaged "
            + self.domain.lower()
            + " electrode surface potential difference [V]": ocp_ref
            + delta_phi_av * pot_scale,
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

        # Average, and broadcast if necessary
        if ocp.domain == []:
            ocp_av = ocp
            ocp = pybamm.FullBroadcast(
                ocp, self.domain_for_broadcast, "current collector"
            )
        elif ocp.domain == ["current collector"]:
            ocp_av = ocp
            ocp = pybamm.PrimaryBroadcast(ocp, self.domain_for_broadcast)
        else:
            ocp_av = pybamm.x_average(ocp)
        dUdT_av = pybamm.x_average(dUdT)

        if self.domain == "Negative":
            ocp_dim = self.param.U_n_ref + self.param.potential_scale * ocp
            ocp_av_dim = self.param.U_n_ref + self.param.potential_scale * ocp_av
        elif self.domain == "Positive":
            ocp_dim = self.param.U_p_ref + self.param.potential_scale * ocp
            ocp_av_dim = self.param.U_p_ref + self.param.potential_scale * ocp_av

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
            self.domain + " electrode entropic change": dUdT,
            "X-averaged " + self.domain.lower() + " electrode entropic change": dUdT_av,
        }

        return variables
