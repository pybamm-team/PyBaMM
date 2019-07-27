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


    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

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

        i_typ = self.param.i_typ
        L_x = self.param.L_x
        j_n_scale = i_typ / (self.param.a_n_dim * L_x)
        j_p_scale = i_typ / (self.param.a_p_dim * L_x)

        j_n = variables[
            "Negative electrode" + self.reaction_name + " interfacial current density"
        ]
        j_s = pybamm.FullBroadcast(0, "separator", "current collector")
        j_p = variables[
            "Positive electrode" + self.reaction_name + " interfacial current density"
        ]
        j = pybamm.Concatenation(j_n, j_s, j_p)
        j_dim = pybamm.Concatenation(j_n_scale * j_n, j_s, j_p_scale * j_p)

        if self.reaction_name == "":
            variables = {
                "Interfacial current density": j,
                "Interfacial current density [A.m-2]": j_dim,
                "Interfacial current density per volume [A.m-3]": i_typ / L_x * j,
            }
        else:
            reaction_name = self.reaction_name[1:].capitalize()
            variables = {
                reaction_name + " interfacial current density": j,
                reaction_name + " interfacial current density [A.m-2]": j_dim,
                reaction_name
                + " interfacial current density per volume [A.m-3]": i_typ / L_x * j,
            }

        return variables

    def _get_standard_exchange_current_variables(self, j0):

        i_typ = self.param.i_typ
        L_x = self.param.L_x
        if self.domain == "Negative":
            j_scale = i_typ / (self.param.a_n_dim * L_x)
        elif self.domain == "Positive":
            j_scale = i_typ / (self.param.a_p_dim * L_x)

        # Average, and broadcast if necessary
        j0_av = pybamm.x_average(j0)
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

    def _get_standard_surface_potential_difference_variables(self, delta_phi):

        if self.domain == "Negative":
            ocp_ref = self.param.U_n_ref
        elif self.domain == "Positive":
            ocp_ref = self.param.U_p_ref
        pot_scale = self.param.potential_scale

        # Average, and broadcast if necessary
        delta_phi_av = pybamm.x_average(delta_phi)
        if delta_phi.domain == []:
            delta_phi = pybamm.FullBroadcast(
                delta_phi, self.domain_for_broadcast, "current collector"
            )
        elif delta_phi.domain == ["current collector"]:
            delta_phi = pybamm.PrimaryBroadcast(delta_phi, self.domain_for_broadcast)

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
        ocp_av = pybamm.x_average(ocp)
        if ocp.domain == []:
            ocp = pybamm.FullBroadcast(
                ocp, self.domain_for_broadcast, "current collector"
            )
        elif ocp.domain == ["current collector"]:
            ocp = pybamm.PrimaryBroadcast(ocp, self.domain_for_broadcast)
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
