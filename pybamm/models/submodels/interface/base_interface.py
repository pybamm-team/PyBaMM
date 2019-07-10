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

        # Average, and broadcast if necessary
        j_av = pybamm.average(j)
        if j.domain == []:
            j = pybamm.Broadcast(j, self.domain_for_broadcast)
        elif j.domain == ["current collector"]:
            j = pybamm.Broadcast(j, self.domain_for_broadcast, broadcast_type="primary")

        variables = {
            self.domain + " electrode interfacial current density": j,
            "Average "
            + self.domain.lower()
            + " electrode interfacial current density": j_av,
            self.domain + " interfacial current density [A.m-2]": i_typ * j,
            "Average "
            + self.domain.lower()
            + " electrode interfacial current density [A.m-2]": i_typ * j_av,
        }

        return variables

    def _get_standard_total_interfacial_current_variables(self, j_tot_av):

        i_typ = self.param.i_typ

        variables = {
            "Average "
            + self.domain.lower()
            + " electrode total interfacial current density": j_tot_av,
            "Average "
            + self.domain.lower()
            + " electrode total interfacial current density [A.m-2]": i_typ * j_tot_av,
        }

        return variables

    def _get_standard_whole_cell_interfacial_current_variables(self, variables):

        i_typ = self.param.i_typ

        j_n = variables["Negative electrode interfacial current density"]
        j_s = pybamm.Broadcast(0, ["separator"])
        j_p = variables["Positive electrode interfacial current density"]
        j = pybamm.Concatenation(j_n, j_s, j_p)

        variables.update(
            {
                "Interfacial current density": j,
                "Interfacial current density [A.m-2]": i_typ * j,
            }
        )

        return variables

    def _get_standard_exchange_current_variables(self, j0):

        i_typ = self.param.i_typ
        # Average, and broadcast if necessary
        j0_av = pybamm.average(j0)
        if j0.domain == []:
            j0 = pybamm.Broadcast(j0, self.domain_for_broadcast)
        elif j0.domain == ["current collector"]:
            j0 = pybamm.Broadcast(
                j0, self.domain_for_broadcast, broadcast_type="primary"
            )

        variables = {
            self.domain + " electrode exchange current density": j0,
            "Average "
            + self.domain.lower()
            + " electrode exchange current density": j0_av,
            self.domain + " electrode exchange current density [A.m-2]": i_typ * j0,
            "Average "
            + self.domain.lower()
            + " electrode exchange current density [A.m-2]": i_typ * j0_av,
        }

        return variables

    def _get_standard_whole_cell_exchange_current_variables(self, variables):

        i_typ = self.param.i_typ

        j0_n = variables["Negative electrode exchange current density"]
        j0_s = pybamm.Broadcast(0, ["separator"])
        j0_p = variables["Positive electrode exchange current density"]
        j0 = pybamm.Concatenation(j0_n, j0_s, j0_p)

        variables.update(
            {
                "Exchange current density": j0,
                "Exchange current density [A.m-2]": i_typ * j0,
            }
        )

        return variables

    def _get_standard_overpotential_variables(self, eta_r):

        pot_scale = self.param.potential_scale
        # Average, and broadcast if necessary
        eta_r_av = pybamm.average(eta_r)
        if eta_r.domain == []:
            eta_r = pybamm.Broadcast(eta_r, self.domain_for_broadcast)
        elif eta_r.domain == ["current collector"]:
            eta_r = pybamm.Broadcast(
                eta_r, self.domain_for_broadcast, broadcast_type="primary"
            )

        variables = {
            self.domain + " electrode reaction overpotential": eta_r,
            "Average "
            + self.domain.lower()
            + " electrode reaction overpotential": eta_r_av,
            self.domain + " electrode reaction overpotential [V]": eta_r * pot_scale,
            "Average "
            + self.domain.lower()
            + " electrode reaction overpotential [V]": eta_r_av * pot_scale,
        }

        return variables

    def _get_standard_surface_potential_difference_variables(self, delta_phi):

        if self.domain == "Negative":
            ocp_ref = self.param.U_n_ref
        elif self.domain == "Positive":
            ocp_ref = self.param.U_p_ref
        pot_scale = self.param.potential_scale

        # Average, and broadcast if necessary
        delta_phi_av = pybamm.average(delta_phi)
        if delta_phi.domain == []:
            delta_phi = pybamm.Broadcast(delta_phi, self.domain_for_broadcast)
        elif delta_phi.domain == ["current collector"]:
            delta_phi = pybamm.Broadcast(
                delta_phi, self.domain_for_broadcast, broadcast_type="primary"
            )

        variables = {
            self.domain + " electrode surface potential difference": delta_phi,
            "Average "
            + self.domain.lower()
            + " electrode surface potential difference": delta_phi_av,
            self.domain
            + " electrode surface potential difference [V]": ocp_ref
            + delta_phi * pot_scale,
            "Average "
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
        ocp_av = pybamm.average(ocp)
        if ocp.domain == []:
            ocp = pybamm.Broadcast(ocp, self.domain_for_broadcast)
        elif ocp.domain == ["current collector"]:
            ocp = pybamm.Broadcast(
                ocp, self.domain_for_broadcast, broadcast_type="primary"
            )
        dUdT_av = pybamm.average(dUdT)

        if self.domain == "Negative":
            ocp_dim = self.param.U_n_ref + self.param.potential_scale * ocp
            ocp_av_dim = self.param.U_n_ref + self.param.potential_scale * ocp_av
        elif self.domain == "Positive":
            ocp_dim = self.param.U_p_ref + self.param.potential_scale * ocp
            ocp_av_dim = self.param.U_p_ref + self.param.potential_scale * ocp_av

        variables = {
            self.domain + " electrode open circuit potential": ocp,
            self.domain + " electrode open circuit potential [V]": ocp_dim,
            "Average "
            + self.domain.lower()
            + " electrode open circuit potential": ocp_av,
            "Average "
            + self.domain.lower()
            + " electrode open circuit potential [V]": ocp_av_dim,
            self.domain + " electrode entropic change": dUdT,
            "Average " + self.domain.lower() + " electrode entropic change": dUdT_av,
        }

        return variables
