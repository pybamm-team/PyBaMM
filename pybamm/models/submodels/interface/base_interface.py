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

    def _get_average_interfacial_current_density(self, variables):
        """
        Method to obtain the average interfacial current density.
        """

        i_boundary_cc = variables["Current collector current density"]

        if self.domain == "Negative":
            j_av = i_boundary_cc / pybamm.geometric_parameters.l_n

        elif self.domain == "Positive":
            j_av = -i_boundary_cc / pybamm.geometric_parameters.l_p

        return j_av

    def _get_standard_interfacial_current_variables(self, j, j_av):

        i_typ = self.param.i_typ

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
        if j0.domain in [[], ["current collector"]]:
            j0 = pybamm.Broadcast(j0, self.domain_for_broadcast)

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
        eta_r_av = pybamm.average(eta_r)

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

        delta_phi_av = pybamm.average(delta_phi)

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
