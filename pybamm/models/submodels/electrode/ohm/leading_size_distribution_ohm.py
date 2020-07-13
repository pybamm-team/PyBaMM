#
# Full model for Ohm's law in the electrode
#
import pybamm

from .base_ohm import BaseModel


class LeadingOrderSizeDistribution(BaseModel):
    """An electrode submodel that employs Ohm's law the leading-order approximation to
    governing equations when there is a distribution of particle sizes. An algebraic
    equation is imposed for the x-averaged surface potential difference.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'Negative' or 'Positive'
    set_positive_potential :  bool, optional
        If True the battery model sets the positive potential based on the current.
        If False, the potential is specified by the user. Default is True.

    **Extends:** :class:`pybamm.electrode.ohm.BaseModel`
    """

    def __init__(self, param, domain, set_positive_potential=True):
        super().__init__(param, domain, set_positive_potential=set_positive_potential)

    def get_fundamental_variables(self):

        delta_phi_av = pybamm.Variable(
            "X-averaged "
            + self.domain.lower()
            + " electrode surface potential difference",
            domain="current collector",
        )
        variables = self._get_standard_surface_potential_difference_variables(
            delta_phi_av
        )

        return variables

    def get_coupled_variables(self, variables):

        i_boundary_cc = variables["Current collector current density"]
        phi_s_cn = variables["Negative current collector potential"]

        # import parameters and spatial variables
        l_n = self.param.l_n
        l_p = self.param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        if self.domain == "Negative":
            phi_s = pybamm.PrimaryBroadcast(phi_s_cn, ["negative electrode"])
            i_s = i_boundary_cc * (1 - x_n / l_n)

        elif self.domain == "Positive":
            # recall delta_phi = phi_s - phi_e
            delta_phi_p_av = variables[
                "X-averaged positive electrode surface potential difference"
            ]
            phi_e_p_av = variables["X-averaged positive electrolyte potential"]

            v = delta_phi_p_av + phi_e_p_av

            phi_s = pybamm.PrimaryBroadcast(v, ["positive electrode"])
            i_s = i_boundary_cc * (1 - (1 - x_p) / l_p)

        variables.update(self._get_standard_potential_variables(phi_s))
        variables.update(self._get_standard_current_variables(i_s))

        if self.domain == "Positive":
            variables.update(self._get_standard_whole_cell_variables(variables))

        return variables

    def set_algebraic(self, variables):

        j_tot_av = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode total interfacial current density"
        ]

        # Extract total sum of interfacial current densities
        sum_j_av = variables[
            "Sum of x-averaged "
            + self.domain.lower()
            + " electrode interfacial current densities"
        ]
        delta_phi_av = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode surface potential difference"
        ]
        self.algebraic[delta_phi_av] = sum_j_av - j_tot_av

    def set_initial_conditions(self, variables):

        delta_phi_av = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode surface potential difference"
        ]
        T_init = self.param.T_init

        if self.domain == "Negative":
            delta_phi_av_init = self.param.U_n(self.param.c_n_init(0), T_init)
        elif self.domain == "Positive":
            delta_phi_av_init = self.param.U_p(
                self.param.c_p_init(1), T_init
            )

        self.initial_conditions[delta_phi_av] = delta_phi_av_init

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

        #        # For particle-size distributions (true here), must broadcast further
        #        delta_phi = pybamm.PrimaryBroadcast(delta_phi, [self.domain.lower() + " particle-size domain"])
        #        delta_phi_av = pybamm.PrimaryBroadcast(delta_phi_av, [self.domain.lower() + " particle-size domain"])

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
