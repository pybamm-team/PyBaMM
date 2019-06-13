#
# Class for the combined electrolyte potential employing stefan-maxwell
#
import pybamm

import numpy as np
from .base_stefan_maxwell_conductivity import BaseModel


class CombinedOrderModel(BaseModel):
    """Class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Combined refers to a combined
    leading and first-order expression from the asymptotic reduction)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseStefanMaxwellConductivity`
    """

    def __init__(self, param, domain):
        super().__init__(param)
        self._domain = domain

    def get_coupled_variables(self, variables):

        i_boundary_cc = variables["Current collector current density"]
        eps = variables["Porosity"]
        c_e = variables["Electrolyte concentration"]
        c_e_av = variables["Average electrolyte concentration"]
        ocp_n_av = variables["Average negative electrode open circuit potential"]
        eta_r_n_av = variables["Average negative reaction overpotential"]
        phi_s_n_av = variables["Average negative electrode potential"]

        c_e_n, c_e_s, c_e_p = c_e.orphans
        eps_n, eps_s, eps_p = [e.orphans[0] for e in eps.orphans]

        param = self.param
        l_n = param.l_n
        l_p = param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        # bulk conductivities
        kappa_n = param.kappa_e(c_e_av) * eps_n ** param.b
        kappa_s = param.kappa_e(c_e_av) * eps_s ** param.b
        kappa_p = param.kappa_e(c_e_av) * eps_p ** param.b

        # electrolyte current
        i_e_n = i_boundary_cc * x_n / l_n
        i_e_s = pybamm.Broadcast(i_boundary_cc, ["separator"])
        i_e_p = i_boundary_cc * (1 - x_p) / l_p
        i_e = pybamm.Concatenation(i_e_n, i_e_s, i_e_p)

        # electrolyte potential
        phi_e_const = (
            -ocp_n_av
            - eta_r_n_av
            + phi_s_n_av
            - 2
            * (1 - param.t_plus)
            * pybamm.average(pybamm.Function(np.log, c_e_n / c_e_av))
            - i_boundary_cc
            * param.C_e
            * l_n
            / param.gamma_e
            * (1 / (3 * kappa_n) - 1 / kappa_s)
        )

        phi_e_n = (
            phi_e_const
            + 2 * (1 - param.t_plus) * pybamm.Function(np.log, c_e_n / c_e_av)
            - (i_boundary_cc * param.C_e / param.gamma_e)
            * ((x_n ** 2 - l_n ** 2) / (2 * kappa_n * l_n) + l_n / kappa_s)
        )

        phi_e_s = (
            phi_e_const
            + 2 * (1 - param.t_plus) * pybamm.Function(np.log, c_e_s / c_e_av)
            - (i_boundary_cc * param.C_e / param.gamma_e) * (x_s / kappa_s)
        )

        phi_e_p = (
            phi_e_const
            + 2 * (1 - param.t_plus) * pybamm.Function(np.log, c_e_p / c_e_av)
            - (i_boundary_cc * param.C_e / param.gamma_e)
            * (
                (x_p * (2 - x_p) + l_p ** 2 - 1) / (2 * kappa_p * l_p)
                + (1 - l_p) / kappa_s
            )
        )
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)
        phi_e_av = pybamm.average(phi_e)

        # concentration overpotential
        eta_c_av = (
            2
            * (1 - param.t_plus)
            * (
                pybamm.average(pybamm.Function(np.log, c_e_p / c_e_av))
                - pybamm.average(pybamm.Function(np.log, c_e_n / c_e_av))
            )
        )

        # average electrolyte ohmic losses
        delta_phi_e_av = -(param.C_e * i_boundary_cc / param.gamma_e) * (
            param.l_n / (3 * kappa_n)
            + param.l_s / (kappa_s)
            + param.l_p / (3 * kappa_p)
        )

        variables.update(self._get_standard_potential_variables(phi_e, phi_e_av))
        variables.update(self._get_standard_current_variables(i_e))
        variables.update(self._get_split_overpotential(eta_c_av, delta_phi_e_av))

        return variables

