#
# Base class for higher order electrolyte potential employing stefan-maxwell
#
import pybamm
from .base_stefan_maxwell_conductivity import BaseModel


class BaseHigherOrder(BaseModel):
    """Base class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str, optional
        The domain in which the model holds

    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.conductivity.BaseModel`
    """

    def __init__(self, param, domain=None):
        super().__init__(param, domain)

    def _higher_order_macinnes_function(self, x):
        "Function to differentiate between composite and first-order models"
        raise NotImplementedError

    def unpack(self, variables):
        raise NotImplementedError

    def get_coupled_variables(self, variables):
        # NOTE: the heavy use of Broadcast and outer in this method is mainly so
        # that products are handled correctly when using 1 or 2D current collector
        # models. In standard 1D battery models outer behaves as a normal multiply.
        # In the future, multiply will automatically handle switching between
        # normal multiply and outer products as appropriate.

        c_e_av = self.unpack(variables)

        i_boundary_cc = variables["Current collector current density"]
        c_e = variables["Electrolyte concentration"]
        delta_phi_n_av = variables[
            "Average negative electrode surface potential difference"
        ]
        phi_s_n_av = variables["Average negative electrode potential"]
        eps_n_av = variables["Average negative electrode porosity"]
        eps_s_av = variables["Average separator porosity"]
        eps_p_av = variables["Average positive electrode porosity"]
        c_e_n, c_e_s, c_e_p = c_e.orphans

        param = self.param
        l_n = param.l_n
        l_p = param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        # bulk conductivities
        kappa_n_av = param.kappa_e(c_e_av) * eps_n_av ** param.b
        kappa_s_av = param.kappa_e(c_e_av) * eps_s_av ** param.b
        kappa_p_av = param.kappa_e(c_e_av) * eps_p_av ** param.b

        chi_av = param.chi(c_e_av)

        # electrolyte current
        i_e_n = pybamm.outer(i_boundary_cc, x_n / l_n)
        i_e_s = pybamm.PrimaryBroadcast(i_boundary_cc, "separator")
        i_e_p = pybamm.outer(i_boundary_cc, (1 - x_p) / l_p)
        i_e = pybamm.Concatenation(i_e_n, i_e_s, i_e_p)

        # electrolyte potential
        import ipdb

        ipdb.set_trace()
        phi_e_const = (
            -delta_phi_n_av
            + phi_s_n_av
            - (
                chi_av
                * pybamm.average(
                    self._higher_order_macinnes_function(
                        c_e_n / pybamm.PrimaryBroadcast(c_e_av, "negative electrode")
                    )
                )
            )
            - (
                (i_boundary_cc * param.C_e * l_n / param.gamma_e)
                * (1 / (3 * kappa_n_av) - 1 / kappa_s_av)
            )
        )

        phi_e_n = (
            pybamm.PrimaryBroadcast(phi_e_const, "negative electrode")
            + (
                pybamm.PrimaryBroadcast(chi_av, "negative electrode")
                * self._higher_order_macinnes_function(
                    c_e_n / pybamm.PrimaryBroadcast(c_e_av, "negative electrode")
                )
            )
            - pybamm.outer(
                i_boundary_cc * (param.C_e / param.gamma_e) / kappa_n_av,
                (x_n ** 2 - l_n ** 2) / (2 * l_n),
            )
            - pybamm.PrimaryBroadcast(
                i_boundary_cc * l_n * (param.C_e / param.gamma_e) / kappa_s_av,
                "negative electrode",
            )
        )

        phi_e_s = pybamm.PrimaryBroadcast(phi_e_const, "separator") + (
            pybamm.PrimaryBroadcast(chi_av, "separator")
            * self._higher_order_macinnes_function(
                c_e_s / pybamm.PrimaryBroadcast(c_e_av, "separator")
            )
        )
        -pybamm.outer(i_boundary_cc * param.C_e / param.gamma_e / kappa_s_av, x_s)

        phi_e_p = (
            pybamm.PrimaryBroadcast(phi_e_const, "positive electrode")
            + (
                pybamm.PrimaryBroadcast(chi_av, "positive electrode")
                * self._higher_order_macinnes_function(
                    c_e_p / pybamm.PrimaryBroadcast(c_e_av, "positive electrode")
                )
            )
            - pybamm.outer(
                i_boundary_cc * (param.C_e / param.gamma_e) / kappa_p_av,
                (x_p * (2 - x_p) + l_p ** 2 - 1) / (2 * l_p),
            )
            - pybamm.PrimaryBroadcast(
                i_boundary_cc * (1 - l_p) * (param.C_e / param.gamma_e) / kappa_s_av,
                "positive electrode",
            )
        )

        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)
        phi_e_av = pybamm.average(phi_e)

        # concentration overpotential
        eta_c_av = chi_av * (
            pybamm.average(
                self._higher_order_macinnes_function(
                    c_e_p / pybamm.PrimaryBroadcast(c_e_av, "positive electrode")
                )
            )
            - pybamm.average(
                self._higher_order_macinnes_function(
                    c_e_n / pybamm.PrimaryBroadcast(c_e_av, "negative electrode")
                )
            )
        )

        # average electrolyte ohmic losses
        delta_phi_e_av = -(param.C_e * i_boundary_cc / param.gamma_e) * (
            param.l_n / (3 * kappa_n_av)
            + param.l_s / (kappa_s_av)
            + param.l_p / (3 * kappa_p_av)
        )

        variables.update(self._get_standard_potential_variables(phi_e, phi_e_av))
        variables.update(self._get_standard_current_variables(i_e))
        variables.update(self._get_split_overpotential(eta_c_av, delta_phi_e_av))

        return variables
