#
# Class for the leading-order electrolyte potential employing stefan-maxwell
#
import pybamm

from .base_electrolyte_conductivity import BaseElectrolyteConductivity


class LeadingOrder(BaseElectrolyteConductivity):
    """Leading-order model for conservation of charge in the electrolyte
    employing the Stefan-Maxwell constitutive equations. (Leading refers
    to leading-order in the asymptotic reduction)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str, optional
        The domain in which the model holds
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, domain=None, options=None):
        super().__init__(param, domain, options=options)

    def build(self):
        if self.options.electrode_types["negative"] == "planar":
            delta_phi_n_av = None
        else:
            delta_phi_n_av = self.coupled_variable(
                "X-averaged negative electrode surface potential difference [V]",
                domain="current collector",
            )
        self._build_leading_order(delta_phi_n_av)

    def _build_leading_order(self, delta_phi_n_av):
        """
        Set the leading-order electrolyte potential and current variables, and the
        boundary conditions on the electrolyte potential.

        Parameters
        ----------
        delta_phi_n_av : :class:`pybamm.Symbol` or None
            The x-averaged negative electrode surface potential difference, or None
            for a planar negative electrode
        """
        i_boundary_cc = self.coupled_variable(
            "Current collector current density [A.m-2]", domain="current collector"
        )
        if self.options.electrode_types["negative"] == "planar":
            phi_e_av = self._coupled_lithium_metal_electrolyte_potential()
            i_e_n = None
        else:
            # delta_phi = phi_s - phi_e
            phi_s_n_av = self.coupled_variable(
                "X-averaged negative electrode potential [V]",
                domain="current collector",
            )
            phi_e_av = phi_s_n_av - delta_phi_n_av
            x_n = pybamm.standard_spatial_vars.x_n
            i_e_n = i_boundary_cc * x_n / self.param.n.L

        x_p = pybamm.standard_spatial_vars.x_p
        i_e_s = pybamm.PrimaryBroadcast(i_boundary_cc, ["separator"])
        i_e_p = i_boundary_cc * (self.param.L_x - x_p) / self.param.p.L
        i_e = pybamm.concatenation(i_e_n, i_e_s, i_e_p)

        phi_e_dict = {
            domain: pybamm.PrimaryBroadcast(phi_e_av, domain)
            for domain in self.options.whole_cell_domains
        }
        variables = self._get_standard_potential_variables(phi_e_dict)
        variables.update(self._get_standard_current_variables(i_e))

        # concentration overpotential and ohmic losses are zero at leading order
        eta_c_av = pybamm.PrimaryBroadcast(0, "current collector")
        delta_phi_e_av = pybamm.PrimaryBroadcast(0, "current collector")
        variables.update(self._get_split_overpotential(eta_c_av, delta_phi_e_av))
        self.variables.update(variables)

        self.boundary_conditions.update(
            self._get_electrolyte_potential_boundary_conditions(
                variables["Electrolyte potential [V]"]
            )
        )
