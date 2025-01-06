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

    def build(self, submodels):
        if "negative electrode" not in self.options.whole_cell_domains:
            phi_e_av = pybamm.CoupledVariable(
                "Lithium metal interface electrolyte potential [V]",
                domain="current collector",
            )
            self.coupled_variables.update({phi_e_av.name: phi_e_av})
        else:
            # delta_phi = phi_s - phi_e
            delta_phi_n_av = pybamm.CoupledVariable(
                "X-averaged negative electrode surface potential difference [V]",
                domain="current collector",
            )
            self.coupled_variables.update({delta_phi_n_av.name: delta_phi_n_av})
            phi_s_n_av = pybamm.CoupledVariable(
                "X-averaged negative electrode potential [V]",
                domain="current collector",
            )
            self.coupled_variables.update({phi_s_n_av.name: phi_s_n_av})
            phi_e_av = phi_s_n_av - delta_phi_n_av

        i_boundary_cc = pybamm.CoupledVariable(
            "Current collector current density [A.m-2]",
            domain="current collector",
        )
        self.coupled_variables.update({i_boundary_cc.name: i_boundary_cc})

        L_n = self.param.n.L
        L_p = self.param.p.L
        L_x = self.param.L_x
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        if "negative electrode" not in self.options.whole_cell_domains:
            i_e_n = None
        else:
            i_e_n = i_boundary_cc * x_n / L_n

        phi_e_dict = {
            domain: pybamm.PrimaryBroadcast(phi_e_av, domain)
            for domain in self.options.whole_cell_domains
        }

        i_e_s = pybamm.PrimaryBroadcast(i_boundary_cc, ["separator"])
        i_e_p = i_boundary_cc * (L_x - x_p) / L_p
        i_e = pybamm.concatenation(i_e_n, i_e_s, i_e_p)

        variables = {}
        variables.update(self._get_standard_potential_variables(phi_e_dict))
        variables.update(self._get_standard_current_variables(i_e))

        # concentration overpotential
        eta_c_av = pybamm.PrimaryBroadcast(0, "current collector")
        # ohmic losses
        delta_phi_e_av = pybamm.PrimaryBroadcast(0, "current collector")
        variables.update(self._get_split_overpotential(eta_c_av, delta_phi_e_av))

        self.variables.update(variables)
