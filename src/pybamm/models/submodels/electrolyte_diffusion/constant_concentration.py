#
# Class for leading-order electrolyte diffusion employing stefan-maxwell
#
import pybamm

from .base_electrolyte_diffusion import BaseElectrolyteDiffusion


class ConstantConcentration(BaseElectrolyteDiffusion):
    """Class for constant concentration of electrolyte

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, options=None):
        super().__init__(param, options)

    def build(self):
        c_e_init = self.param.c_e_init
        eps_c_e_dict = {
            domain: self.param.domain_params[domain.split()[0]].epsilon_init * c_e_init
            for domain in self.options.whole_cell_domains
        }
        variables = self._get_standard_porosity_times_concentration_variables(
            eps_c_e_dict
        )
        N_e = pybamm.FullBroadcastToEdges(
            0,
            [domain for domain in self.options.whole_cell_domains],
            "current collector",
        )

        variables.update(self._get_standard_flux_variables(N_e))
        c_e_dict = {}
        for domain in self.options.whole_cell_domains:
            Domain = domain.capitalize()
            eps_k = pybamm.CoupledVariable(
                f"{Domain} porosity",
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
            )
            self.coupled_variables.update({eps_k.name: eps_k})
            eps_c_e_k = pybamm.CoupledVariable(
                f"{Domain} porosity times concentration [mol.m-3]",
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
            )
            self.coupled_variables.update({eps_c_e_k.name: eps_c_e_k})
            c_e_k = eps_c_e_k / eps_k
            c_e_dict[domain] = c_e_k

        variables["Electrolyte concentration concatenation [mol.m-3]"] = (
            pybamm.concatenation(*c_e_dict.values())
        )
        variables.update(self._get_standard_domain_concentration_variables(c_e_dict))

        eps = pybamm.concatenation(
            *[
                self.coupled_variables[f"{domain.capitalize()} porosity"]
                for domain in self.options.whole_cell_domains
            ]
        )

        c_e = variables["Porosity times concentration [mol.m-3]"] / eps
        variables.update(self._get_standard_whole_cell_concentration_variables(c_e))

        self.boundary_conditions = {
            c_e: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }

        self.variables.update(variables)

    def add_events_from(self, variables):
        # No event since the concentration is constant
        pass
