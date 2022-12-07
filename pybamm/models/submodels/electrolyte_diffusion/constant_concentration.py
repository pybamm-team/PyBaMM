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

    **Extends:** :class:`pybamm.electrolyte_diffusion.BaseElectrolyteDiffusion`
    """

    def __init__(self, param, options=None):
        super().__init__(param, options)

    def get_fundamental_variables(self):
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

        return variables

    def get_coupled_variables(self, variables):
        c_e_dict = {}
        for domain in self.options.whole_cell_domains:
            Domain = domain.capitalize()
            eps_k = variables[f"{Domain} porosity"]
            eps_c_e_k = variables[f"{Domain} porosity times concentration [mol.m-3]"]
            c_e_k = eps_c_e_k / eps_k
            c_e_dict[domain] = c_e_k

        variables[
            "Electrolyte concentration concatenation [mol.m-3]"
        ] = pybamm.concatenation(*c_e_dict.values())
        variables.update(self._get_standard_domain_concentration_variables(c_e_dict))

        c_e = (
            variables["Porosity times concentration [mol.m-3]"] / variables["Porosity"]
        )
        variables.update(self._get_standard_whole_cell_concentration_variables(c_e))

        return variables

    def set_boundary_conditions(self, variables):
        """
        We provide boundary conditions even though the concentration is constant
        so that the gradient of the concentration has the correct shape after
        discretisation.
        """

        c_e = variables["Electrolyte concentration [mol.m-3]"]

        self.boundary_conditions = {
            c_e: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }

    def set_events(self, variables):
        # No event since the concentration is constant
        pass
