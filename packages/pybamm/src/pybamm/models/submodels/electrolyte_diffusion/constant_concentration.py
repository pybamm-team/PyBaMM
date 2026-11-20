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

    def _get_porosities(self):
        if (
            self.options["SEI porosity change"] == "true"
            or self.options["lithium plating porosity change"] == "true"
        ):
            return super()._get_porosities()
        # Use the constant porosity directly so the concentration simplifies to a
        # broadcast of the initial concentration
        eps_dict = {
            domain: self.param.domain_params[domain.split()[0]].epsilon_init
            for domain in self.options.whole_cell_domains
        }
        return eps_dict, pybamm.concatenation(*eps_dict.values())

    def build(self):
        whole_cell = self.options.whole_cell_domains
        c_e_init = self.param.c_e_init
        eps_c_e_dict = {
            domain: self.param.domain_params[domain.split()[0]].epsilon_init * c_e_init
            for domain in whole_cell
        }
        variables = self._get_standard_porosity_times_concentration_variables(
            eps_c_e_dict
        )
        variables.update(
            self._get_concentration_from_porosity_times_concentration(eps_c_e_dict)
        )
        N_e = pybamm.FullBroadcastToEdges(0, whole_cell, "current collector")
        variables.update(self._get_standard_flux_variables(N_e))
        self.variables.update(variables)

        # Boundary conditions give the gradient of the constant concentration the
        # right shape after discretisation
        c_e = variables["Electrolyte concentration [mol.m-3]"]
        self.boundary_conditions = {
            c_e: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
