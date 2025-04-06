import pybamm
from typing import Optional, Any, TYPE_CHECKING

from .base_electrolyte_diffusion import BaseElectrolyteDiffusion

if TYPE_CHECKING:
    from pybamm import ParameterValues, Symbol


class ConstantConcentration(BaseElectrolyteDiffusion):
    """Class for constant concentration of electrolyte

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param: "ParameterValues", options: Optional[dict[str, Any]] = None) -> None:
        super().__init__(param, options)

    def get_fundamental_variables(self) -> dict[str, pybamm.Symbol]:
        c_e_init: float = self.param.c_e_init
        eps_c_e_dict: dict[str, pybamm.Symbol] = {
            domain: self.param.domain_params[domain.split()[0]].epsilon_init * c_e_init
            for domain in self.options.whole_cell_domains
        }
        variables: dict[str, pybamm.Symbol] = self._get_standard_porosity_times_concentration_variables(
            eps_c_e_dict
        )
        N_e: pybamm.Symbol = pybamm.FullBroadcastToEdges(
            0,
            [domain for domain in self.options.whole_cell_domains],
            "current collector",
        )

        variables.update(self._get_standard_flux_variables(N_e))

        return variables

    def get_coupled_variables(self, variables: dict[str, pybamm.Symbol]) -> dict[str, pybamm.Symbol]:
        c_e_dict: dict[str, pybamm.Symbol] = {}
        for domain in self.options.whole_cell_domains:
            Domain = domain.capitalize()
            eps_k: pybamm.Symbol = variables[f"{Domain} porosity"]
            eps_c_e_k: pybamm.Symbol = variables[f"{Domain} porosity times concentration [mol.m-3]"]
            c_e_k: pybamm.Symbol = eps_c_e_k / eps_k
            c_e_dict[domain] = c_e_k

        variables["Electrolyte concentration concatenation [mol.m-3]"] = (
            pybamm.concatenation(*c_e_dict.values())
        )
        variables.update(self._get_standard_domain_concentration_variables(c_e_dict))

        c_e: pybamm.Symbol = (
            variables["Porosity times concentration [mol.m-3]"] / variables["Porosity"]
        )
        variables.update(self._get_standard_whole_cell_concentration_variables(c_e))

        return variables

    def set_boundary_conditions(self, variables: dict[str, pybamm.Symbol]) -> None:
        """
        We provide boundary conditions even though the concentration is constant
        so that the gradient of the concentration has the correct shape after
        discretisation.
        """

        c_e: pybamm.Symbol = variables["Electrolyte concentration [mol.m-3]"]

        self.boundary_conditions = {
            c_e: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }

    def add_events_from(self, variables: dict[str, pybamm.Symbol]) -> None:
        # No event since the concentration is constant
        pass