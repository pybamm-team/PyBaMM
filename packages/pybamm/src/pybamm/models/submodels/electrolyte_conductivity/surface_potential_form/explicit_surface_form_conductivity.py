#
# Class for explicit surface form potentials
#
import pybamm
from pybamm.models.submodels.electrolyte_conductivity.base_electrolyte_conductivity import (
    BaseElectrolyteConductivity,
)


class Explicit(BaseElectrolyteConductivity):
    """Class for deriving surface potential difference variables from the electrode
    and electrolyte potentials

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain in which the model holds
    options : dict
        A dictionary of options to be passed to the model.
    inverse_kinetics : bool, optional
        Whether the interface submodel uses inverse kinetics, in which case it
        defines the x-averaged surface potential difference. Default is False.
    """

    def __init__(self, param, domain, options, inverse_kinetics=False):
        super().__init__(param, domain, options)
        self.inverse_kinetics = inverse_kinetics

    def build(self):
        if self.domain == "separator":
            return

        Domain = self.domain.capitalize()
        electrode = f"{self.domain} electrode"
        phi_s = self._coupled_domain_variable(
            f"{Domain} electrode potential [V]", electrode
        )
        phi_e = self._coupled_domain_variable(
            f"{Domain} electrolyte potential [V]", electrode
        )
        delta_phi = phi_s - phi_e
        self.variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi)
        )
        if not self.inverse_kinetics:
            self.variables.update(
                self._get_standard_average_surface_potential_difference_variables(
                    pybamm.x_average(delta_phi)
                )
            )
