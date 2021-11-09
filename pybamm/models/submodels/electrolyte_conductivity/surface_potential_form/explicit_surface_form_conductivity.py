#
# Class for explicit surface form potentials
#
import pybamm
from ..base_electrolyte_conductivity import BaseElectrolyteConductivity


class Explicit(BaseElectrolyteConductivity):
    """Class for deriving surface potential difference variables from the electrode
    and electrolyte potentials

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain in which the model holds
    options : dict, optional
        A dictionary of options to be passed to the model.


    **Extends:** :class:`pybamm.electrolyte_conductivity.BaseElectrolyteConductivity`
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options)

    def get_coupled_variables(self, variables):
        # skip for separator
        if self.domain == "Separator":
            return variables

        if self.half_cell and self.domain == "Negative":
            domain = "Lithium metal interface"
        else:
            domain = self.domain
        phi_s = variables[domain + " electrode potential"]
        phi_e = variables[domain + " electrolyte potential"]
        delta_phi = phi_s - phi_e
        variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi)
        )
        if (
            "X-averaged"
            + self.domain.lower()
            + "electrode surface potential difference"
            not in variables
        ):
            delta_phi_av = pybamm.x_average(delta_phi)
            variables.update(
                self._get_standard_average_surface_potential_difference_variables(
                    delta_phi_av
                )
            )
        return variables

    def set_boundary_conditions(self, variables):
        # don't set any boundary conditions
        return
