#
# Class for explicit surface form potentials
#
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


    **Extends:** :class:`pybamm.electrolyte_conductivity.BaseElectrolyteConductivity`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_coupled_variables(self, variables):
        # skip for separator
        if self.domain == "Separator":
            return variables

        phi_s = variables[self.domain + " electrode potential"]
        phi_e = variables[self.domain + " electrolyte potential"]
        delta_phi = phi_s - phi_e
        variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi)
        )
        return variables

    def set_boundary_conditions(self, variables):
        # don't set any boundary conditions
        return
