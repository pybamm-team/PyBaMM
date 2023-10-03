#
# Class for no SEI
#
import pybamm
from .base_sei import BaseModel


class NoSEI(BaseModel):
    """
    Class for no SEI.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict
        A dictionary of options to be passed to the model.
    phase : str, optional
        Phase of the particle (default is "primary")
    cracks : bool, optional
        Whether this is a submodel for standard SEI or SEI on cracks
    """

    def __init__(self, param, domain, options, phase="primary", cracks=False):
        super().__init__(param, domain, options=options, phase=phase, cracks=cracks)
        if self.options.electrode_types[domain] == "planar":
            self.reaction_loc = "interface"
        else:
            self.reaction_loc = "full electrode"

    def get_fundamental_variables(self):
        domain = self.domain.lower()
        if self.reaction_loc == "interface":
            zero = pybamm.PrimaryBroadcast(pybamm.Scalar(0), "current collector")
        else:
            zero = pybamm.FullBroadcast(
                pybamm.Scalar(0), f"{domain} electrode", "current collector"
            )
        variables = self._get_standard_thickness_variables(zero, zero)
        variables.update(self._get_standard_reaction_variables(zero, zero))
        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_concentration_variables(variables))
        # Update whole cell variables, which also updates the "sum of" variables
        variables.update(super().get_coupled_variables(variables))
        return variables
