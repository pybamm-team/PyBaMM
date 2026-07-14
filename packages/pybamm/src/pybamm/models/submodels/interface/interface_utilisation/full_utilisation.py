#
# Class for full interface utilisation
#
import pybamm

from .base_utilisation import BaseModel


class Full(BaseModel):
    """Submodel for constant interface utilisation

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'negative' or 'positive'
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def get_fundamental_variables(self):
        domain = self.domain
        u = pybamm.FullBroadcast(1, f"{domain} electrode", "current collector")

        variables = self._get_standard_interface_utilisation_variables(u)
        return variables
