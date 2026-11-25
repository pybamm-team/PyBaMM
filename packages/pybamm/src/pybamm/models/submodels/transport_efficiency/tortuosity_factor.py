#
# Class for tortuosity factor transport_efficiency
#
from .base_transport_efficiency import BaseModel


class TortuosityFactor(BaseModel):
    """Submodel for user supplied tortuosity factor transport_efficiency

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    component : str
        The material for the model ('Electrolyte' or 'Electrode').
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def _transport_efficiency(self, volume_fraction, domain_param):
        if self.component == "Electrolyte":
            return volume_fraction / domain_param.tau_e
        return volume_fraction / domain_param.tau_s
