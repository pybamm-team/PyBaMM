#
# Class for Bruggeman transport_efficiency
#
import pybamm

from .base_transport_efficiency import BaseModel


class Bruggeman(BaseModel):
    """Submodel for Bruggeman transport_efficiency,
    :footcite:t:`Bruggeman1935`

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    component : str
        The material for the model ('Electrolyte' or 'Electrode').
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, component, options=None):
        super().__init__(param, component, options=options)
        pybamm.citations.register("Bruggeman1935")

    def _transport_efficiency(self, volume_fraction, domain_param):
        if self.component == "Electrolyte":
            return volume_fraction**domain_param.b_e
        return volume_fraction**domain_param.b_s
