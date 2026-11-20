#
# Class for cation-exchange membrane transport_efficiency
#
import pybamm

from .base_transport_efficiency import BaseModel


class CationExchangeMembrane(BaseModel):
    """Submodel for Cation Exchange Membrane transport_efficiency,
    :footcite:t:`Bruggeman1935`, :footcite:t:`Shen2007`

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
        pybamm.citations.register("Shen2007")
        pybamm.citations.register("Mackie1955")

    def _transport_efficiency(self, volume_fraction, domain_param):
        return ((2 - volume_fraction) / volume_fraction) ** 2
