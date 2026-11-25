#
# Class for Ordered packing transport_efficiency
#
import pybamm

from .base_transport_efficiency import BaseModel


class OrderedPacking(BaseModel):
    """Submodel for Ordered Packing transport_efficiency
    :footcite:t:`Akanni1987`, :footcite:t:`Shen2007`

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
        pybamm.citations.register("Akanni1987")

    def _transport_efficiency(self, volume_fraction, domain_param):
        return (3 - volume_fraction) * 0.5
