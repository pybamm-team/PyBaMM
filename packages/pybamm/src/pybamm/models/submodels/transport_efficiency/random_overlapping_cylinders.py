#
# Class for random overlapping cylinders transport_efficiency
#
import pybamm

from .base_transport_efficiency import BaseModel


class RandomOverlappingCylinders(BaseModel):
    """Submodel for Random Overlapping Cylinders transport_efficiency,
    :footcite:t:`Tomadakis1993`, :footcite:t:`Shen2007`

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
        pybamm.citations.register("Tomadakis1993")

    def _transport_efficiency(self, volume_fraction, domain_param):
        return 1 - pybamm.Log(volume_fraction)
