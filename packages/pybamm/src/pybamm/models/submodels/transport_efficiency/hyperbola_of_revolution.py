#
# Class for hyperbola of revolution transport_efficiency
#
import pybamm

from .base_transport_efficiency import BaseModel


class HyperbolaOfRevolution(BaseModel):
    """Submodel for Hyperbola of revolution transport_efficiency
    :footcite:t:`Petersen1958`, :footcite:t:`Shen2007`

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
        pybamm.citations.register("Petersen1958")

    def _transport_efficiency(self, volume_fraction, domain_param):
        return 2 - volume_fraction
