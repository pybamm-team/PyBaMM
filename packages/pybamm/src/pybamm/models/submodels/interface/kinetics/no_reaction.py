#
# Bulter volmer class
#

import pybamm
from pybamm.models.submodels.interface.base_interface import BaseInterface


class NoReaction(BaseInterface):
    """
    Base submodel for when no reaction occurs

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")
    """

    def __init__(self, param, domain, reaction, options, phase="primary"):
        options = {
            "SEI film resistance": "none",
            "total interfacial current density as a state": "false",
        }
        super().__init__(param, domain, reaction, options, phase)

    def build(self):
        zero = pybamm.Scalar(0)
        variables = self._get_standard_interfacial_current_variables(zero)
        variables.update(self._get_standard_exchange_current_variables(zero))
        j = variables[
            f"{self.domain.capitalize()} electrode {self.reaction_name}"
            "interfacial current density [A.m-2]"
        ]
        variables.update(self._get_standard_volumetric_current_density_variables(j))
        self.variables.update(variables)
