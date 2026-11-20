#
# Class for no mechanics
#
from .base_mechanics import BaseMechanics


class NoMechanics(BaseMechanics):
    """
    Class for no particle mechanics.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")
    """

    def __init__(self, param, domain, options, phase="primary"):
        super().__init__(param, domain, options, phase)

    def build(self):
        variables = self._get_zero_cracking_variables()
        l_cr = variables[self._build_var_name("particle crack length", "[m]")]
        variables.update(self._get_standard_surface_variables(l_cr))
        self.variables.update(variables)
