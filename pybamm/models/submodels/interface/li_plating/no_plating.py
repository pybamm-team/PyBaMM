#
# Class for no plating
#
import pybamm
from .base_plating import BasePlating


class NoPlating(BasePlating):
    """Base class for no Li plating/stripping.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'

    References
    ----------
    .. [1] SEJ O'Kane, ID Campbell, MWJ Marzook, GJ Offer and M Marinescu. "Physical
           Origin of the Differential Voltage Minimum Associated with Li Plating in 
           Lithium-Ion Batteries". Journal of The Electrochemical Society,
           167:090540, 2019

    **Extends:** :class:`pybamm.li_plating.BasePlating`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        zero = pybamm.FullBroadcast(
            pybamm.Scalar(0), self.domain.lower() + " electrode", "current collector"
        )
        variables = self._get_standard_concentration_variables(zero)
        variables.update(self._get_standard_reaction_variables(zero))
        return variables

    def get_coupled_variables(self, variables):
        # Update whole cell variables, which also updates the "sum of" variables
        if (
            "Negative electrode Li plating interfacial current density" in variables
            and "Positive electrode Li plating interfacial current density" in variables
            and "Li plating interfacial current density" not in variables
        ):
            variables.update(
                self._get_standard_whole_cell_interfacial_current_variables(variables)
            )

        return variables
