#
# Class for no SEI
#
import pybamm
from .base_sei import BaseModel


class NoSEI(BaseModel):
    """
    Class for no SEI.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.sei.BaseModel`
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options=options)

    def get_fundamental_variables(self):
        if self.half_cell and self.domain == "Negative":
            zero = pybamm.PrimaryBroadcast(pybamm.Scalar(0), "current collector")
        else:
            zero = pybamm.FullBroadcast(
                pybamm.Scalar(0),
                self.domain.lower() + " electrode",
                "current collector",
            )
        variables = self._get_standard_thickness_variables(zero, zero)
        variables.update(self._get_standard_concentration_variables(variables))
        variables.update(self._get_standard_reaction_variables(zero, zero))
        return variables

    def get_coupled_variables(self, variables):
        # Update whole cell variables, which also updates the "sum of" variables
        if (
            "Negative electrode SEI interfacial current density" in variables
            and "Positive electrode SEI interfacial current density" in variables
            and "SEI interfacial current density" not in variables
        ):
            variables.update(
                self._get_standard_whole_cell_interfacial_current_variables(variables)
            )

        return variables
