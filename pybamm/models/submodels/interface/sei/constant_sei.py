#
# Class for constant SEI thickness
#
import pybamm
from .base_sei import BaseModel


class ConstantSEI(BaseModel):
    """Base class for SEI with constant thickness.

    Note that there is no SEI current, so we don't need to update the "sum of
    interfacial current densities" variables from
    :class:`pybamm.interface.BaseInterface`

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'

    **Extends:** :class:`pybamm.sei.BaseModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        # Constant thicknesses
        L_inner = pybamm.sei_parameters.L_inner_0
        L_outer = pybamm.sei_parameters.L_outer_0
        variables = self._get_standard_thickness_variables(L_inner, L_outer)

        # Reactions
        zero = pybamm.FullBroadcast(
            pybamm.Scalar(0), self.domain.lower() + " electrode", "current collector"
        )
        variables.update(self._get_standard_reaction_variables(zero, zero))

        return variables

    def get_coupled_variables(self, variables):
        # Update whole cell variables, which also updates the "sum of" variables
        if (
            "Negative electrode sei interfacial current density" in variables
            and "Positive electrode sei interfacial current density" in variables
            and "Sei interfacial current density" not in variables
        ):
            variables.update(
                self._get_standard_whole_cell_interfacial_current_variables(variables)
            )

        return variables
