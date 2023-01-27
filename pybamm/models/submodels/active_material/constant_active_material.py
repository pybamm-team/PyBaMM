#
# Class for constant active material
#
import pybamm

from .base_active_material import BaseModel


class Constant(BaseModel):
    """Submodel for constant active material

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    options : dict
        Additional options to pass to the model
    phase : str, optional
        Phase of the particle (default is "primary")

    **Extends:** :class:`pybamm.active_material.BaseModel`
    """

    def get_fundamental_variables(self):
        domain = self.domain
        eps_solid = self.phase_param.epsilon_s
        deps_solid_dt = pybamm.FullBroadcast(
            0, f"{domain} electrode", "current collector"
        )

        variables = self._get_standard_active_material_variables(eps_solid)
        variables.update(
            self._get_standard_active_material_change_variables(deps_solid_dt)
        )

        variables.update(
            {
                "Loss of lithium due to loss of active material "
                f"in {domain} electrode [mol]": pybamm.Scalar(0)
            }
        )

        return variables
