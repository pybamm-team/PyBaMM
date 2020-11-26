#
# Class for constant porosity
#
import pybamm

from .base_active_material import BaseModel


class Constant(BaseModel):
    """Submodel for constant porosity

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'


    **Extends:** :class:`pybamm.porosity.BaseModel`
    """

    def get_fundamental_variables(self):
        if self.domain == "Negative":
            x_n = pybamm.standard_spatial_vars.x_n
            eps_solid = self.param.epsilon_s_n(x_n)
            deps_solid_dt = pybamm.FullBroadcast(
                0, "negative electrode", "current collector"
            )
        elif self.domain == "Positive":
            x_p = pybamm.standard_spatial_vars.x_p
            eps_solid = self.param.epsilon_s_p(x_p)
            deps_solid_dt = pybamm.FullBroadcast(
                0, "positive electrode", "current collector"
            )

        variables = self._get_standard_active_material_variables(eps_solid)
        variables.update(
            self._get_standard_active_material_change_variables(deps_solid_dt)
        )

        return variables
