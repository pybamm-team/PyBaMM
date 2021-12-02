#
# Class for constant interface utilisation
#
import pybamm

from .base_utilisation import BaseModel


class Constant(BaseModel):
    """Submodel for constant interface utilisation

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'Negative' or 'Positive'
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.interface_utilisation.BaseModel`
    """

    def get_fundamental_variables(self):
        if self.domain == "Negative":
            u_av = self.param.u_n_init
        else:
            u_av = self.param.u_p_init
        u = pybamm.FullBroadcast(
            u_av, self.domain.lower() + " electrode", "current collector"
        )

        variables = self._get_standard_interface_utilisation_variables(u)
        return variables
