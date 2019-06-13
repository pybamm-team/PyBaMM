#
# Base class for current collectors
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Base class for current collectors

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param)
        self._domain = domain

    def _get_standard_potential_variables(self, phi_cc):

        pot_scale = self.param.potential_scale

        # add more to this
        variables = {
            self._domain + "current collector potential": phi_cc,
            self._domain + " current collector potential [V]": phi_cc * pot_scale,
        }

        return variables

    def _get_standard_current_variables(self, i_cc, i_boundary_cc):

        # just need this to get 1D models working for now
        variables = {"Current collector current density": i_boundary_cc}

        return variables

