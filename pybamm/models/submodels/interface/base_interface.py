#
# Base interface class
#

import pybamm


class BaseInterface(pybamm.BaseSubModel):
    """
    Base class for interfacial currents

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param)
        self._domain = domain

    def _get_standard_interfacial_current_variables(self, j, j_av):

        i_typ = self.param.i_typ

        variables = {
            self._domain + " electrode interfacial current density": j,
            "Average "
            + self._domain.lower()
            + " electrode interfacial current density": j_av,
            self._domain + " interfacial current density [A.m-2]": i_typ * j,
            "Average "
            + self._domain.lower()
            + " electrode interfacial current density [A.m-2]": i_typ * j_av,
        }

        return variables

    def _get_standard_exchange_current_variables(self, j0, j0_av):

        i_typ = self.param.i_typ

        variables = {
            self._domain + " electrode exchange current density": j0,
            "Average "
            + self._domain.lower()
            + " electrode exchange current density": j0_av,
            self._domain + " exchange current density [A.m-2]": i_typ * j0,
            "Average "
            + self._domain.lower()
            + " electrode exchange current density [A.m-2]": i_typ * j0_av,
        }

        return variables

    def _get_standard_overpotential_variables(self, eta_r, eta_r_av):

        pot_scale = self.param.potential_scale

        variables = {
            self._domain + " reaction overpotential": eta_r,
            "Average " + self._domain.lower() + " reaction overpotential": eta_r_av,
            self._domain + " reaction overpotential [V]": eta_r * pot_scale,
            "Average "
            + self._domain.lower()
            + " reaction overpotential [V]": eta_r_av * pot_scale,
        }

        return variables

    def _get_standard_surface_potential_difference_variables(
        self, delta_phi, delta_phi_av
    ):

        pot_scale = self.param.potential_scale

        variables = {
            self._domain + " electrode surface potential difference": delta_phi,
            "Average "
            + self._domain.lower()
            + " electrode surface potential difference": delta_phi_av,
            self._domain
            + " electrode surface potential difference [V]": delta_phi * pot_scale,
            "Average "
            + self._domain.lower()
            + " electrode surface potential difference [V]": delta_phi_av * pot_scale,
        }
