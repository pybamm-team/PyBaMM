#
# Base class for the electrode
#
import pybamm


class BaseElectrode(pybamm.BaseSubModel):
    """Base class for conservation of current for the current in the electrodes.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param)
        self._domain = domain

    def _get_standard_potential_variables(self, phi_s):

        param = self.param
        phi_s_av = pybamm.average(phi_s)

        if self._domain == "Negative":
            phi_s_dim = param.potential_scale * phi_s
            phi_s_av_dim = param.potential_scale * phi_s_av
            delta_phi_s = phi_s

        elif self._domain == "Positive":
            phi_s_dim = param.U_p_ref - param.U_n_ref + param.potential_scale * phi_s
            phi_s_av_dim = (
                param.U_p_ref - param.U_n_ref + param.potential_scale * phi_s_av
            )
            V = pybamm.BoundaryValue(phi_s, "right")
            V_dim = param.U_p_ref - param.U_n_ref + param.potential_scale * V
            delta_phi_s = phi_s - V

        delta_phi_s_av = pybamm.average(delta_phi_s)
        delta_phi_s_dim = delta_phi_s * param.potential_scale
        delta_phi_s_av_dim = delta_phi_s_av * param.potential_scale

        variables = {
            self._domain + " electrode potential": phi_s,
            self._domain + " electrode potential [V]": phi_s_dim,
            "Average " + self._domain.lower() + " electrode potential": phi_s_av,
            "Average "
            + self._domain.lower()
            + " electrode potential [V]": phi_s_av_dim,
            self._domain + " electrode ohmic losses": delta_phi_s,
            self._domain + " electrode ohmic losses [V]": delta_phi_s_dim,
            "Average "
            + self._domain.lower()
            + " electrode ohmic losses": delta_phi_s_av,
            "Average "
            + self._domain.lower()
            + " electrode ohmic losses [V]": delta_phi_s_av_dim,
        }

        if self._domain == "Positive":
            variables.update({"Voltage": V_dim})

        return variables

    def _get_standard_current_variables(self, i_s):
        param = self.param
        i_s_av = pybamm.average(i_s)

        i_s_dim = param.i_typ * i_s
        i_s_av_dim = param.i_typ * i_s_av

        variables = {
            self._domain + " electrode current density": i_s,
            "Average " + self._domain.lower() + " electrode current density": i_s_av,
            self._domain + " electrode current density [A.m-2]": i_s_dim,
            "Average "
            + self._domain.lower()
            + " electrode current density [A.m-2]": i_s_av_dim,
        }

        return variables

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        return pybamm.ScikitsDaeSolver()
