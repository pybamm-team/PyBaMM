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

    def get_standard_derived_variables(self, derived_variables):

        if self._domain == "Positive":
            phi_s = derived_variables["Positive electrode potential"]
            V = pybamm.boundary_value(phi_s, "right")
            derived_variables.update({"Voltage": V})

        derived_variables.update(self.get_average_variables(derived_variables))
        derived_variables.update(self.get_dimensional_variables(derived_variables))

        return derived_variables

    def get_average_variables(self, variables):
        """
        Calculates averaged variables.

        Parameters
        ----------
        variables : dict
            A dictionary containing all the dimensionless variables in the
            model
        
        Returns
        -------
        dict
            A dictionary with all the average variables.
        """

        phi_s = variables[self._domain + " electrode potential"]
        i_s = variables[self._domain + " electrode current density"]

        if self._domain == "Negative":
            delta_phi_av = pybamm.average(phi_s)

        elif self._domain == "Positive":
            V = variables["Voltage"]
            delta_phi_av = pybamm.average(phi_s - V)

        average_variables = {
            "Average "
            + self._domain.lower()
            + " electrode potential": pybamm.average(phi_s),
            "Average "
            + self._domain.lower()
            + " electrode current density": pybamm.average(i_s),
            "Average " + self._domain.lower() + " electrode ohmic losses": delta_phi_av,
        }

        return average_variables

    def get_dimensional_variables(self, variables):
        """
        Calculates dimensional variables.

        Parameters
        ----------
        variables : dict
            A dictionary containing all the dimensionless variables in the
            model
        
        Returns
        -------
        dict
            A dictionary with all the dimensional variables.
        """
        param = self.param

        phi_s = variables[self._domain + " electrode potential"]
        i_s = variables[self._domain + " electrode current density"]
        phi_s_av = variables["Average " + self._domain.lower() + " electrode potential"]
        i_s_av = variables[
            "Average " + self._domain.lower() + " electrode current density"
        ]
        delta_phi_av = variables[
            "Average " + self._domain.lower() + " electrode ohmic losses"
        ]

        if self._domain == "Negative":
            phi_s_dim = param.potential_scale * phi_s
            phi_s_av_dim = param.potential_scale * phi_s_av

        elif self._domain == "Positive":
            phi_s_dim = param.U_p_ref - param.U_n_ref + param.potential_scale * phi_s
            phi_s_av_dim = (
                param.U_p_ref - param.U_n_ref + param.potential_scale * phi_s_av
            )
            V = variables["Voltage"]
            V_dim = param.U_p_ref - param.U_n_ref + param.potential_scale * V

        i_s_dim = param.i_typ * i_s
        i_s_av_dim = param.i_typ * i_s_av
        delta_phi_av_dim = param.potential_scale * delta_phi_av

        dimensional_variables = {
            self._domain + " electrode potential [V]": phi_s_dim,
            self._domain + " electrode current density [A.m-2]": i_s_dim,
            "Average "
            + self._domain.lower()
            + " electrode potential [V]": phi_s_av_dim,
            "Average "
            + self._domain.lower()
            + " electrode current density [A.m-2]": i_s_av_dim,
            "Average "
            + self._domain.lower()
            + " electrode ohmic losses [V]": delta_phi_av_dim,
        }

        if self._domain == "Positive":
            dimensional_variables.update({"Voltage": V_dim})

        return dimensional_variables

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        return pybamm.ScikitsDaeSolver()
