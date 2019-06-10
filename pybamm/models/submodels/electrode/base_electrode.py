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

        phi_s = variables[self._domain + " potential"]
        i_s = variables[self._domain + " current density"]

        average_variables = {
            "Average " + self._domain.lower() + " potential": pybamm.average(phi_s),
            "Average " + self._domain.lower() + " current density": pybamm.average(i_s),
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

        phi_s = variables[self._domain + " potential"]
        i_s = variables[self._domain + " current density"]
        phi_s_av = variables["Average " + self._domain.lower() + " potential"]
        i_s_av = variables["Average " + self._domain.lower() + " current density"]

        if self._domain == "Negative electrode":
            phi_s_dim = param.potential_scale * phi_s
            phi_s_av_dim = param.potential_scale * phi_s_av
        elif self._domain == "Positive electrode":
            phi_s_dim = param.U_p_ref - param.U_n_ref + param.potential_scale * phi_s
            phi_s_av_dim = (
                param.U_p_ref - param.U_n_ref + param.potential_scale * phi_s_av
            )

        i_s_dim = param.i_typ * i_s
        i_s_av_dim = param.i_typ * i_s_av

        dimensional_variables = {
            self._domain + " potential [V]": phi_s_dim,
            self._domain + " current density [A.m-2]": i_s_dim,
            "Average " + self._domain.lower() + " potential [V]": phi_s_av_dim,
            "Average " + self._domain.lower() + " current density [A.m-2]": i_s_av_dim,
        }

        return dimensional_variables

    def get_output_variables_old(
        self, phi_s_n, phi_s_p, i_s_n, i_s_p, delta_phi_s_av=None
    ):
        """
        Calculate dimensionless and dimensional variables for the electrode submodel

        Parameters
        ----------
        phi_s_n : :class:`pybamm.Symbol`
            The electrode potential in the negative electrode
        phi_s_p : :class:`pybamm.Symbol`
            The electrode potential in the positive electrode
        i_s_n : :class:`pybamm.Symbol`
            The electrode current density in the negative electrode
        i_s_p : :class:`pybamm.Symbol`
            The electrode current density in the positive electrode
        delta_phi_s_av : :class:`pybamm,Symbol`, optional
            Average solid phase Ohmic losses. Default is None, in which case
            delta_phi_s_av is calculated from phi_s_n and phi_s_p

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        param = self.set_of_parameters

        if delta_phi_s_av is None:
            delta_phi_s_n = phi_s_n - pybamm.boundary_value(phi_s_n, "left")
            delta_phi_s_n_av = pybamm.average(delta_phi_s_n)
            delta_phi_s_p = phi_s_p - pybamm.boundary_value(phi_s_p, "right")
            delta_phi_s_p_av = pybamm.average(delta_phi_s_p)
            delta_phi_s_av = delta_phi_s_p_av - delta_phi_s_n_av

        # Unpack
        phi_s_s = pybamm.Broadcast(0, ["separator"])  # can we put NaN?
        phi_s = pybamm.Concatenation(phi_s_n, phi_s_s, phi_s_p)
        i_s_s = pybamm.Broadcast(0, ["separator"])  # can we put NaN?
        i_s = pybamm.Concatenation(i_s_n, i_s_s, i_s_p)

        # Voltage variable
        v = pybamm.boundary_value(phi_s_p, "right")

        # Dimensional
        phi_s_n_dim = param.potential_scale * phi_s_n
        phi_s_s_dim = pybamm.Broadcast(0, ["separator"])
        phi_s_p_dim = param.U_p_ref - param.U_n_ref + param.potential_scale * phi_s_p
        phi_s_dim = pybamm.Concatenation(phi_s_n_dim, phi_s_s_dim, phi_s_p_dim)
        i_s_n_dim = param.i_typ * i_s_n
        i_s_p_dim = param.i_typ * i_s_p
        i_s_dim = param.i_typ * i_s
        delta_phi_s_av_dim = param.potential_scale * delta_phi_s_av
        v_dim = param.U_p_ref - param.U_n_ref + param.potential_scale * v

        # Update variables
        return {
            "Negative electrode potential": phi_s_n,
            "Positive electrode potential": phi_s_p,
            "Electrode potential": phi_s,
            "Negative electrode current density": i_s_n,
            "Positive electrode current density": i_s_p,
            "Electrode current density": i_s,
            "Average solid phase ohmic losses": delta_phi_s_av,
            "Terminal voltage": v,
            "Negative electrode potential [V]": phi_s_n_dim,
            "Positive electrode potential [V]": phi_s_p_dim,
            "Electrode potential [V]": phi_s_dim,
            "Negative electrode current density [A.m-2]": i_s_n_dim,
            "Positive electrode current density [A.m-2]": i_s_p_dim,
            "Electrode current density [A.m-2]": i_s_dim,
            "Average solid phase ohmic losses [V]": delta_phi_s_av_dim,
            "Terminal voltage [V]": v_dim,
        }

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        return pybamm.ScikitsDaeSolver()
