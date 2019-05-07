#
# Equation classes for the open-circuit potentials and reaction overpotentials
#
import pybamm


class Potential(pybamm.SubModel):
    """Compute open-circuit potentials and reaction overpotentials

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def get_derived_open_circuit_potentials(self, ocp_n, ocp_p):
        """
        Compute open-circuit potentials (dimensionless and dimensionless). Note that for
        this submodel, we must specify explicitly which concentration we are using to
        calculate the open-circuit potential.

        Parameters
        ----------
        ocp_n : :class:`pybamm.Symbol`
            Dimensionless negative electrode open-circuit potential
        ocp_p : :class:`pybamm.Symbol`
            Dimensionless positive electrode open-circuit potential
        """
        # Load parameters and spatial variables
        param = self.set_of_parameters

        # Dimensionless
        ocp_n_av = pybamm.average(ocp_n)
        ocp_p_av = pybamm.average(ocp_p)
        ocp_n_left = pybamm.boundary_value(ocp_n, "left")
        ocp_p_right = pybamm.boundary_value(ocp_p, "right")
        ocv_av = ocp_p_av - ocp_n_av
        ocv = ocp_p_right - ocp_n_left

        # Dimensional
        ocp_n_dim = param.U_n_ref + param.potential_scale * ocp_n
        ocp_p_dim = param.U_p_ref + param.potential_scale * ocp_p
        ocp_n_av_dim = param.U_n_ref + param.potential_scale * ocp_n_av
        ocp_p_av_dim = param.U_p_ref + param.potential_scale * ocp_p_av
        ocp_n_left_dim = param.U_n_ref + param.potential_scale * ocp_n_left
        ocp_p_right_dim = param.U_p_ref + param.potential_scale * ocp_p_right
        ocv_av_dim = ocp_p_av_dim - ocp_n_av_dim
        ocv_dim = ocp_p_right_dim - ocp_n_left_dim

        # Variables
        return {
            "Negative electrode open circuit potential": ocp_n,
            "Positive electrode open circuit potential": ocp_p,
            "Average negative electrode open circuit potential": ocp_n_av,
            "Average positive electrode open circuit potential": ocp_p_av,
            "Average open circuit voltage": ocv_av,
            "Measured open circuit voltage": ocv,
            "Negative electrode open circuit potential [V]": ocp_n_dim,
            "Positive electrode open circuit potential [V]": ocp_p_dim,
            "Average negative electrode open circuit potential [V]": ocp_n_av_dim,
            "Average positive electrode open circuit potential [V]": ocp_p_av_dim,
            "Average open circuit voltage [V]": ocv_av_dim,
            "Measured open circuit voltage [V]": ocv_dim,
        }

    def get_derived_reaction_overpotentials(self, eta_r_n, eta_r_p):
        """
        Compute reaction overpotentials (dimensionless and dimensionless).

        Parameters
        ----------
        eta_r_n : :class:`pybamm.Symbol`
            Dimensionless negative electrode reaction overpotential
        eta_r_p : :class:`pybamm.Symbol`
            Dimensionless positive electrode reaction overpotential
        """
        # Load parameters
        param = self.set_of_parameters

        # Derived and dimensional reaction overpotentials
        eta_r_n_av = pybamm.average(eta_r_n)
        eta_r_p_av = pybamm.average(eta_r_p)
        eta_r_av = eta_r_p_av - eta_r_n_av

        eta_r_n_dim = param.potential_scale * eta_r_n
        eta_r_p_dim = param.potential_scale * eta_r_p
        eta_r_n_av_dim = param.potential_scale * eta_r_n_av
        eta_r_p_av_dim = param.potential_scale * eta_r_p_av
        eta_r_av_dim = param.potential_scale * eta_r_av

        # Update variables
        return {
            "Negative reaction overpotential": eta_r_n,
            "Positive reaction overpotential": eta_r_p,
            "Average negative reaction overpotential": eta_r_n_av,
            "Average positive reaction overpotential": eta_r_p_av,
            "Average reaction overpotential": eta_r_av,
            "Negative reaction overpotential [V]": eta_r_n_dim,
            "Positive reaction overpotential [V]": eta_r_p_dim,
            "Average negative reaction overpotential [V]": eta_r_n_av_dim,
            "Average positive reaction overpotential [V]": eta_r_p_av_dim,
            "Average reaction overpotential [V]": eta_r_av_dim,
        }
