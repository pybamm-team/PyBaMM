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

        # Broadcast if necessary
        if ocp_n.domain in [[], ["current collector"]]:
            ocp_n = pybamm.Broadcast(ocp_n, ["negative electrode"])
        if ocp_p.domain in [[], ["current collector"]]:
            ocp_p = pybamm.Broadcast(ocp_p, ["positive electrode"])

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

    def get_derived_potential(self, pot_n, pot_p, name):
        """
        Compute reaction overpotentials (dimensionless and dimensionless).

        Parameters
        ----------
        pot_n : :class:`pybamm.Symbol`
            Dimensionless negative electrode reaction overpotential
        pot_p : :class:`pybamm.Symbol`
            Dimensionless positive electrode reaction overpotential
        """
        # Load parameters
        param = self.set_of_parameters

        # Broadcast if necessary
        if pot_n.domain in [[], ["current collector"]]:
            pot_n = pybamm.Broadcast(pot_n, ["negative electrode"])
        if pot_p.domain in [[], ["current collector"]]:
            pot_p = pybamm.Broadcast(pot_p, ["positive electrode"])

        # Derived and dimensional reaction overpotentials
        pot_n_av = pybamm.average(pot_n)
        pot_p_av = pybamm.average(pot_p)
        pot_av = pot_p_av - pot_n_av

        pot_n_dim = param.potential_scale * pot_n
        pot_p_dim = param.potential_scale * pot_p
        pot_n_av_dim = param.potential_scale * pot_n_av
        pot_p_av_dim = param.potential_scale * pot_p_av
        pot_av_dim = param.potential_scale * pot_av

        # Update variables
        return {
            "Negative electrode " + name: pot_n,
            "Positive electrode " + name: pot_p,
            "Average negative electrode " + name: pot_n_av,
            "Average positive electrode " + name: pot_p_av,
            "Average " + name: pot_av,
            "Negative electrode " + name + " [V]": pot_n_dim,
            "Positive electrode " + name + " [V]": pot_p_dim,
            "Average negative electrode " + name + " [V]": pot_n_av_dim,
            "Average positive electrode " + name + " [V]": pot_p_av_dim,
            "Average " + name + " [V]": pot_av_dim,
        }

    def get_derived_reaction_overpotentials(self, eta_r_n, eta_r_p):
        " See :meth:`pybamm.Potential.get_derived_potential()`"
        return self.get_derived_potential(eta_r_n, eta_r_p, "reaction overpotential")

    def get_derived_surface_potential_differences(self, delta_phi_n, delta_phi_p):
        " See :meth:`pybamm.Potential.get_derived_potential()`"
        return self.get_derived_potential(
            delta_phi_n, delta_phi_p, "surface potential difference"
        )

    def get_all_potentials(self, ocp, eta_r=None, delta_phi=None):
        ocp_n, ocp_p = ocp
        if eta_r is not None:
            eta_r_n, eta_r_p = eta_r
            delta_phi_n = eta_r_n + ocp_n
            delta_phi_p = eta_r_p + ocp_p
        elif delta_phi is not None:
            delta_phi_n, delta_phi_p = delta_phi
            eta_r_n = delta_phi_n - ocp_n
            eta_r_p = delta_phi_p - ocp_p
        else:
            raise ValueError("eta_r and delta_phi cannot both be None")
        ocp_vars = self.get_derived_open_circuit_potentials(ocp_n, ocp_p)
        eta_r_vars = self.get_derived_reaction_overpotentials(eta_r_n, eta_r_p)
        delta_phi_vars = self.get_derived_surface_potential_differences(
            delta_phi_n, delta_phi_p
        )
        return {**ocp_vars, **eta_r_vars, **delta_phi_vars}
