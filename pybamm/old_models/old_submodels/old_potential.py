#
# Equation classes for the open-circuit potentials and reaction overpotentials
#
import pybamm


class OldPotential(pybamm.OldSubModel):
    """Compute open-circuit potentials and reaction overpotentials

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def get_derived_potential(self, pot_n, pot_p, name, dimensional_shift):
        """
        Compute derived potentials (dimensionless and dimensionless).

        Parameters
        ----------
        pot_n : :class:`pybamm.Symbol`
            Dimensionless negative electrode overpotential
        pot_p : :class:`pybamm.Symbol`
            Dimensionless positive electrode overpotential
        name : str
            Name to assign to the symbol
        dimensional_shift : bool, optional
            Whether to shift the symbols by a constant when re-dimensionalising. Default
            is False.
        """
        # Load parameters
        param = self.set_of_parameters

        # Define dimensional shifts
        shift_n = dimensional_shift * param.U_n_ref
        shift_p = dimensional_shift * param.U_p_ref

        # Broadcast if necessary
        if pot_n.domain in [[], ["current collector"]]:
            pot_n = pybamm.Broadcast(pot_n, ["negative electrode"])
        if pot_p.domain in [[], ["current collector"]]:
            pot_p = pybamm.Broadcast(pot_p, ["positive electrode"])

        # Derived and dimensional potentials
        pot_n_av = pybamm.average(pot_n)
        pot_p_av = pybamm.average(pot_p)
        pot_av = pot_p_av - pot_n_av

        pot_n_dim = shift_n + param.potential_scale * pot_n
        pot_p_dim = shift_p + param.potential_scale * pot_p
        pot_n_av_dim = shift_n + param.potential_scale * pot_n_av
        pot_p_av_dim = shift_p + param.potential_scale * pot_p_av
        pot_av_dim = pot_p_av_dim - pot_n_av_dim

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

    def get_derived_open_circuit_potentials(self, ocp_n, ocp_p):
        " See :meth:`pybamm.Potential.get_derived_potential()`"
        vars = self.get_derived_potential(ocp_n, ocp_p, "open circuit potential", True)

        # Get open-circuit voltage
        param = self.set_of_parameters
        ocp_n = vars["Negative electrode open circuit potential"]
        ocp_p = vars["Positive electrode open circuit potential"]
        ocp_n_left = pybamm.boundary_value(ocp_n, "left")
        ocp_p_right = pybamm.boundary_value(ocp_p, "right")
        ocv = ocp_p_right - ocp_n_left
        ocv_dim = param.U_p_ref - param.U_n_ref + param.potential_scale * ocv

        # Change name for average from "potential" to "voltage"
        ocv_av = vars["Average open circuit potential"]
        ocv_av_dim = vars["Average open circuit potential [V]"]

        return {
            **vars,
            "Measured open circuit voltage": ocv,
            "Average open circuit voltage": ocv_av,
            "Measured open circuit voltage [V]": ocv_dim,
            "Average open circuit voltage [V]": ocv_av_dim,
        }

    def get_derived_reaction_overpotentials(self, eta_r_n, eta_r_p):
        " See :meth:`pybamm.Potential.get_derived_potential()`"
        return self.get_derived_potential(
            eta_r_n, eta_r_p, "reaction overpotential", False
        )

    def get_derived_surface_potential_differences(self, delta_phi_n, delta_phi_p):
        " See :meth:`pybamm.Potential.get_derived_potential()`"
        return self.get_derived_potential(
            delta_phi_n, delta_phi_p, "surface potential difference", True
        )

    def get_all_potentials(self, ocp, eta_r=None, delta_phi=None):
        """
        Calculate all three of open-circuit potential, reaction overpotential and
        surface potential difference, using only two of them.
        """
        ocp_n, ocp_p = ocp
        if delta_phi is not None and eta_r is not None:
            eta_r_n, eta_r_p = eta_r
            delta_phi_n, delta_phi_p = delta_phi
        elif delta_phi is None and eta_r is not None:
            eta_r_n, eta_r_p = eta_r
            delta_phi_n = eta_r_n + ocp_n
            delta_phi_p = eta_r_p + ocp_p
        elif eta_r is None and delta_phi is not None:
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
