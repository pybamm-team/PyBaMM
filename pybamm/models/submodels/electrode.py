#
# Equation classes for the electrode
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Ohm(pybamm.SubModel):
    """Ohm's law + conservation of current for the current in the electrodes.

    Parameters
    ----------
    phi : :class:`pybamm.Symbol`
        The electric potential in the electrodes ("electrode potential")
    j : :class:`pybamm.Symbol`
        An expression tree that represents the interfacial current density at the
        electrode-electrolyte interface
    param : parameter class
        The parameters to use for this submodel
    epsilon : :class:`pybamm.Symbol`
        The (electrolyte/liquid phase) porosity (optional)

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_algebraic_system(self, phi_s, reactions, eps=None):
        """
        PDE system for current in the electrodes, using Ohm's law

        Parameters
        ----------
        phi_s : :class:`pybamm.Variable`
            The eletrode potential variable
        variables : dict
            Dictionary of {string: :class:`pybamm.Symbol`}, which can be read to find
            already-calculated variables
        """
        param = self.set_of_parameters
        icell = param.current_with_time

        # algebraic model only
        self.rhs = {}

        # different bounday conditions in each electrode
        if phi_s.domain == ["negative electrode"]:
            j = reactions["main"]["neg"]["aj"]
            # if porosity is not provided, use the input parameter
            if eps is None:
                eps = param.epsilon_n
            # liion sigma_n may already account for porosity
            i_s_n = -param.sigma_n * (1 - eps) ** param.b * pybamm.grad(phi_s)
            self.algebraic = {phi_s: pybamm.div(i_s_n) + j}
            self.boundary_conditions = {phi_s: {"left": 0}, i_s_n: {"right": 0}}
            self.initial_conditions = {phi_s: 0}
            self.variables = {
                "Negative electrode potential": phi_s,
                "Negative electrode current density": i_s_n,
            }
        elif phi_s.domain == ["positive electrode"]:
            j = reactions["main"]["pos"]["aj"]
            # if porosity is not provided, use the input parameter
            if eps is None:
                eps = param.epsilon_p
            # liion sigma_p may already account for porosity
            i_s_p = -param.sigma_p * (1 - eps) ** param.b * pybamm.grad(phi_s)
            self.algebraic = {phi_s: pybamm.div(i_s_p) + j}
            self.boundary_conditions = {i_s_p: {"left": 0, "right": icell}}
            self.initial_conditions = {
                phi_s: param.U_p(param.c_p_init) - param.U_n(param.c_n_init)
            }
            self.variables.update(
                {
                    "Positive electrode potential": phi_s,
                    "Positive electrode current density": i_s_p,
                }
            )
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(phi_s.domain))

        # Set default solver to DAE
        self.default_solver = pybamm.ScikitsDaeSolver()

    def get_explicit_leading_order(self, ocp_p, eta_r_p, phi_e):
        """
        Provides the leading order explicit solution to solid phase current
        conservation with ohm's law.

        Parameters
        ----------
        variables : dict
            Dictionary of {string: :class:`pybamm.Symbol`}, which can be read to find
            already-calculated variables

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        # import parameters and spatial variables
        param = self.set_of_parameters
        l_n = param.l_n
        l_p = param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        # define current
        i_cell = param.current_with_time

        # Take boundary values
        phi_e_right = pybamm.BoundaryValue(phi_e, "right")

        # electode potential
        phi_s_n = pybamm.Broadcast(0, ["negative electrode"])
        v = ocp_p + eta_r_p + phi_e_right
        phi_s_p = pybamm.Broadcast(v, ["positive electrode"])

        # electrode current
        i_s_n = i_cell - i_cell * x_n / l_n
        i_s_p = i_cell - i_cell * (1 - x_p) / l_p

        delta_phi_s_av = pybamm.Scalar(0)

        return self.get_variables(phi_s_n, phi_s_p, i_s_n, i_s_p, delta_phi_s_av)

    def get_explicit_combined(self, ocp_p, eta_r_p, phi_e, epsilon=None):
        """
        Provides an explicit combined leading and first order solution to solid phase
        current conservation with ohm's law. Note that the returned current density is
        only the leading order approximation.

        Parameters
        ----------
        variables : dict
            Dictionary of {string: :class:`pybamm.Symbol`}, which can be read to find
            already-calculated variables

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        # import parameters and spatial vairables
        param = self.set_of_parameters
        l_n = param.l_n
        l_p = param.l_p
        i_cell = param.current_with_time
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        # if porosity is not provided, use the input parameter
        if epsilon is None:
            epsilon = param.epsilon
        eps_n, eps_s, eps_p = [e.orphans[0] for e in epsilon.orphans]

        # extract right-most ocp, overpotential, and electrolyte potential
        if ocp_p.domain == []:
            ocp_p = pybamm.Broadcast(ocp_p, ["positive electrode"])
        if eta_r_p.domain == []:
            eta_r_p = pybamm.Broadcast(eta_r_p, ["positive electrode"])
        if phi_e.domain == []:
            phi_e = pybamm.Broadcast(phi_e, ["positive electrode"])

        ocp_p_right = pybamm.BoundaryValue(ocp_p, "right")
        eta_r_p_right = pybamm.BoundaryValue(eta_r_p, "right")
        phi_e_right = pybamm.BoundaryValue(phi_e, "right")

        # electrode potential
        sigma_n_eff = param.sigma_n * (1 - eps_n)
        sigma_p_eff = param.sigma_p * (1 - eps_p)
        phi_s_n = i_cell * x_n * (2 * l_n - x_n) / (2 * sigma_n_eff * l_n)
        phi_s_p = (ocp_p_right + eta_r_p_right + phi_e_right) + i_cell * (
            (1 - x_p) * (1 - 2 * l_p - x_p) / (2 * sigma_p_eff * l_p)
        )

        # electrode current
        i_s_n = i_cell - i_cell * x_n / l_n
        i_s_p = i_cell - i_cell * (1 - x_p) / l_p

        delta_phi_s_av = (
            -i_cell
            / 3
            * (l_p / param.sigma_p / (1 - eps_p) + l_n / param.sigma_n / (1 - eps_n))
        )

        return self.get_variables(phi_s_n, phi_s_p, i_s_n, i_s_p, delta_phi_s_av)

    def get_post_processed(self, phi_s_n, phi_s_p, i_s_n, i_s_p):
        """
        Calculate dimensionless and dimensional variables for the electrode submodel

        Parameters
        ----------
        variables : dict
            Dictionary of {string: :class:`pybamm.Symbol`}, which can be read to find
            already-calculated variables

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        param = self.set_of_parameters
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        delta_phi_s_n = phi_s_n - pybamm.BoundaryValue(phi_s_n, "left")
        delta_phi_s_n_av = pybamm.Integral(delta_phi_s_n, x_n) / param.l_n
        delta_phi_s_p = phi_s_p - pybamm.BoundaryValue(phi_s_p, "right")
        delta_phi_s_p_av = pybamm.Integral(delta_phi_s_p, x_p) / param.l_p
        delta_phi_s_av = delta_phi_s_p_av - delta_phi_s_n_av

        return self.get_variables(phi_s_n, phi_s_p, i_s_n, i_s_p, delta_phi_s_av)

    def get_variables(self, phi_s_n, phi_s_p, i_s_n, i_s_p, delta_phi_s_av):
        """
        Calculate dimensionless and dimensional variables for the electrode submodel

        Parameters
        ----------
        phi_s :class:`pybamm.Concatenation`
            The electrode potential
        i_s :class:`pybamm.Concatenation`
            The electrode current density

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        param = self.set_of_parameters
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        # Unpack
        phi_s_s = pybamm.Broadcast(0, ["separator"])  # can we put NaN?
        phi_s = pybamm.Concatenation(phi_s_n, phi_s_s, phi_s_p)
        i_s_s = pybamm.Broadcast(0, ["separator"])  # can we put NaN?
        i_s = pybamm.Concatenation(i_s_n, i_s_s, i_s_p)

        # Voltage variable
        v = pybamm.BoundaryValue(phi_s_p, "right") - pybamm.BoundaryValue(
            phi_s_n, "left"
        )

        # Dimensional
        phi_s_n_dim = param.potential_scale * phi_s_n
        phi_s_s_dim = pybamm.Broadcast(0, ["separator"])
        phi_s_p_dim = param.U_p_ref - param.U_n_ref + param.potential_scale * phi_s_p
        phi_s_dim = pybamm.Concatenation(phi_s_n_dim, phi_s_s_dim, phi_s_p_dim)
        i_s_n_dim = param.i_typ * i_s_n
        i_s_p_dim = param.i_typ * i_s_p
        i_s_dim = param.i_typ * i_s
        delta_phi_s_av_dim = param.potential_scale * delta_phi_s_av
        v_dim = param.potential_scale * v

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
            "Negative electrode current density [A m-2]": i_s_n_dim,
            "Positive electrode current density [A m-2]": i_s_p_dim,
            "Electrode current density [A m-2]": i_s_dim,
            "Average solid phase ohmic losses [V]": delta_phi_s_av_dim,
            "Terminal voltage [V]": v_dim,
        }
