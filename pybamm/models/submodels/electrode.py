#
# Equation classes for the electrode
#
import pybamm


class Ohm(pybamm.SubModel):
    """Ohm's law + conservation of current for the current in the electrodes.

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_algebraic_system(self, variables, reactions, domain):
        """
        PDE system for current in the electrodes, using Ohm's law

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        reactions : dict
            Dictionary of reaction variables
        domain : list of str
            Domain in which to set the system

        """
        # unpack variables
        i_boundary_cc = variables["Current collector current density"]
        param = self.set_of_parameters

        # different bounday conditions in each electrode
        if domain == ["negative electrode"]:
            phi_s = variables["Negative electrode potential"]
            j = reactions["main"]["neg"]["aj"]
            # if porosity is not provided, use the input parameter
            try:
                eps = variables["Porosity"].orphans[0]
            except KeyError:
                eps = param.epsilon_n
            # liion sigma_n may already account for porosity
            conductivity = param.sigma_n * (1 - eps) ** param.b
            lbc = (pybamm.Scalar(0), "Dirichlet")
            rbc = (pybamm.Scalar(0), "Neumann")
            self.initial_conditions[phi_s] = pybamm.Scalar(0)
        elif domain == ["positive electrode"]:
            phi_s = variables["Positive electrode potential"]
            j = reactions["main"]["pos"]["aj"]
            # if porosity is not provided, use the input parameter
            try:
                eps = variables["Porosity"].orphans[2]
            except KeyError:
                eps = param.epsilon_p
            # liion sigma_p may already account for porosity
            conductivity = param.sigma_p * (1 - eps) ** param.b
            lbc = (pybamm.Scalar(0), "Neumann")
            rbc = (
                i_boundary_cc / pybamm.boundary_value(-conductivity, "right"),
                "Neumann",
            )
            self.initial_conditions[phi_s] = param.U_p(param.c_p_init) - param.U_n(
                param.c_n_init
            )

        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))

        i_s = -conductivity * pybamm.grad(phi_s)
        self.algebraic[phi_s] = pybamm.div(i_s) + j
        self.boundary_conditions[phi_s] = {"left": lbc, "right": rbc}
        self.variables.update(
            {
                domain[0].capitalize() + " potential": phi_s,
                domain[0].capitalize() + " current density": i_s,
            }
        )

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        return pybamm.ScikitsDaeSolver()

    def get_explicit_leading_order(self, variables):
        """
        Provides the leading order explicit solution to solid phase current
        conservation with ohm's law.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        # unpack variables
        ocp_p = variables["Positive electrode open circuit potential"]
        eta_r_p = variables["Positive reaction overpotential"]
        phi_e = variables["Electrolyte potential"]
        i_boundary_cc = variables["Current collector current density"]

        # import parameters and spatial variables
        param = self.set_of_parameters
        l_n = param.l_n
        l_p = param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        # Take boundary values
        phi_e_right = pybamm.boundary_value(phi_e, "right")
        ocp_p_right = pybamm.boundary_value(ocp_p, "right")
        eta_r_p_right = pybamm.boundary_value(eta_r_p, "right")

        # electode potential
        phi_s_n = pybamm.Broadcast(0, ["negative electrode"])
        v = ocp_p_right + eta_r_p_right + phi_e_right
        phi_s_p = pybamm.Broadcast(v, ["positive electrode"])

        # electrode current
        i_s_n = pybamm.outer(i_boundary_cc, 1 - x_n / l_n)
        i_s_p = pybamm.outer(i_boundary_cc, 1 - (1 - x_p) / l_p)

        delta_phi_s_av = pybamm.Scalar(0)

        return self.get_variables(phi_s_n, phi_s_p, i_s_n, i_s_p, delta_phi_s_av)

    def get_neg_pot_explicit_combined(self, variables):
        """
        Provides an explicit combined leading and first order solution to solid phase
        current conservation with ohm's law in the negative electrode.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model

        Returns
        -------
        phi_s_n :class `pybamm.Symbol`
            The negative electrode potential
        """
        # import parameters and spatial variables
        param = self.set_of_parameters
        l_n = param.l_n
        x_n = pybamm.standard_spatial_vars.x_n

        # Unpack variables
        i_boundary_cc = variables["Current collector current density"]
        try:
            eps_n = variables["Porosity"].orphans[0]
        except KeyError:
            eps_n = param.epsilon_n

        # electrode potential
        sigma_n_eff = param.sigma_n * (1 - eps_n)
        phi_s_n = i_boundary_cc * x_n * (x_n - 2 * l_n) / (2 * sigma_n_eff * l_n)

        return phi_s_n

    def get_explicit_combined(self, variables):
        """
        Provides an explicit combined leading and first order solution to solid phase
        current conservation with ohm's law. Note that the returned current density is
        only the leading order approximation.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model

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

        # Unpack variables
        phi_s_n = variables["Negative electrode potential"]
        phi_e = variables["Electrolyte potential"]
        ocp_p = variables["Positive electrode open circuit potential"]
        eta_r_p = variables["Positive reaction overpotential"]
        i_boundary_cc = variables["Current collector current density"]
        try:
            epsilon = variables["Porosity"]
        except KeyError:
            epsilon = param.epsilon
        eps_n, eps_s, eps_p = [e.orphans[0] for e in epsilon.orphans]
        _, _, phi_e_p = phi_e.orphans

        # obtain averages
        ocp_p_av = pybamm.average(ocp_p)
        eta_r_p_av = pybamm.average(eta_r_p)
        phi_e_p_av = pybamm.average(phi_e_p)

        # electrode potential
        sigma_n_eff = param.sigma_n * (1 - eps_n)
        sigma_p_eff = param.sigma_p * (1 - eps_p)

        const = (
            ocp_p_av
            + eta_r_p_av
            + phi_e_p_av
            - (i_boundary_cc / 6 / l_p / sigma_p_eff) * (2 * l_p ** 2 - 6 * l_p + 3)
        )

        phi_s_p = const - i_boundary_cc * x_p / (2 * l_p * sigma_p_eff) * (
            x_p + 2 * (l_p - 1)
        )

        # electrode current
        i_s_n = i_boundary_cc - i_boundary_cc * x_n / l_n
        i_s_p = i_boundary_cc - i_boundary_cc * (1 - x_p) / l_p

        delta_phi_s_av = -i_boundary_cc / 3 * (l_p / sigma_p_eff + l_n / sigma_n_eff)

        return self.get_variables(phi_s_n, phi_s_p, i_s_n, i_s_p, delta_phi_s_av)

    def get_variables(self, phi_s_n, phi_s_p, i_s_n, i_s_p, delta_phi_s_av=None):
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
