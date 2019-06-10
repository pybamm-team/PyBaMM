#
# Full model for Ohm's law in the electrode
#
import pybamm


class FullOhm(pybamm.BaseSubModel):
    """Full model for ohm's law with conservation of current for the current in the 
    electrodes.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        """
        Returns the variables in the submodel for which a PDE must be solved to obtains
        """

        if self._domain == "Negative electrode":
            phi_s = pybamm.standard_variables.phi_s_n
        elif self._domain == "Positive electrode":
            phi_s = pybamm.standard_variables.phi_s_p
        else:
            pybamm.DomainError(
                "Domain must be either: 'Negative electrode' or 'Positive electode'"
            )

        fundamental_variables = {self._domain + " potential": phi_s}

        return fundamental_variables

    def get_derived_variables(self, variables, reactions):
        """
        Returns variables which are derived from the fundamental variables in the model.
        """

        phi_s = variables[self._domain + " potential"]
        eps = variables[self._domain + " porosity"]

        if self._domain == "Negative electrode":
            sigma = self.param.sigma_n
        elif self._domain == "Positive electrode":
            sigma = self.param.sigma_p

        sigma_eff = sigma * (1 - eps) ** self.param.b
        i_s = -sigma_eff * pybamm.grad(phi_s)

        derived_variables = {
            self._domain + " current density": i_s,
            self._domain + " effective conductivity": sigma_eff,
        }

        return derived_variables

    def set_equations(self, variables):
        """
        Sets the governing equations, boundary conditions, and initial conditions
        in the model.
        """
        self.set_rhs(variables)
        self.set_algebraic(variables)
        self.set_boundary_conditions(variables)
        self.set_initial_conditions(variables)

    def set_rhs(self, variables):
        self.rhs = {}

    def set_algebraic(self, variables):
        """
        PDE for current in the electrodes, using Ohm's law

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        """
        phi_s = variables[self._domain + " potential"]
        i_s = variables[self._domain + " current density"]
        j = variables[self._domain + " interfacial current density"]

        self.algebraic[phi_s] = pybamm.div(i_s) + j

    def set_boundary_conditions(self, variables):
        """
        Boundary conditions for current in the electrodes.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        """
        phi_s = variables[self._domain + " potential"]
        sigma_eff = variables[self._domain + " effective conductivity"]
        i_boundary_cc = variables["Current collector current density"]

        if self._domain == ["Negative electrode"]:
            lbc = (pybamm.Scalar(0), "Dirichlet")
            rbc = (pybamm.Scalar(0), "Neumann")

        elif self._domain == ["Positive electrode"]:
            lbc = (pybamm.Scalar(0), "Neumann")
            rbc = (
                i_boundary_cc / pybamm.boundary_value(-sigma_eff, "right"),
                "Neumann",
            )

        self.boundary_conditions[phi_s] = {"left": lbc, "right": rbc}

    def set_initial_conditions(self, variables):
        """
        Initial conditions for current and potentials in the electrodes.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        """
        phi_s = variables[self._domain + " potential"]

        if self._domain == "Negative electrode":
            phi_s_init = pybamm.Scalar(0)
        elif self._domain == "Positive electrode":
            phi_s_init = self.param.U_p(self.param.c_p_init) - self.param.U_n(
                self.param.c_n_init
            )

        self.initial_conditions[phi_s] = phi_s_init

    def set_algebraic_system(self, variables, reactions, domain):
        
        # unpack variables
        i_boundary_cc, phi_s, eps = self.unpack(variables, domain)

        param = self.set_of_parameters

        # different bounday conditions in each electrode
        if domain == ["negative electrode"]:
            j = reactions["main"]["neg"]["aj"]
            # liion sigma_n may already account for porosity
            conductivity = param.sigma_n * (1 - eps) ** param.b
            lbc = (pybamm.Scalar(0), "Dirichlet")
            rbc = (pybamm.Scalar(0), "Neumann")
            self.initial_conditions[phi_s] = pybamm.Scalar(0)
        elif domain == ["positive electrode"]:
            j = reactions["main"]["pos"]["aj"]
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
        self.variables[domain[0].capitalize() + " current density"] = i_s

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
        i_boundary_cc, ocp_p_av, eta_r_p_av, phi_e_p_av = self.unpack_post(variables)

        # import parameters and spatial variables
        param = self.set_of_parameters
        l_n = param.l_n
        l_p = param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        # electode potential
        phi_s_n = pybamm.Broadcast(0, ["negative electrode"])
        v = ocp_p_av + eta_r_p_av + phi_e_p_av
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
        i_boundary_cc, _, eps_n = self.unpack(variables, ["negative electrode"])

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
        i_boundary_cc, (phi_s_n, _), epsilon = self.unpack(variables)
        i_boundary_cc, ocp_p_av, eta_r_p_av, phi_e_p_av = self.unpack_post(variables)
        eps_n, eps_s, eps_p = [e.orphans[0] for e in epsilon.orphans]

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

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        return pybamm.ScikitsDaeSolver()
