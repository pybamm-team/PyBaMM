#
# Equations for the electrode-electrolyte interface
#
import pybamm
import autograd.numpy as np


class InterfacialReaction(pybamm.SubModel):
    """
    Base class for interfacial currents

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def get_homogeneous_interfacial_current(self, i_boundary_cc, domain):
        """
        Homogeneous reaction at the electrode-electrolyte interface

        Parameters
        ----------
        i_boundary_cc : :class:`pybamm.Symbol`
            The current in the current collectors (can be 0D, 1D or 2D)
        domain : iter of str
            The domain(s) in which to compute the interfacial current.
        Returns
        -------
        :class:`pybamm.Symbol`
            Homogeneous interfacial current density
        """
        if domain == ["negative electrode"]:
            return i_boundary_cc / pybamm.geometric_parameters.l_n
        elif domain == ["positive electrode"]:
            return -i_boundary_cc / pybamm.geometric_parameters.l_p
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))

    def get_butler_volmer(self, j0, eta_r, domain=None):
        """
        Butler-Volmer reactions

        .. math::
            j = j_0(c) * \\sinh(\\eta_r(c))

        Parameters
        ----------
        j0 : :class:`pybamm.Symbol`
            Exchange-current density
        eta_r : :class:`pybamm.Symbol`
            Reaction overpotential
        domain : iter of str, optional
            The domain(s) in which to compute the interfacial current. Default is None,
            in which case j0.domain is used.

        Returns
        -------
        :class:`pybamm.Symbol`
            Interfacial current density

        """
        param = self.set_of_parameters

        domain = domain or j0.domain
        if domain == ["negative electrode"]:
            return 2 * j0 * pybamm.sinh((param.ne_n / 2) * eta_r)
        elif domain == ["positive electrode"]:
            return 2 * j0 * pybamm.sinh((param.ne_p / 2) * eta_r)
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))

    def get_butler_volmer_from_variables(self, c_e, delta_phi, domain=None):
        """
        Butler-Volmer reactions, using the variables directly

        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            Electrolyte concentration
        delta_phi : :class:`pybamm.Symbol`
            Surface potential difference
        domain : iter of str, optional
            The domain(s) in which to compute the interfacial current. Default is None,
            in which case c_e.domain is used.

        Returns
        -------
        :class:`pybamm.Symbol`
            Interfacial current density

        """
        domain = domain or c_e.domain
        if domain == ["negative electrode"]:
            ocp = self.set_of_parameters.U_n
        if domain == ["positive electrode"]:
            ocp = self.set_of_parameters.U_p

        j0 = self.get_exchange_current_densities(c_e, domain)
        return self.get_butler_volmer(j0, delta_phi - ocp(c_e), domain)

    def get_inverse_butler_volmer(self, j, j0, domain=None):
        """
        Inverts the Butler-Volmer relation to solve for the reaction overpotential.

        Parameters
        ----------
        j : :class:`pybamm.Symbol`
            Interfacial current density
        j0 : :class:`pybamm.Symbol`
            Exchange-current density
        domain : iter of str, optional
            The domain(s) in which to compute the interfacial current. Default is None,
            in which case j.domain is used.

        Returns
        -------
        :class:`pybamm.Symbol`
            Reaction overpotential

        """
        param = self.set_of_parameters

        # Broadcast if necessary (requires input domain)
        if domain and j.domain == ["current collector"]:
            j = pybamm.Broadcast(j, domain)
        else:
            domain = domain or j.domain

        if domain == ["negative electrode"]:
            return (2 / param.ne_n) * pybamm.Function(np.arcsinh, j / (2 * j0))
        elif domain == ["positive electrode"]:
            return (2 / param.ne_p) * pybamm.Function(np.arcsinh, j / (2 * j0))
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))

    def get_derived_interfacial_currents(self, j_n, j_p, j0_n, j0_p, reaction="main"):
        """
        Calculate dimensionless and dimensional variables for the interfacial current
        submodel

        Parameters
        ----------
        j_n : :class:`pybamm.Symbol`
            Interfacial current density in the negative electrode
        j_p : :class:`pybamm.Symbol`
            Interfacial current density in the positive electrode
        j0_n : :class:`pybamm.Symbol`
            Exchange-current density in the negative electrode
        j0_p : :class:`pybamm.Symbol`
            Exchange-current density in the positive electrode
        reaction : str, optional
            Name of the reaction to set interfacial currents for (default "main")

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        param = self.set_of_parameters
        j_n_scale = param.i_typ / (param.a_n_dim * param.L_x)
        j_p_scale = param.i_typ / (param.a_p_dim * param.L_x)

        # Broadcast if necessary
        if j_n.domain in [[], ["current collector"]]:
            j_n = pybamm.Broadcast(j_n, ["negative electrode"])
        if j_p.domain in [[], ["current collector"]]:
            j_p = pybamm.Broadcast(j_p, ["positive electrode"])
        if j0_n.domain in [[], ["current collector"]]:
            j0_n = pybamm.Broadcast(j0_n, ["negative electrode"])
        if j0_p.domain in [[], ["current collector"]]:
            j0_p = pybamm.Broadcast(j0_p, ["positive electrode"])

        # Concatenations
        j = pybamm.Concatenation(*[j_n, pybamm.Broadcast(0, ["separator"]), j_p])
        j0 = pybamm.Concatenation(*[j0_n, pybamm.Broadcast(0, ["separator"]), j0_p])
        j_dimensional = pybamm.Concatenation(
            *[j_n_scale * j_n, pybamm.Broadcast(0, ["separator"]), j_p_scale * j_p]
        )
        j0_dimensional = pybamm.Concatenation(
            *[j_n_scale * j0_n, pybamm.Broadcast(0, ["separator"]), j_p_scale * j0_p]
        )

        if reaction == "main":
            name = "interfacial current density"
            ecd_name = "exchange-current density"
        elif reaction == "oxygen":
            name = "oxygen interfacial current density"
            ecd_name = "oxygen exchange-current density"

        variables = {
            name.capitalize(): j,
            ecd_name.capitalize(): j0,
            name.capitalize() + " [A.m-2]": j_dimensional,
            ecd_name.capitalize() + " [A.m-2]": j0_dimensional,
        }
        for domain, j, j0, j_scale in [
            ["negative electrode", j_n, j0_n, j_n_scale],
            ["positive electrode", j_p, j0_p, j_p_scale],
        ]:
            j_bar = pybamm.average(j)
            domain_variables = {
                domain.capitalize() + " " + name: j,
                "Average " + domain + " " + name: j_bar,
                domain.capitalize() + " " + ecd_name: j0,
                domain.capitalize() + " " + name + " [A.m-2]": j_scale * j,
                "Average " + domain + " " + name + " [A.m-2]": j_scale * j_bar,
                domain.capitalize() + " " + ecd_name + " [A.m-2]": j_scale * j0,
            }
            variables.update(domain_variables)
        return variables

    def get_first_order_butler_volmer(
        self, c_e, delta_phi, c_e_0, delta_phi_0, domain=None
    ):
        """
        First-order correction for the Butler-Volmer reactions

        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            Electrolyte concentration
        delta_phi : :class:`pybamm.Symbol`
            Surface potential difference
        c_e_0 : :class:`pybamm.Symbol`
            Leading-order electrolyte concentration
        delta_phi_0 : :class:`pybamm.Symbol`
            Leading-order surface potential difference

        Returns
        -------
        :class:`pybamm.Symbol`
            Interfacial current density

        """
        param = self.set_of_parameters
        # Take 1 * c_e_0 as a hack for differentiation
        c_e_0 *= 1
        domain = domain or c_e.domain
        j_0 = self.get_butler_volmer_from_variables(c_e_0, delta_phi_0, domain)
        c_e_1 = (c_e - c_e_0) / param.C_e
        delta_phi_1 = (delta_phi - delta_phi_0) / param.C_e

        j_1 = j_0.diff(c_e_0) * c_e_1 + j_0.diff(delta_phi_0) * delta_phi_1
        return j_0 + param.C_e * j_1

    def get_first_order_potential_differences(self, variables, leading_order_vars):
        """
        Calculates surface potential difference using the linear first-order correction
        to the Butler-Volmer, and then calculates derived potentials.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        param = self.set_of_parameters
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        delta_phi_n_0 = pybamm.average(
            leading_order_vars["Negative electrode surface potential difference"]
        )
        delta_phi_p_0 = pybamm.average(
            leading_order_vars["Positive electrode surface potential difference"]
        )

        # Take 1 * c_e_0 so that it doesn't appear in delta_phi_n_0 and delta_phi_p_0
        c_e_0 = 1 * leading_order_vars["Average electrolyte concentration"]

        j_n_0 = self.get_butler_volmer_from_variables(c_e_0, delta_phi_n_0, neg)
        j_p_0 = self.get_butler_volmer_from_variables(c_e_0, delta_phi_p_0, pos)

        c_e = variables["Electrolyte concentration"]
        c_e_n, c_e_s, c_e_p = c_e.orphans
        c_e_n_1_bar = (pybamm.average(c_e_n) - c_e_0) / param.C_e
        c_e_p_1_bar = (pybamm.average(c_e_p) - c_e_0) / param.C_e
        delta_phi_n_1_bar = -j_n_0.diff(c_e_0) * c_e_n_1_bar / j_n_0.diff(delta_phi_n_0)
        delta_phi_p_1_bar = -j_p_0.diff(c_e_0) * c_e_p_1_bar / j_p_0.diff(delta_phi_p_0)

        delta_phi_n = delta_phi_n_0 + param.C_e * delta_phi_n_1_bar
        delta_phi_p = delta_phi_p_0 + param.C_e * delta_phi_p_1_bar
        ocp_n = param.U_n(c_e_n)
        ocp_p = param.U_p(c_e_p)

        pot_model = pybamm.potential.Potential(param)
        return pot_model.get_all_potentials(
            (ocp_n, ocp_p), delta_phi=(delta_phi_n, delta_phi_p)
        )

    def get_average_potential_differences(self, variables):
        """
        Calculates surface potential difference using the average Butler-Volmer, and
        then calculates derived potentials.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        # Set up
        param = self.set_of_parameters
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        # Unpack and average variables
        i_bnd_cc = variables["Current collector current density"]
        c_e = variables["Electrolyte concentration"]
        c_e_n, _, c_e_p = c_e.orphans
        c_e_n_bar = pybamm.average(c_e_n)
        c_e_p_bar = pybamm.average(c_e_p)

        # Calculate reaction overpotentials
        j0_n_bar = self.get_exchange_current_densities(c_e_n_bar, neg)
        j0_p_bar = self.get_exchange_current_densities(c_e_p_bar, pos)
        j_n_bar = self.get_homogeneous_interfacial_current(i_bnd_cc, neg)
        j_p_bar = self.get_homogeneous_interfacial_current(i_bnd_cc, pos)
        eta_r_n_bar = self.get_inverse_butler_volmer(j_n_bar, j0_n_bar, neg)
        eta_r_p_bar = self.get_inverse_butler_volmer(j_p_bar, j0_p_bar, pos)

        # Set derived potential
        ocp_n_bar = param.U_n(c_e_n_bar)
        ocp_p_bar = param.U_p(c_e_p_bar)
        pot_model = pybamm.potential.Potential(param)
        return pot_model.get_all_potentials(
            (ocp_n_bar, ocp_p_bar), eta_r=(eta_r_n_bar, eta_r_p_bar)
        )

    def get_current_from_current_densities(self, variables):
        param = self.set_of_parameters
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p
        j_n = variables["Negative electrode interfacial current density"]
        j_p = variables["Positive electrode interfacial current density"]
        i_boundary_cc = variables["Current collector current density"]

        # Electrolyte current
        i_e_n = pybamm.IndefiniteIntegral(j_n, x_n)
        # Shift i_e_p to be equal to 0 at x_p = 1
        i_e_p = pybamm.IndefiniteIntegral(j_p, x_p) - pybamm.Integral(j_p, x_p)
        eleclyte_model = pybamm.electrolyte_current.ElectrolyteCurrentBaseModel(param)
        eleclyte_variables = eleclyte_model.get_current_variables(
            (i_e_n, i_e_p), i_boundary_cc
        )

        # Electrode current
        i_s_n = i_boundary_cc - i_e_n
        i_s_p = i_boundary_cc - i_e_p
        electrode_model = pybamm.electrode.Ohm(param)
        electrode_variables = electrode_model.get_current_variables(i_s_n, i_s_p)
        return {**eleclyte_variables, **electrode_variables}


class LithiumIonReaction(InterfacialReaction):
    """
    Interfacial current from lithium-ion reactions

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`InterfacialReaction`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def get_exchange_current_densities(self, c_e, c_s_k_surf, domain=None):
        """The exchange current-density as a function of concentration

        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            Electrolyte concentration
        c_s_k_surf : :class:`pybamm.Symbol`
            Electrode surface concentration
        domain : iter of str, optional
            The domain(s) in which to compute the interfacial current. Default is None,
            in which case c_e.domain is used.

        Returns
        -------
        :class:`pybamm.Symbol`
            Exchange-current density
        """
        param = self.set_of_parameters
        domain = domain or c_e.domain

        if domain == ["negative electrode"]:
            return (1 / param.C_r_n) * (
                c_e ** (1 / 2) * c_s_k_surf ** (1 / 2) * (1 - c_s_k_surf) ** (1 / 2)
            )
        elif domain == ["positive electrode"]:
            return (param.gamma_p / param.C_r_p) * (
                c_e ** (1 / 2) * c_s_k_surf ** (1 / 2) * (1 - c_s_k_surf) ** (1 / 2)
            )
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))
