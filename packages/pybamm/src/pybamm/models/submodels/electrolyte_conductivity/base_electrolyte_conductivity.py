#
# Base class for electrolyte conductivity
#

import pybamm


class BaseElectrolyteConductivity(pybamm.BaseSubModel):
    """Base class for conservation of charge in the electrolyte.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str, optional
        The domain in which the model holds
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, domain=None, options=None):
        super().__init__(param, domain, options=options)

    def _coupled_domain_variable(self, name, domain):
        """Return a placeholder for a variable on ``domain`` and the current collector."""
        return self.coupled_variable(
            name, domain=domain, auxiliary_domains={"secondary": "current collector"}
        )

    def _coupled_lithium_metal_electrolyte_potential(self):
        """Return a placeholder for the electrolyte potential at a lithium metal
        counter electrode (half-cell)."""
        return self.coupled_variable(
            "Lithium metal interface electrolyte potential [V]",
            domain="current collector",
        )

    def _get_standard_potential_variables(self, phi_e_dict):
        """
        A private function to obtain the standard variables which
        can be derived from the potential in the electrolyte.

        Parameters
        ----------
        phi_e_dict : dict of :class:`pybamm.Symbol`
            Dictionary of electrolyte potentials in the relevant domains

        Returns
        -------
        variables : dict
            The variables which can be derived from the potential in the
            electrolyte.
        """
        variables = self._get_standard_whole_cell_potential_variables(phi_e_dict)
        for domain, phi_e_k in phi_e_dict.items():
            variables.update(
                self._get_standard_domain_potential_variables(domain, phi_e_k)
            )
        if "negative electrode" not in self.options.whole_cell_domains:
            variables.update(
                self._get_standard_domain_potential_variables(
                    "negative electrode",
                    self._get_half_cell_negative_potential(phi_e_dict),
                )
            )
        return variables

    def _get_half_cell_negative_potential(self, phi_e_dict):
        """Return the electrolyte potential at the negative side of a half-cell."""
        return pybamm.boundary_value(phi_e_dict["separator"], "left")

    def _get_standard_whole_cell_potential_variables(self, phi_e_dict):
        """
        A private function to obtain the standard variables of the electrolyte
        potential over the whole cell.

        Parameters
        ----------
        phi_e_dict : dict of :class:`pybamm.Symbol`
            Dictionary of electrolyte potentials in the relevant domains

        Returns
        -------
        variables : dict
            The whole-cell electrolyte potential variables.
        """
        phi_e = pybamm.concatenation(*phi_e_dict.values())
        if "negative electrode" in phi_e_dict:
            phi_e_n = phi_e_dict["negative electrode"]
        else:
            phi_e_n = self._get_half_cell_negative_potential(phi_e_dict)
        eta_e_av = pybamm.x_average(
            phi_e_dict["positive electrode"]
        ) - pybamm.x_average(phi_e_n)

        return {
            "Electrolyte potential [V]": phi_e,
            "X-averaged electrolyte potential [V]": pybamm.x_average(phi_e),
            "X-averaged electrolyte overpotential [V]": eta_e_av,
            "Gradient of electrolyte potential [V.m-1]": pybamm.grad(phi_e),
        }

    def _get_standard_domain_potential_variables(self, domain, phi_e_k):
        """
        A private function to obtain the standard variables of the electrolyte
        potential in one domain.

        Parameters
        ----------
        domain : str
            The domain, e.g. "negative electrode" or "separator"
        phi_e_k : :class:`pybamm.Symbol`
            The electrolyte potential in the domain

        Returns
        -------
        variables : dict
            The electrolyte potential variables of the domain.
        """
        name = f"{domain.split()[0]} electrolyte potential"
        Name = name.capitalize()
        variables = {
            f"{Name} [V]": phi_e_k,
            f"X-averaged {name} [V]": pybamm.x_average(phi_e_k),
        }
        if domain in self.options.whole_cell_domains:
            variables[f"Gradient of {name} [V.m-1]"] = pybamm.grad(phi_e_k)
        return variables

    def _get_standard_current_variables(self, i_e):
        """
        A private function to obtain the standard variables which
        can be derived from the current in the electrolyte.

        Parameters
        ----------
        i_e : :class:`pybamm.Symbol`
            The current in the electrolyte.

        Returns
        -------
        variables : dict
            The variables which can be derived from the current in the
            electrolyte.
        """

        variables = {"Electrolyte current density [A.m-2]": i_e}

        if isinstance(i_e, pybamm.Concatenation):
            i_e_dict = dict(zip(self.options.whole_cell_domains, i_e.orphans))
            for domain in ["negative electrode", "positive electrode"]:
                if domain in i_e_dict:
                    Domain = domain.split()[0].capitalize()
                    variables[f"{Domain} electrolyte current density [A.m-2]"] = (
                        i_e_dict[domain]
                    )

        return variables

    def _get_split_overpotential(self, eta_c_av, delta_phi_e_av):
        """
        A private function to obtain the standard variables which
        can be derived from the electrode-averaged concentration
        overpotential and Ohmic losses in the electrolyte.

        Parameters
        ----------
        eta_c_av : :class:`pybamm.Symbol`
            The electrode-averaged concentration overpotential
        delta_phi_e_av: :class:`pybamm.Symbol`
            The electrode-averaged electrolyte Ohmic losses

        Returns
        -------
        variables : dict
            The variables which can be derived from the electrode-averaged
            concentration overpotential and Ohmic losses in the electrolyte
            electrolyte.
        """
        variables = {
            "X-averaged concentration overpotential [V]": eta_c_av,
            "X-averaged electrolyte ohmic losses [V]": delta_phi_e_av,
        }

        return variables

    def _get_standard_average_surface_potential_difference_variables(
        self, delta_phi_av
    ):
        """
        A private function to obtain the standard variables which
        can be derived from the surface potential difference.

        Parameters
        ----------
        delta_phi_av : :class:`pybamm.Symbol`
            The x-averaged surface potential difference.

        Returns
        -------
        variables : dict
            The variables which can be derived from the surface potential difference.
        """
        domain = self.domain
        variables = {
            f"X-averaged {domain} electrode "
            "surface potential difference [V]": delta_phi_av,
        }
        return variables

    def _get_standard_surface_potential_difference_variables(self, delta_phi):
        """
        A private function to obtain the standard variables which
        can be derived from the surface potential difference.

        Parameters
        ----------
        delta_phi : :class:`pybamm.Symbol`
            The surface potential difference.

        Returns
        -------
        variables : dict
            The variables which can be derived from the surface potential difference.
        """
        domain, Domain = self.domain_Domain

        # Broadcast if necessary
        if delta_phi.domain == ["current collector"]:
            delta_phi = pybamm.PrimaryBroadcast(delta_phi, f"{domain} electrode")

        variables = {
            f"{Domain} electrode surface potential difference [V]": delta_phi,
        }

        if Domain == "Negative":
            variables[
                "Negative electrode surface potential difference "
                "at separator interface [V]"
            ] = pybamm.boundary_value(delta_phi, "right")
        elif Domain == "Positive":
            variables[
                "Positive electrode surface potential difference "
                "at separator interface [V]"
            ] = pybamm.boundary_value(delta_phi, "left")

        return variables

    def _get_electrolyte_overpotentials(self, phi_e_dict):
        """
        A private function to obtain the electrolyte overpotential and Ohmic losses.

        Parameters
        ----------
        phi_e_dict : dict of :class:`pybamm.Symbol`
            Dictionary of electrolyte potentials in the domains of the cell

        Returns
        -------
        variables : dict
            The x-averaged concentration overpotential and electrolyte Ohmic losses.
        """
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        def concentration_integral(domain, x):
            Domain = domain.capitalize()
            c_e_k = self._coupled_domain_variable(
                f"{Domain.split()[0]} electrolyte concentration [mol.m-3]", domain
            )
            T_k = self._coupled_domain_variable(f"{Domain} temperature [K]", domain)
            return pybamm.IndefiniteIntegral(
                self.param.chiRT_over_Fc(c_e_k, T_k) * pybamm.grad(c_e_k), x
            )

        if self.options.electrode_types["negative"] == "planar":
            # No concentration overpotential in the counter electrode
            phi_e_n = self._coupled_lithium_metal_electrolyte_potential()
            indef_integral_n = pybamm.Scalar(0)
        else:
            phi_e_n = phi_e_dict["negative electrode"]
            indef_integral_n = concentration_integral("negative electrode", x_n)
        indef_integral_s = concentration_integral("separator", x_s)
        indef_integral_p = concentration_integral("positive electrode", x_p)
        phi_e_p = phi_e_dict["positive electrode"]

        integral_n = indef_integral_n
        integral_s = indef_integral_s + pybamm.boundary_value(integral_n, "right")
        integral_p = indef_integral_p + pybamm.boundary_value(integral_s, "right")

        eta_c_av = pybamm.x_average(integral_p) - pybamm.x_average(integral_n)

        delta_phi_e_av = (
            pybamm.x_average(phi_e_p) - pybamm.x_average(phi_e_n) - eta_c_av
        )

        return self._get_split_overpotential(eta_c_av, delta_phi_e_av)

    def _get_electrolyte_potential_boundary_conditions(self, phi_e):
        """
        Return the boundary conditions on the electrolyte potential over the whole
        cell: no flux at the current collectors, or the lithium metal interface
        potential at a planar negative electrode.

        Parameters
        ----------
        phi_e : :class:`pybamm.Symbol`
            The electrolyte potential over the whole cell

        Returns
        -------
        dict
            The boundary conditions, keyed by ``phi_e``.
        """
        if self.options.electrode_types["negative"] == "planar":
            phi_e_ref = self._coupled_lithium_metal_electrolyte_potential()
            lbc = (phi_e_ref, "Dirichlet")
        else:
            lbc = (pybamm.Scalar(0), "Neumann")
        return {phi_e: {"left": lbc, "right": (pybamm.Scalar(0), "Neumann")}}

    def _get_x_averaged_current_imbalance(self):
        """
        Return the x-averaged surface potential difference of the submodel's
        electrode and the imbalance of x-averaged interfacial currents that drives it.

        Returns
        -------
        delta_phi_av : :class:`pybamm.Variable`
            The x-averaged surface potential difference
        current_imbalance : :class:`pybamm.Symbol`
            The x-averaged total volumetric interfacial current density minus the
            sum of the x-averaged volumetric interfacial current densities
        """
        domain = self.domain
        delta_phi_av = self.variables[
            f"X-averaged {domain} electrode surface potential difference [V]"
        ]
        sum_a_j_av = self.coupled_variable(
            f"X-averaged {domain} electrode total volumetric "
            "interfacial current density [A.m-3]",
            domain="current collector",
        )
        sum_a_j = self.coupled_variable(
            f"Sum of x-averaged {domain} electrode volumetric "
            "interfacial current densities [A.m-3]",
            domain="current collector",
        )
        return delta_phi_av, sum_a_j_av - sum_a_j

    def _get_x_averaged_double_layer_rhs(self, current_imbalance):
        """
        Return the rate of change of the x-averaged surface potential difference
        due to double-layer capacitance.

        Parameters
        ----------
        current_imbalance : :class:`pybamm.Symbol`
            The imbalance of x-averaged interfacial currents

        Returns
        -------
        :class:`pybamm.Symbol`
        """
        domain = self.domain
        T_av = self.coupled_variable(
            f"X-averaged {domain} electrode temperature [K]",
            domain="current collector",
        )
        a_av = self.coupled_variable(
            f"X-averaged {domain} electrode surface area to volume ratio [m-1]",
            domain="current collector",
        )
        C_dl = self.domain_param.C_dl(T_av)
        return 1 / (a_av * C_dl) * current_imbalance
