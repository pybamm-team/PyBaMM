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

        phi_e = pybamm.concatenation(*phi_e_dict.values())

        # Case where negative electrode is not included (half-cell)
        if "negative electrode" not in self.options.whole_cell_domains:
            phi_e_s = phi_e_dict["separator"]
            phi_e_dict["negative electrode"] = pybamm.boundary_value(phi_e_s, "left")

        eta_e_av = pybamm.x_average(
            phi_e_dict["positive electrode"]
        ) - pybamm.x_average(phi_e_dict["negative electrode"])
        phi_e_av = pybamm.x_average(phi_e)

        variables = {
            "Electrolyte potential [V]": phi_e,
            "X-averaged electrolyte potential [V]": phi_e_av,
            "X-averaged electrolyte overpotential [V]": eta_e_av,
            "Gradient of electrolyte potential [V.m-1]": pybamm.grad(phi_e),
        }

        for domain, phi_e_k in phi_e_dict.items():
            name = f"{domain.split()[0]} electrolyte potential"
            Name = name.capitalize()
            phi_e_k_av = pybamm.x_average(phi_e_k)
            variables.update(
                {f"{Name} [V]": phi_e_k, f"X-averaged {name} [V]": phi_e_k_av}
            )
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
            if self.options.whole_cell_domains == [
                "negative electrode",
                "separator",
                "positive electrode",
            ]:
                i_e_n, _, i_e_p = i_e.orphans
            elif self.options.whole_cell_domains == ["separator", "positive electrode"]:
                _, i_e_p = i_e.orphans
                i_e_n = None

            if i_e_n is not None:
                variables.update(
                    {"Negative electrolyte current density [A.m-2]": i_e_n}
                )
            if i_e_p is not None:
                variables.update(
                    {"Positive electrolyte current density [A.m-2]": i_e_p}
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

    def _get_electrolyte_overpotentials(self, variables):
        """
        A private function to obtain the electrolyte overpotential and Ohmic losses.
        Note: requires 'variables' to contain the potential, electrolyte concentration
        and temperature the subdomains: 'negative electrode', 'separator', and
        'positive electrode'.

        Parameters
        ----------
        variables : dict
            The variables that have been set in the rest of the model.

        Returns
        -------
        variables : dict
            The variables including the whole-cell electrolyte potentials
            and currents.
        """
        param = self.param

        if self.options.electrode_types["negative"] == "planar":
            # No concentration overpotential in the counter electrode
            phi_e_n = pybamm.Scalar(0)
            indef_integral_n = pybamm.Scalar(0)
        else:
            phi_e_n = variables["Negative electrolyte potential [V]"]
            # concentration overpotential
            c_e_n = variables["Negative electrolyte concentration [mol.m-3]"]
            T_n = variables["Negative electrode temperature [K]"]
            indef_integral_n = pybamm.IndefiniteIntegral(
                param.chiRT_over_Fc(c_e_n, T_n) * pybamm.grad(c_e_n),
                pybamm.standard_spatial_vars.x_n,
            )

        phi_e_p = variables["Positive electrolyte potential [V]"]

        c_e_s = variables["Separator electrolyte concentration [mol.m-3]"]
        c_e_p = variables["Positive electrolyte concentration [mol.m-3]"]

        T_s = variables["Separator temperature [K]"]
        T_p = variables["Positive electrode temperature [K]"]

        # concentration overpotential
        indef_integral_s = pybamm.IndefiniteIntegral(
            param.chiRT_over_Fc(c_e_s, T_s) * pybamm.grad(c_e_s),
            pybamm.standard_spatial_vars.x_s,
        )
        indef_integral_p = pybamm.IndefiniteIntegral(
            param.chiRT_over_Fc(c_e_p, T_p) * pybamm.grad(c_e_p),
            pybamm.standard_spatial_vars.x_p,
        )

        integral_n = indef_integral_n
        integral_s = indef_integral_s + pybamm.boundary_value(integral_n, "right")
        integral_p = indef_integral_p + pybamm.boundary_value(integral_s, "right")

        eta_c_av = pybamm.x_average(integral_p) - pybamm.x_average(integral_n)

        delta_phi_e_av = (
            pybamm.x_average(phi_e_p) - pybamm.x_average(phi_e_n) - eta_c_av
        )

        variables.update(self._get_split_overpotential(eta_c_av, delta_phi_e_av))

        return variables

    def set_boundary_conditions(self, variables):
        phi_e = variables["Electrolyte potential [V]"]

        if self.options.electrode_types["negative"] == "planar":
            phi_e_ref = variables["Lithium metal interface electrolyte potential [V]"]
            lbc = (phi_e_ref, "Dirichlet")
        else:
            lbc = (pybamm.Scalar(0), "Neumann")
        self.boundary_conditions = {
            phi_e: {"left": lbc, "right": (pybamm.Scalar(0), "Neumann")}
        }
