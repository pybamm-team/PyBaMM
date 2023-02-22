#
# Base class for electrolyte diffusion
#
import pybamm


class BaseElectrolyteDiffusion(pybamm.BaseSubModel):
    """Base class for conservation of mass in the electrolyte.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def _get_standard_concentration_variables(self, c_e_dict):
        """
        A private function to obtain the standard variables which
        can be derived from the concentration in the electrolyte.

        Parameters
        ----------
        c_e_dict : dict of :class:`pybamm.Symbol`
            Electrolyte concentrations in the various domains

        Returns
        -------
        variables : dict
            The variables which can be derived from the concentration in the
            electrolyte.
        """

        c_e = pybamm.concatenation(*c_e_dict.values())
        # Override print_name
        c_e.print_name = "c_e"

        variables = self._get_standard_domain_concentration_variables(c_e_dict)
        variables.update(self._get_standard_whole_cell_concentration_variables(c_e))
        return variables

    def _get_standard_domain_concentration_variables(self, c_e_dict):
        variables = {}
        # Case where an electrode is not included (half-cell)
        if "negative electrode" not in self.options.whole_cell_domains:
            c_e_s = c_e_dict["separator"]
            c_e_dict["negative electrode"] = pybamm.boundary_value(c_e_s, "left")

        for domain, c_e_k in c_e_dict.items():
            domain = domain.split()[0]
            Domain = domain.capitalize()
            c_e_k_av = pybamm.x_average(c_e_k)
            variables.update(
                {
                    f"{Domain} electrolyte concentration [mol.m-3]": c_e_k,
                    f"X-averaged {domain} electrolyte "
                    "concentration [mol.m-3]": c_e_k_av,
                }
            )

        # Calculate dimensionless and molar variables
        variables_dim = variables.copy()
        for name, var in variables_dim.items():
            name = name.replace("[mol.m-3]", "[Molar]")
            variables[name] = var / 1000

        return variables

    def _get_standard_whole_cell_concentration_variables(self, c_e):
        variables = {
            "Electrolyte concentration [mol.m-3]": c_e,
            "X-averaged electrolyte concentration [mol.m-3]": pybamm.x_average(c_e),
        }
        variables_nondim = variables.copy()
        for name, var in variables_nondim.items():
            name = name.replace("[mol.m-3]", "[Molar]")
            variables[name] = var / 1000

        return variables

    def _get_standard_porosity_times_concentration_variables(self, eps_c_e_dict):
        eps_c_e = pybamm.concatenation(*eps_c_e_dict.values())
        variables = {"Porosity times concentration [mol.m-3]": eps_c_e}

        for domain, eps_c_e_k in eps_c_e_dict.items():
            Domain = domain.capitalize()
            variables[f"{Domain} porosity times concentration [mol.m-3]"] = eps_c_e_k

        # Total lithium concentration in electrolyte
        variables.update(self._get_total_concentration_electrolyte(eps_c_e))

        return variables

    def _get_standard_flux_variables(self, N_e):
        """
        A private function to obtain the standard variables which
        can be derived from the mass flux in the electrolyte.

        Parameters
        ----------
        N_e : :class:`pybamm.Symbol`
            The flux in the electrolyte.

        Returns
        -------
        variables : dict
            The variables which can be derived from the flux in the
            electrolyte.
        """
        variables = {"Electrolyte flux [mol.m-2.s-1]": N_e}
        return variables

    def _get_total_concentration_electrolyte(self, eps_c_e):
        """
        A private function to obtain the total ion concentration in the electrolyte.
        Parameters
        ----------
        eps_c_e : :class:`pybamm.Symbol`
            Porosity times electrolyte concentration
        Returns
        -------
        variables : dict
            The "Total lithium in electrolyte [mol]" variable.
        """
        L_x = self.param.L_x
        A = self.param.A_cc

        eps_c_e_av = pybamm.yz_average(pybamm.x_average(eps_c_e))

        variables = {"Total lithium in electrolyte [mol]": L_x * A * eps_c_e_av}

        return variables
