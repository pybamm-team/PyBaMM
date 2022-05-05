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

    def _get_standard_concentration_variables(self, c_e_list):
        """
        A private function to obtain the standard variables which
        can be derived from the concentration in the electrolyte.

        Parameters
        ----------
        c_e_list : list of :class:`pybamm.Symbol`
            Electrolyte concentrations in the various domains

        Returns
        -------
        variables : dict
            The variables which can be derived from the concentration in the
            electrolyte.
        """

        c_e_typ = self.param.c_e_typ
        c_e = pybamm.concatenation(*c_e_list)
        # Override print_name
        c_e.print_name = "c_e"

        c_e_av = pybamm.x_average(c_e)

        variables = {
            "Electrolyte concentration": c_e,
            "Electrolyte concentration [mol.m-3]": c_e_typ * c_e,
            "Electrolyte concentration [Molar]": c_e_typ * c_e / 1000,
            "X-averaged electrolyte concentration": c_e_av,
            "X-averaged electrolyte concentration [mol.m-3]": c_e_typ * c_e_av,
            "X-averaged electrolyte concentration [Molar]": c_e_typ * c_e_av / 1000,
        }
        # if self.half_cell:
        #     # overwrite c_e_n to be the boundary value of c_e_s
        #     c_e_n = pybamm.boundary_value(c_e_s, "left")

        for dom, c_e_dom in zip(self.domains, c_e_list):
            name = dom.split(" ")[0] + " electrolyte concentration"
            Name = name.capitalize()
            c_e_dom_av = pybamm.x_average(c_e_dom)
            variables.update(
                {
                    f"{Name}": c_e_dom,
                    f"{Name} [mol.m-3]": c_e_typ * c_e_dom,
                    f"{Name} [Molar]": c_e_typ * c_e_dom / 1000,
                    f"X-averaged {name}": c_e_dom_av,
                    f"X-averaged {name} [mol.m-3]": c_e_typ * c_e_dom_av,
                }
            )

        return variables

    def _get_standard_porosity_times_concentration_variables(self, eps_c_e_list):
        eps_c_e = pybamm.concatenation(*eps_c_e_list)
        variables = {"Porosity times concentration": eps_c_e}

        for dom, eps_c_e_dom in zip(self.domains, eps_c_e_list):
            variables[f"{dom} porosity times concentration"] = eps_c_e_dom

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

        param = self.param
        flux_scale = param.D_e_typ * param.c_e_typ / param.L_x

        variables = {
            "Electrolyte flux": N_e,
            "Electrolyte flux [mol.m-2.s-1]": N_e * flux_scale,
        }

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

        c_e_typ = self.param.c_e_typ
        L_x = self.param.L_x
        A = self.param.A_cc

        eps_c_e_av = pybamm.yz_average(pybamm.x_average(eps_c_e))

        variables = {
            "Total lithium in electrolyte": eps_c_e_av,
            "Total lithium in electrolyte [mol]": c_e_typ * L_x * A * eps_c_e_av,
        }

        return variables

    def set_events(self, variables):
        c_e = variables["Electrolyte concentration"]
        self.events.append(
            pybamm.Event(
                "Zero electrolyte concentration cut-off",
                pybamm.min(c_e) - 0.002,
                pybamm.EventType.TERMINATION,
            )
        )
