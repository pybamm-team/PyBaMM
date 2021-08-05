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

    def _get_standard_concentration_variables(self, c_e_n, c_e_s, c_e_p):
        """
        A private function to obtain the standard variables which
        can be derived from the concentration in the electrolyte.

        Parameters
        ----------
        c_e_n : :class:`pybamm.Symbol`
            The electrolyte concentration in the negative electrode.
        c_e_s : :class:`pybamm.Symbol`
            The electrolyte concentration in the separator.
        c_e_p : :class:`pybamm.Symbol`
            The electrolyte concentration in the positive electrode.

        Returns
        -------
        variables : dict
            The variables which can be derived from the concentration in the
            electrolyte.
        """

        c_e_typ = self.param.c_e_typ
        c_e = pybamm.concatenation(c_e_n, c_e_s, c_e_p)

        if self.half_cell:
            # overwrite c_e_n to be the boundary value of c_e_s
            c_e_n = pybamm.boundary_value(c_e_s, "left")

        c_e_n_av = pybamm.x_average(c_e_n)
        c_e_av = pybamm.x_average(c_e)
        c_e_s_av = pybamm.x_average(c_e_s)
        c_e_p_av = pybamm.x_average(c_e_p)

        variables = {
            "Electrolyte concentration": c_e,
            "Electrolyte concentration [mol.m-3]": c_e_typ * c_e,
            "Electrolyte concentration [Molar]": c_e_typ * c_e / 1000,
            "X-averaged electrolyte concentration": c_e_av,
            "X-averaged electrolyte concentration [mol.m-3]": c_e_typ * c_e_av,
            "X-averaged electrolyte concentration [Molar]": c_e_typ * c_e_av / 1000,
            "Negative electrolyte concentration": c_e_n,
            "Negative electrolyte concentration [mol.m-3]": c_e_typ * c_e_n,
            "Negative electrolyte concentration [Molar]": c_e_typ * c_e_n / 1000,
            "Separator electrolyte concentration": c_e_s,
            "Separator electrolyte concentration [mol.m-3]": c_e_typ * c_e_s,
            "Separator electrolyte concentration [Molar]": c_e_typ * c_e_s / 1000,
            "Positive electrolyte concentration": c_e_p,
            "Positive electrolyte concentration [mol.m-3]": c_e_typ * c_e_p,
            "Positive electrolyte concentration [Molar]": c_e_typ * c_e_p / 1000,
            "X-averaged negative electrolyte concentration": c_e_n_av,
            "X-averaged negative electrolyte concentration [mol.m-3]": c_e_typ
            * c_e_n_av,
            "X-averaged separator electrolyte concentration": c_e_s_av,
            "X-averaged separator electrolyte concentration [mol.m-3]": c_e_typ
            * c_e_s_av,
            "X-averaged positive electrolyte concentration": c_e_p_av,
            "X-averaged positive electrolyte concentration [mol.m-3]": c_e_typ
            * c_e_p_av,
        }

        # Override print_name
        c_e.print_name = "c_e"

        return variables

    def _get_standard_porosity_times_concentration_variables(
        self, eps_c_e_n, eps_c_e_s, eps_c_e_p
    ):
        eps_c_e = pybamm.concatenation(eps_c_e_n, eps_c_e_s, eps_c_e_p)

        variables = {
            "Porosity times concentration": eps_c_e,
            "Separator porosity times concentration": eps_c_e_s,
            "Positive electrode porosity times concentration": eps_c_e_p,
        }

        if not self.half_cell:
            variables.update(
                {"Negative electrode porosity times concentration": eps_c_e_n}
            )
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
