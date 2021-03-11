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
    reactions : dict, optional
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param):
        super().__init__(param)

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

        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)
        c_e_av = pybamm.x_average(c_e)
        c_e_n_av = pybamm.x_average(c_e_n)
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

    def _get_total_concentration_electrolyte(self, c_e, epsilon):
        """
        A private function to obtain the total ion concentration in the electrolyte.

        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            The electrolyte concentration
        epsilon : :class:`pybamm.Symbol`
            The porosity

        Returns
        -------
        variables : dict
            The "Total concentration in electrolyte [mol]" variable.
        """

        c_e_typ = self.param.c_e_typ
        L_x = self.param.L_x
        A = self.param.A_cc

        c_e_total = pybamm.x_average(epsilon * c_e)

        variables = {
            "Total concentration in electrolyte [mol]": c_e_typ * L_x * A * c_e_total
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
