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


    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def _get_standard_concentration_variables(self, c_e):
        """
        A private function to obtain the standard variables which
        can be derived from the concentration in the electrolyte.

        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            The concentration in the electrolyte.
        c_e_av : :class:`pybamm.Symbol`
            The cell-averaged concentration in the electrolyte.

        Returns
        -------
        variables : dict
            The variables which can be derived from the concentration in the
            electrolyte.
        """

        c_e_typ = self.param.c_e_typ
        c_e_n, c_e_s, c_e_p = c_e.orphans
        c_e_av = pybamm.average(c_e)

        variables = {
            "Electrolyte concentration": c_e,
            "Electrolyte concentration [mol.m-3]": c_e_typ * c_e,
            "Average electrolyte concentration": c_e_av,
            "Average electrolyte concentration [mol.m-3]": c_e_typ * c_e_av,
            "Negative electrolyte concentration": c_e_n,
            "Negative electrolyte concentration [mol.m-3]": c_e_typ * c_e_n,
            "Separator electrolyte concentration": c_e_s,
            "Separator electrolyte concentration [mol.m-3]": c_e_typ * c_e_s,
            "Positive electrolyte concentration": c_e_p,
            "Positive electrolyte concentration [mol.m-3]": c_e_typ * c_e_p,
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
        D_e_typ = param.D_e(param.c_e_typ)
        flux_scale = D_e_typ * param.c_e_typ / param.L_x

        variables = {
            "Electrolyte flux": N_e,
            "Electrolyte flux [mol.m-2.s-1]": N_e * flux_scale,
        }

        return variables

    def set_events(self, variables):
        c_e = variables["Electrolyte concentration"]
        self.events["Zero electrolyte concentration cut-off"] = pybamm.min(c_e) - 0.002
