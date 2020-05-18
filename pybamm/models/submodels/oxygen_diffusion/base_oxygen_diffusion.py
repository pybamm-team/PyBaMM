#
# Base class for oxygen diffusion
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Base class for conservation of mass of oxygen.

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

    def _get_standard_concentration_variables(self, c_ox):
        """
        A private function to obtain the standard variables which
        can be derived from the concentration of oxygen.

        Parameters
        ----------
        c_ox : :class:`pybamm.Symbol`
            The concentration of oxygen.
        c_ox_av : :class:`pybamm.Symbol`
            The cell-averaged concentration of oxygen.

        Returns
        -------
        variables : dict
            The variables which can be derived from the concentration in the
            oxygen.
        """

        c_ox_av = pybamm.x_average(c_ox)
        c_ox_typ = self.param.c_ox_typ
        c_ox_n, c_ox_s, c_ox_p = c_ox.orphans

        variables = {
            "Oxygen concentration": c_ox,
            "Oxygen concentration [mol.m-3]": c_ox_typ * c_ox,
            "Oxygen concentration [Molar]": c_ox_typ * c_ox / 1000,
            "X-averaged oxygen concentration": c_ox_av,
            "X-averaged oxygen concentration [mol.m-3]": c_ox_typ * c_ox_av,
            "X-averaged oxygen concentration [Molar]": c_ox_typ * c_ox_av / 1000,
            "Negative oxygen concentration": c_ox_n,
            "Negative oxygen concentration [mol.m-3]": c_ox_typ * c_ox_n,
            "Negative oxygen concentration [Molar]": c_ox_typ * c_ox_n / 1000,
            "Separator oxygen concentration": c_ox_s,
            "Separator oxygen concentration [mol.m-3]": c_ox_typ * c_ox_s,
            "Separator oxygen concentration [Molar]": c_ox_typ * c_ox_s / 1000,
            "Positive oxygen concentration": c_ox_p,
            "Positive oxygen concentration [mol.m-3]": c_ox_typ * c_ox_p,
            "Positive oxygen concentration [Molar]": c_ox_typ * c_ox_p / 1000,
        }

        return variables

    def _get_standard_flux_variables(self, N_ox):
        """
        A private function to obtain the standard variables which
        can be derived from the mass flux of oxygen.

        Parameters
        ----------
        N_ox : :class:`pybamm.Symbol`
            The flux of oxygen.

        Returns
        -------
        variables : dict
            The variables which can be derived from the flux of oxygen.
        """

        param = self.param
        flux_scale = param.curlyD_ox * param.c_ox_typ / param.L_x

        variables = {
            "Oxygen flux": N_ox,
            "Oxygen flux [mol.m-2.s-1]": N_ox * flux_scale,
        }

        return variables
