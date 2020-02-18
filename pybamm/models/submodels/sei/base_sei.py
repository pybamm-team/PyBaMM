#
# Base class for SEI models.
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Base class for SEI models.

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

    def _get_standard_thickness_variables(self, L_inner, L_outer):
        """
        A private function to obtain the standard variables which
        can be derived from the local SEI thickness.

        Parameters
        ----------
        L_inner : :class:`pybamm.Symbol`
            The inner SEI thickness.
        L_outer : :class:`pybamm.Symbol`
            The outer SEI thickness.

        Returns
        -------
        variables : dict
            The variables which can be derived from the SEI thicknesses.
        """

        L_inner_av = pybamm.x_average(L_inner)
        L_outer_av = pybamm.x_average(L_outer)

        n_inner = L_inner  # inner SEI concentration
        n_outer = L_outer  # outer SEI concentration
        n_inner_av = pybamm.x_average(L_inner)
        n_outer_av = pybamm.x_average(L_outer)

        n_SEI = n_inner + n_outer  # SEI concentration
        n_SEI_av = pybamm.x_average(n_SEI)

        Q_sei = n_SEI_av * self.param.L_n * self.param.L_y * self.L_z

        variables = {
            "Inner SEI thickness": L_inner,
            "Inner SEI thickness [m]": L_inner,
            "X-averaged inner SEI thickness": L_inner_av,
            "X-averaged inner SEI thickness [m]": L_inner_av,
            "Outer SEI thickness": L_outer,
            "Outer SEI thickness [m]": L_outer,
            "X-averaged outer SEI thickness": L_outer_av,
            "X-averaged outer SEI thickness [m]": L_outer_av,
            "Inner SEI concentration [mol.m-3]": n_inner,
            "X-averaged inner SEI concentration [mol.m-3]": n_inner_av,
            "Outer SEI concentration [mol.m-3]": n_outer,
            "X-averaged outer SEI concentration [mol.m-3]": n_outer_av,
            "SEI concentration [mol.m-3]": n_SEI,
            "X-averaged SEI concentration [mol.m-3]": n_SEI_av,
            "Loss of lithium to SEI [mols]": Q_sei,
        }

        return variables

