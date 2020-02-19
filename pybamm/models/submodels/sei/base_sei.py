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
        sp = pybamm.sei_parameters

        L_inner_av = pybamm.x_average(L_inner)
        L_outer_av = pybamm.x_average(L_outer)

        L_sei = L_inner + L_outer
        L_sei_av = pybamm.x_average(L_sei)

        n_inner = L_inner  # inner SEI concentration
        n_outer = L_outer  # outer SEI concentration
        n_inner_av = pybamm.x_average(L_inner)
        n_outer_av = pybamm.x_average(L_outer)

        n_SEI = n_inner + n_outer / sp.v_bar  # SEI concentration
        n_SEI_av = pybamm.x_average(n_SEI)

        Q_sei = n_SEI_av * self.param.L_n * self.param.L_y * self.param.L_z

        L_scale = sp.L_sei_0_dim
        n_scale = sp.L_sei_0_dim * sp.a_n / sp.V_bar_inner_dimensional
        n_outer_scale = sp.L_sei_0_dim * sp.a_n / sp.V_bar_outer_dimensional

        variables = {
            "Inner SEI thickness": L_inner,
            "Inner SEI thickness [m]": L_inner * L_scale,
            "X-averaged inner SEI thickness": L_inner_av,
            "X-averaged inner SEI thickness [m]": L_inner_av * L_scale,
            "Outer SEI thickness": L_outer,
            "Outer SEI thickness [m]": L_outer * L_scale,
            "X-averaged outer SEI thickness": L_outer_av,
            "X-averaged outer SEI thickness [m]": L_outer_av * L_scale,
            "Total SEI thickness": L_sei,
            "Total SEI thickness [m]": L_sei * L_scale,
            "X-averaged total SEI thickness": L_sei_av,
            "X-averaged total SEI thickness [m]": L_sei_av * L_scale,
            "Inner SEI concentration [mol.m-3]": n_inner * n_scale,
            "X-averaged inner SEI concentration [mol.m-3]": n_inner_av * n_scale,
            "Outer SEI concentration [mol.m-3]": n_outer * n_outer_scale,
            "X-averaged outer SEI concentration [mol.m-3]": n_outer_av * n_outer_scale,
            "SEI concentration [mol.m-3]": n_SEI * n_scale,
            "X-averaged SEI concentration [mol.m-3]": n_SEI_av * n_scale,
            "Loss of lithium to SEI [mols]": Q_sei * n_scale,
        }

        return variables

    def _get_standard_reaction_variables(self, j_inner, j_outer):
        """
            A private function to obtain the standard variables which
            can be derived from the SEI interfacial reaction current

            Parameters
            ----------
            j_inner : :class:`pybamm.Symbol`
                The inner SEI interfacial reaction current.
            j_outer : :class:`pybamm.Symbol`
                The outer SEI interfacial reaction current.

            Returns
            -------
            variables : dict
                The variables which can be derived from the SEI thicknesses.
        """

        j_i_av = pybamm.x_average(j_inner)
        j_o_av = pybamm.x_average(j_outer)

        j_sei = j_inner + j_outer
        j_sei_av = pybamm.x_average(j_sei)

        sp = pybamm.sei_parameters
        j_scale = sp.F * sp.L_sei_0_dim / sp.V_bar_inner_dimensional / sp.tau_discharge

        variables = {
            "Inner SEI reaction interfacial current density": j_inner,
            "Inner SEI reaction interfacial current density [A.m-2]": j_inner * j_scale,
            "X-averaged inner SEI reaction interfacial current density": j_i_av,
            "X-averaged inner SEI reaction interfacial current density [A.m-2]": j_i_av
            * j_scale,
            "Outer SEI reaction interfacial current density": j_outer,
            "Outer SEI reaction interfacial current density [A.m-2]": j_outer * j_scale,
            "X-averaged outer SEI reaction interfacial current density": j_o_av,
            "X-averaged outer SEI reaction interfacial current density [A.m-2]": j_o_av
            * j_scale,
            "SEI reaction interfacial current density": j_sei,
            "SEI reaction interfacial current density [A.m-2]": j_sei * j_scale,
            "X-averaged SEI reaction interfacial current density": j_sei_av,
            "X-averaged SEI reaction interfacial current density [A.m-2]": j_sei_av
            * j_scale,
        }

        return variables
