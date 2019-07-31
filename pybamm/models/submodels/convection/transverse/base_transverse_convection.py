#
# Base class for convection submodels in transverse directions
#
import pybamm
from ..base_convection import BaseModel


class BaseTransverseModel(BaseModel):
    """Base class for convection submodels in transverse directions.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    **Extends:** :class:`pybamm.convection.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def _get_standard_separator_pressure_variables(self, p_s):
        """Pressure in the separator"""

        variables = {
            "Separator pressure": pybamm.PrimaryBroadcast(p_s, "separator"),
            "X-averaged separator pressure": p_s,
        }

        return variables

    def _get_standard_transverse_velocity_variables(self, var_s_av, typ):
        """Vertical acceleration in the separator"""
        if typ == "velocity":
            scale = self.param.velocity_scale
        elif typ == "acceleration":
            scale = self.param.velocity_scale / self.param.L_z

        var_n_av = pybamm.PrimaryBroadcast(0, "current collector")
        var_p_av = pybamm.PrimaryBroadcast(0, "current collector")
        var_n = pybamm.PrimaryBroadcast(var_n_av, "negative electrode")
        var_s = pybamm.PrimaryBroadcast(var_s_av, "separator")
        var_p = pybamm.PrimaryBroadcast(var_p_av, "positive electrode")

        variables = {
            "Negative electrode transverse volume-averaged " + typ: var_n,
            "Separator transverse volume-averaged " + typ: var_s,
            "Positive electrode transverse volume-averaged " + typ: var_p,
            "Negative electrode transverse volume-averaged "
            + typ
            + " [m.s-2]": scale * var_n,
            "Separator transverse volume-averaged " + typ + " [m.s-2]": scale * var_s,
            "Positive electrode transverse volume-averaged "
            + typ
            + " [m.s-2]": scale * var_p,
            "X-averaged negative electrode transverse volume-averaged " + typ: var_n_av,
            "X-averaged separator transverse volume-averaged " + typ: var_s_av,
            "X-averaged positive electrode transverse volume-averaged " + typ: var_p_av,
            "X-averaged negative electrode transverse volume-averaged "
            + typ
            + " [m.s-2]": scale * var_n_av,
            "X-averaged separator transverse volume-averaged "
            + typ
            + " [m.s-2]": scale * var_s_av,
            "X-averaged positive electrode transverse volume-averaged "
            + typ
            + " [m.s-2]": scale * var_p_av,
        }

        return variables

    def _get_separator_velocity(self, variables):
        """
        A private method to calculate x- and z-components of velocity in the separator

        Parameters
        ----------
        variables : dict
            Dictionary of variables in the whole model.

        Returns
        -------
        v_box_s : :class:`pybamm.Symbol`
            The x-component of velocity in the separator
        dVbox_dz : :class:`pybamm.Symbol`
            The z-component of velocity in the separator
        """
        # Set up
        param = self.param

        # Difference in negative and positive electrode velocities determines the
        # velocity in the separator
        i_boundary_cc = variables["Current collector current density"]
        v_box_n_right = param.beta_n * i_boundary_cc
        v_box_p_left = param.beta_p * i_boundary_cc
        d_vbox_s_dx = (v_box_p_left - v_box_n_right) / param.l_s

        # Simple formula for velocity in the separator
        div_Vbox_s = -d_vbox_s_dx

        return div_Vbox_s
