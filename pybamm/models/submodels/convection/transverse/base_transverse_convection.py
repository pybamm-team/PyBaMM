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
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.convection.BaseModel`
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

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

        if self.half_cell:
            var_n_av = None
            var_n = None
        else:
            var_n_av = pybamm.PrimaryBroadcast(0, "current collector")
            var_n = pybamm.PrimaryBroadcast(var_n_av, "negative electrode")

        var_p_av = pybamm.PrimaryBroadcast(0, "current collector")
        var_s = pybamm.PrimaryBroadcast(var_s_av, "separator")
        var_p = pybamm.PrimaryBroadcast(var_p_av, "positive electrode")
        var = pybamm.concatenation(var_n, var_s, var_p)

        variables = {
            "Separator transverse volume-averaged " + typ: var_s,
            "Positive electrode transverse volume-averaged " + typ: var_p,
            "Separator transverse volume-averaged " + typ + " [m.s-2]": scale * var_s,
            "Positive electrode transverse volume-averaged "
            + typ
            + " [m.s-2]": scale * var_p,
            "X-averaged separator transverse volume-averaged " + typ: var_s_av,
            "X-averaged positive electrode transverse volume-averaged " + typ: var_p_av,
            "X-averaged separator transverse volume-averaged "
            + typ
            + " [m.s-2]": scale * var_s_av,
            "X-averaged positive electrode transverse volume-averaged "
            + typ
            + " [m.s-2]": scale * var_p_av,
            "Transverse volume-averaged " + typ: var,
            "Transverse volume-averaged " + typ + " [m.s-2]": scale * var,
        }

        if not self.half_cell:
            variables.update(
                {
                    "Negative electrode transverse volume-averaged " + typ: var_n,
                    "Negative electrode transverse volume-averaged "
                    + typ
                    + " [m.s-2]": scale * var_n,
                    "X-averaged negative electrode transverse volume-averaged "
                    + typ: var_n_av,
                    "X-averaged negative electrode transverse volume-averaged "
                    + typ
                    + " [m.s-2]": scale * var_n_av,
                }
            )

        return variables
