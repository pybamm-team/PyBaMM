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
            typ_dim = "velocity [m.s-1]"
        elif typ == "acceleration":
            scale = self.param.velocity_scale / self.param.L_z
            typ_dim = "acceleration [m.s-2]"

        var_dict = {}
        variables = {}
        for domain in self.options.whole_cell_domains:
            if domain == "separator":
                var_k_av = var_s_av
            else:
                var_k_av = pybamm.PrimaryBroadcast(0, "current collector")
            var_k = pybamm.PrimaryBroadcast(var_k_av, domain)
            var_dict[domain] = var_k

            variables.update(
                {
                    f"{domain} transverse volume-averaged {typ}": var_k,
                    f"{domain} transverse volume-averaged {typ_dim}": scale * var_k,
                    f"X-averaged {domain} transverse volume-averaged "
                    f"{typ}": var_k_av,
                    f"X-averaged {domain} transverse volume-averaged "
                    f"{typ_dim}": scale * var_k_av,
                }
            )

        var = pybamm.concatenation(*var_dict.values())

        variables.update(
            {
                f"Transverse volume-averaged {typ}": var,
                f"Transverse volume-averaged {typ_dim}": scale * var,
            }
        )

        return variables
