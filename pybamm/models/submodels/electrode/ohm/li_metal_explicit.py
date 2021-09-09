#
# Explicit model for potential drop across a lithium metal electrode
#
from .base_ohm import BaseModel


class LithiumMetalExplicit(BaseModel):
    """Explicit model for potential drop across a lithium metal electrode.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.electrode.ohm.BaseModel`
    """

    def __init__(self, param, options=None):
        super().__init__(param, "Negative", options=options)

    def get_coupled_variables(self, variables):
        param = self.param

        i_boundary_cc = variables["Current collector current density"]
        T_n = variables["Negative current collector temperature"]
        l_n = param.l_n
        delta_phi_s = i_boundary_cc * l_n / param.sigma_n(T_n)
        delta_phi_s_dim = param.potential_scale * delta_phi_s

        variables.update(
            {
                "Negative electrode potential drop": delta_phi_s,
                "Negative electrode potential drop [V]": delta_phi_s_dim,
                "X-averaged negative electrode ohmic losses": delta_phi_s / 2,
                "X-averaged negative electrode ohmic losses [V]": delta_phi_s_dim / 2,
            }
        )

        return variables

    def set_boundary_conditions(self, variables):
        pass
