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
    domain : str
        Either 'Negative' or 'Positive'
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.electrode.ohm.BaseModel`
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options=options)
        if self.domain == "Positive":
            raise NotImplementedError(
                "LithiumMetalExplicit model only implemented in negative electrode"
            )

    def get_coupled_variables(self, variables):
        param = self.param

        i_boundary_cc = variables["Current collector current density"]
        l_n = variables["Lithium metal electrode thickness"]
        delta_phi_s = i_boundary_cc * l_n / param.sigma_n
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
