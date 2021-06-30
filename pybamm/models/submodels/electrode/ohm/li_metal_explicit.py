#
# Explicit model for potential drop across a lithium metal electrode
#
import pybamm
from .base_ohm import BaseModel


class LiMetalExplicit(BaseModel):
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
                "LiMetalExplicit model only implemented in negative electrode"
            )

    def get_coupled_variables(self, variables):
        param = self.param

        # Reference value at the negative electrode / separator interface
        delta_phi_s_right = variables[
            "X-averaged negative electrode surface potential difference"
        ]
        phi_e_right = param.U_n_ref / param.potential_scale
        phi_s_right = delta_phi_s_right + phi_e_right

        i_boundary_cc = variables["Current collector current density"]
        sigma = self.param.sigma_n
        x_n = pybamm.standard_spatial_vars.x_n

        phi_s = phi_s_right - i_boundary_cc * (x_n - param.l_n) / sigma
        variables.update(self._get_standard_potential_variables(phi_s))

        return variables
