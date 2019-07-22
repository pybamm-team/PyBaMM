#
# Class for oxygen diffusion
#
import pybamm

from .base_oxygen_diffusion import BaseModel


class FirstOrder(BaseModel):
    """Class for conservation of mass of oxygen. (First-order refers to first-order
    expansion in asymptotic methods)
    In this model, extremely fast oxygen kinetics in the negative electrode imposes
    zero oxygen concentration there, and so the oxygen variable only lives in the
    separator and positive electrode. The boundary condition at the negative electrode/
    separator interface is homogeneous Dirichlet.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.oxygen_diffusion.BaseModel`
    """

    def __init__(self, param, reactions):
        super().__init__(param, reactions)

    def get_coupled_variables(self, variables):

        param = self.param
        l_n = param.l_n
        l_s = param.l_s
        l_p = param.l_p
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        # Unpack
        eps_0 = variables["Leading-order porosity"]
        _, eps_s_0, eps_p_0 = [e.orphans[0] for e in eps_0.orphans]

        # Diffusivities
        D_ox_s = (eps_s_0 ** param.b) * param.curlyD_ox
        D_ox_p = (eps_p_0 ** param.b) * param.curlyD_ox

        # Reactions
        sj_ox_p = sum(
            reaction["Positive"]["s_ox"]
            * variables["Leading-order " + reaction["Positive"]["aj"].lower()].orphans[
                0
            ]
            for reaction in self.reactions.values()
        )

        # Fluxes
        N_ox_n_1 = pybamm.SecondaryBroadcast(
            pybamm.Broadcast(0, "negative electrode"), "current collector"
        )
        N_ox_s_1 = -pybamm.PrimaryBroadcast(sj_ox_p * l_p, "separator")
        N_ox_p_1 = pybamm.outer(sj_ox_p, x_p - 1)

        # Concentrations
        c_ox_n_1 = pybamm.Broadcast(0, "negative electrode")
        c_ox_s_1 = pybamm.outer(sj_ox_p * l_p / D_ox_s, x_s - l_n)
        c_ox_p_1 = pybamm.outer(
            -sj_ox_p / (2 * D_ox_p), (x_p - 1) ** 2 - l_p ** 2
        ) + pybamm.PrimaryBroadcast(sj_ox_p * l_p * l_s / D_ox_s, "positive electrode")

        # Update variables
        c_ox = pybamm.Concatenation(
            param.C_e * c_ox_n_1, param.C_e * c_ox_s_1, param.C_e * c_ox_p_1
        )
        variables.update(self._get_standard_concentration_variables(c_ox))

        N_ox = pybamm.Concatenation(
            param.C_e * N_ox_n_1, param.C_e * N_ox_s_1, param.C_e * N_ox_p_1
        )
        variables.update(self._get_standard_flux_variables(N_ox))

        return variables
