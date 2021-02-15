#
# Class for oxygen diffusion
#
import pybamm

from .full_oxygen_diffusion import Full, separator_and_positive_only


class Composite(Full):
    """Class for conservation of mass of oxygen. (Composite refers to composite
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
    extended : bool
        Whether to include feedback from the first-order terms

    **Extends:** :class:`pybamm.oxygen_diffusion.Full`
    """

    def __init__(self, param, extended=False):
        super().__init__(param)
        self.extended = extended

    def get_coupled_variables(self, variables):

        tor_0 = separator_and_positive_only(
            variables["Leading-order electrolyte tortuosity"]
        )
        c_ox = variables["Separator and positive electrode oxygen concentration"]

        param = self.param

        N_ox_diffusion = -tor_0 * param.curlyD_ox * pybamm.grad(c_ox)

        # Note: no convection because c_ox_0 = 0 (at leading order)
        N_ox = N_ox_diffusion
        # Flux in the negative electrode is zero
        N_ox = pybamm.Concatenation(
            pybamm.FullBroadcast(0, "negative electrode", "current collector"), N_ox
        )

        variables.update(self._get_standard_flux_variables(N_ox))

        return variables

    def set_rhs(self, variables):
        """Composite reaction-diffusion with source terms from leading order."""

        param = self.param

        eps_0 = separator_and_positive_only(variables["Leading-order porosity"])
        deps_0_dt = separator_and_positive_only(
            variables["Leading-order porosity change"]
        )
        c_ox = variables["Separator and positive electrode oxygen concentration"]
        N_ox = variables["Oxygen flux"].orphans[1]

        if self.extended is False:
            j_ox_0 = variables[
                "Leading-order positive electrode oxygen interfacial current density"
            ]
            pos_reactions = param.s_ox_Ox * j_ox_0
        else:
            j_ox_0 = variables["Positive electrode oxygen interfacial current density"]
            pos_reactions = param.s_ox_Ox * j_ox_0
        sep_reactions = pybamm.FullBroadcast(0, "separator", "current collector")
        source_terms_0 = (
            pybamm.Concatenation(sep_reactions, pos_reactions) / param.gamma_e
        )

        self.rhs = {
            c_ox: (1 / eps_0)
            * (-pybamm.div(N_ox) / param.C_e + source_terms_0 - c_ox * deps_0_dt)
        }
