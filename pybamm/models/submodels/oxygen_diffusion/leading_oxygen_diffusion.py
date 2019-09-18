#
# Class for leading-order oxygen diffusion
#
import pybamm

from .base_oxygen_diffusion import BaseModel


class LeadingOrder(BaseModel):
    """Class for conservation of mass of oxygen. (Leading refers to leading order
    of asymptotic reduction)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.oxgen_diffusion.BaseModel`
    """

    def __init__(self, param, reactions):
        super().__init__(param, reactions)

    def get_fundamental_variables(self):
        c_ox_av = pybamm.Variable("X-averaged oxygen concentration")
        c_ox_n = pybamm.FullBroadcast(
            c_ox_av, ["negative electrode"], "current collector"
        )
        c_ox_s = pybamm.FullBroadcast(c_ox_av, ["separator"], "current collector")
        c_ox_p = pybamm.FullBroadcast(
            c_ox_av, ["positive electrode"], "current collector"
        )
        c_ox = pybamm.Concatenation(c_ox_n, c_ox_s, c_ox_p)

        return self._get_standard_concentration_variables(c_ox)

    def get_coupled_variables(self, variables):

        N_ox = pybamm.FullBroadcast(
            0,
            ["negative electrode", "separator", "positive electrode"],
            "current collector",
        )

        variables.update(self._get_standard_flux_variables(N_ox))

        return variables

    def set_rhs(self, variables):

        param = self.param

        c_ox_av = variables["X-averaged oxygen concentration"]

        eps_n_av = variables["X-averaged negative electrode porosity"]
        eps_s_av = variables["X-averaged separator porosity"]
        eps_p_av = variables["X-averaged positive electrode porosity"]

        deps_n_dt_av = variables["X-averaged negative electrode porosity change"]
        deps_p_dt_av = variables["X-averaged positive electrode porosity change"]

        source_terms = sum(
            param.l_n
            * rxn["Negative"]["s_ox"]
            * variables["X-averaged " + rxn["Negative"]["aj"].lower()]
            + param.l_p
            * rxn["Positive"]["s_ox"]
            * variables["X-averaged " + rxn["Positive"]["aj"].lower()]
            for rxn in self.reactions.values()
        )

        self.rhs = {
            c_ox_av: 1
            / (param.l_n * eps_n_av + param.l_s * eps_s_av + param.l_p * eps_p_av)
            * (
                source_terms
                - c_ox_av * (param.l_n * deps_n_dt_av + param.l_p * deps_p_dt_av)
            )
        }

    def set_initial_conditions(self, variables):
        c_ox = variables["X-averaged oxygen concentration"]
        self.initial_conditions = {c_ox: self.param.c_ox_init}
