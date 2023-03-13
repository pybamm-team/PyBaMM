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


    **Extends:** :class:`pybamm.oxgen_diffusion.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        c_ox_av = pybamm.Variable(
            "X-averaged oxygen concentration", domain="current collector"
        )
        c_ox_n = pybamm.PrimaryBroadcast(c_ox_av, "negative electrode")
        c_ox_s = pybamm.PrimaryBroadcast(c_ox_av, "separator")
        c_ox_p = pybamm.PrimaryBroadcast(c_ox_av, "positive electrode")

        return self._get_standard_concentration_variables(c_ox_n, c_ox_s, c_ox_p)

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

        j_ox_n_av = variables[
            "X-averaged negative electrode oxygen interfacial current density"
        ]
        j_ox_p_av = variables[
            "X-averaged positive electrode oxygen interfacial current density"
        ]

        source_terms = (
            param.n.l * param.s_ox_Ox * j_ox_n_av
            + param.p.l * param.s_ox_Ox * j_ox_p_av
        )

        self.rhs = {
            c_ox_av: 1
            / (param.n.l * eps_n_av + param.s.l * eps_s_av + param.p.l * eps_p_av)
            * (
                source_terms
                - c_ox_av * (param.n.l * deps_n_dt_av + param.p.l * deps_p_dt_av)
            )
        }

    def set_initial_conditions(self, variables):
        c_ox = variables["X-averaged oxygen concentration"]
        self.initial_conditions = {c_ox: self.param.c_ox_init}
