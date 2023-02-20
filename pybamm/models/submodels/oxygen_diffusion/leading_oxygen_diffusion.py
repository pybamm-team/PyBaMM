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
            "X-averaged oxygen concentration [mol.m-3]", domain="current collector"
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

        c_ox_av = variables["X-averaged oxygen concentration [mol.m-3]"]

        eps_n_av = variables["X-averaged negative electrode porosity"]
        eps_s_av = variables["X-averaged separator porosity"]
        eps_p_av = variables["X-averaged positive electrode porosity"]

        deps_n_dt_av = variables["X-averaged negative electrode porosity change [s-1]"]
        deps_p_dt_av = variables["X-averaged positive electrode porosity change [s-1]"]

        a_j_ox_n_av = variables[
            "X-averaged negative electrode oxygen "
            "volumetric interfacial current density [A.m-3]"
        ]
        a_j_ox_p_av = variables[
            "X-averaged positive electrode oxygen "
            "volumetric interfacial current density [A.m-3]"
        ]

        source_terms = (
            param.n.L * param.s_ox_Ox * a_j_ox_n_av
            + param.p.L * param.s_ox_Ox * a_j_ox_p_av
        )

        self.rhs = {
            c_ox_av: 1
            / (param.n.L * eps_n_av + param.s.L * eps_s_av + param.p.L * eps_p_av)
            * (
                source_terms / param.F
                - c_ox_av * (param.n.L * deps_n_dt_av + param.p.L * deps_p_dt_av)
            )
        }

    def set_initial_conditions(self, variables):
        c_ox = variables["X-averaged oxygen concentration [mol.m-3]"]
        self.initial_conditions = {c_ox: self.param.c_ox_init}
