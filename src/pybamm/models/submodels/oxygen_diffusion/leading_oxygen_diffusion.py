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
    """

    def __init__(self, param):
        super().__init__(param)

    def build(self):
        c_ox_av = pybamm.Variable(
            "X-averaged oxygen concentration [mol.m-3]", domain="current collector"
        )
        c_ox_n = pybamm.PrimaryBroadcast(c_ox_av, "negative electrode")
        c_ox_s = pybamm.PrimaryBroadcast(c_ox_av, "separator")
        c_ox_p = pybamm.PrimaryBroadcast(c_ox_av, "positive electrode")

        variables = self._get_standard_concentration_variables(c_ox_n, c_ox_s, c_ox_p)

        N_ox = pybamm.FullBroadcast(
            0,
            ["negative electrode", "separator", "positive electrode"],
            "current collector",
        )

        variables.update(self._get_standard_flux_variables(N_ox))

        eps_n_av = pybamm.CoupledVariable(
            "X-averaged negative electrode porosity",
            domain="current collector"
        )
        self.coupled_variables.update({eps_n_av.name: eps_n_av})
        eps_s_av = pybamm.CoupledVariable(
            "X-averaged separator porosity",
            domain="current collector",
        )
        self.coupled_variables.update({eps_s_av.name: eps_s_av})
        eps_p_av = pybamm.CoupledVariable(
            "X-averaged positive electrode porosity",
            domain="current collector",
        )
        self.coupled_variables.update({eps_p_av.name: eps_p_av})

        deps_n_dt_av = pybamm.CoupledVariable(
            "X-averaged negative electrode porosity change [s-1]",
            domain="current collector",
        )
        self.coupled_variables.update({deps_n_dt_av.name: deps_n_dt_av})
        deps_p_dt_av = pybamm.CoupledVariable(
            "X-averaged positive electrode porosity change [s-1]",
            domain="current collector",
        )
        self.coupled_variables.update({deps_p_dt_av.name: deps_p_dt_av})

        a_j_ox_n_av = pybamm.CoupledVariable(
            "X-averaged negative electrode oxygen "
            "volumetric interfacial current density [A.m-3]",
            domain="current collector",
        )
        self.coupled_variables.update({a_j_ox_n_av.name: a_j_ox_n_av})
        a_j_ox_p_av = pybamm.CoupledVariable(
            "X-averaged positive electrode oxygen "
            "volumetric interfacial current density [A.m-3]",
            domain="current collector",
        )
        self.coupled_variables.update({a_j_ox_p_av.name: a_j_ox_p_av})

        source_terms = (
            self.param.n.L * self.param.s_ox_Ox * a_j_ox_n_av
            + self.param.p.L * self.param.s_ox_Ox * a_j_ox_p_av
        )

        self.rhs = {
            c_ox_av: 1
            / (
                self.param.n.L * eps_n_av
                + self.param.s.L * eps_s_av
                + self.param.p.L * eps_p_av
            )
            * (
                source_terms / self.param.F
                - c_ox_av
                * (self.param.n.L * deps_n_dt_av + self.param.p.L * deps_p_dt_av)
            )
        }
        self.initial_conditions = {c_ox_av: self.param.c_ox_init}
        self.variables.update(variables)
