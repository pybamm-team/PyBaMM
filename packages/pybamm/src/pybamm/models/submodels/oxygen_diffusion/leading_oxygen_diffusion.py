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
        self.variables.update(
            self._get_standard_concentration_variables(c_ox_n, c_ox_s, c_ox_p)
        )

        N_ox = pybamm.FullBroadcast(
            0,
            ["negative electrode", "separator", "positive electrode"],
            "current collector",
        )
        self.variables.update(self._get_standard_flux_variables(N_ox))

        def coupled(name):
            return self.coupled_variable(name, domain="current collector")

        eps_n_av = coupled("X-averaged negative electrode porosity")
        eps_s_av = coupled("X-averaged separator porosity")
        eps_p_av = coupled("X-averaged positive electrode porosity")
        deps_n_dt_av = coupled("X-averaged negative electrode porosity change [s-1]")
        deps_p_dt_av = coupled("X-averaged positive electrode porosity change [s-1]")
        a_j_ox_n_av = coupled(
            "X-averaged negative electrode oxygen "
            "volumetric interfacial current density [A.m-3]"
        )
        a_j_ox_p_av = coupled(
            "X-averaged positive electrode oxygen "
            "volumetric interfacial current density [A.m-3]"
        )

        source_terms = (
            self.param.n.L * self.param.s_ox_Ox * a_j_ox_n_av
            + self.param.p.L * self.param.s_ox_Ox * a_j_ox_p_av
        )
        self.rhs[c_ox_av] = (
            1
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
        )
        self.initial_conditions[c_ox_av] = self.param.c_ox_init
