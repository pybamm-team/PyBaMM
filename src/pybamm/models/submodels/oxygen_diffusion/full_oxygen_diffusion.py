#
# Class for oxygen diffusion
#
import pybamm

from .base_oxygen_diffusion import BaseModel


class Full(BaseModel):
    """Class for conservation of mass of oxygen. (Full refers to unreduced by
    asymptotic methods)
    In this model, extremely fast oxygen kinetics in the negative electrode imposes
    zero oxygen concentration there, and so the oxygen variable only lives in the
    separator and positive electrode. The boundary condition at the negative electrode/
    separator interface is homogeneous Dirichlet.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    """

    def __init__(self, param):
        super().__init__(param)

    def build(self):
        # Oxygen concentration (oxygen concentration is zero in the negative electrode)
        c_ox_n = pybamm.FullBroadcast(0, "negative electrode", "current collector")
        c_ox_s = pybamm.Variable(
            "Separator oxygen concentration [mol.m-3]",
            domain="separator",
            auxiliary_domains={"secondary": "current collector"},
        )
        c_ox_p = pybamm.Variable(
            "Positive oxygen concentration [mol.m-3]",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        c_ox_s_p = pybamm.concatenation(c_ox_s, c_ox_p)
        variables = {
            "Separator and positive electrode oxygen concentration [mol.m-3]": c_ox_s_p
        }

        variables.update(
            self._get_standard_concentration_variables(c_ox_n, c_ox_s, c_ox_p)
        )

        tor_s = pybamm.CoupledVariable(
            "Separator electrolyte transport efficiency",
            domain="separator",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({tor_s.name: tor_s})  
        tor_p = pybamm.CoupledVariable(
            "Positive electrolyte transport efficiency",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({tor_p.name: tor_p})
        tor = pybamm.concatenation(tor_s, tor_p)

        c_ox = c_ox_s_p

        # TODO: allow charge and convection?
        v_box = pybamm.Scalar(0)

        N_ox_diffusion = -tor * self.param.D_ox * pybamm.grad(c_ox)

        N_ox = N_ox_diffusion + c_ox * v_box
        # Flux in the negative electrode is zero
        N_ox = pybamm.concatenation(
            pybamm.FullBroadcast(0, "negative electrode", "current collector"), N_ox
        )

        variables.update(self._get_standard_flux_variables(N_ox))

        deps_dt_s = pybamm.CoupledVariable(
            "Separator porosity change [s-1]",
            domain="separator",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({deps_dt_s.name: deps_dt_s})
        deps_dt_p = pybamm.CoupledVariable(
            "Positive electrode porosity change [s-1]",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({deps_dt_p.name: deps_dt_p})
        deps_dt = pybamm.concatenation(deps_dt_s, deps_dt_p)


        N_ox = variables["Oxygen flux [mol.m-2.s-1]"].orphans[1]

        a_j_ox = pybamm.CoupledVariable(
            "Positive electrode oxygen volumetric interfacial current density [A.m-3]",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({a_j_ox.name: a_j_ox})
        source_terms = pybamm.concatenation(
            pybamm.FullBroadcast(0, "separator", "current collector"),
            self.param.s_ox_Ox * a_j_ox,
        )

        eps_s = pybamm.CoupledVariable(
            "Separator porosity",
            domain="separator",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({eps_s.name: eps_s})
        eps_p = pybamm.CoupledVariable(
            "Positive electrode porosity",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({eps_p.name: eps_p})
        eps = pybamm.concatenation(eps_s, eps_p)
        self.rhs = {
            c_ox: (1 / eps)
            * (-pybamm.div(N_ox) + source_terms / self.param.F - c_ox * deps_dt)
        }
        self.boundary_conditions = {
            c_ox: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        self.initial_conditions = {c_ox: self.param.c_ox_init}
        self.variables.update(variables)
