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
        c_ox = pybamm.concatenation(
            c_ox_s,
            c_ox_p,
            name="Separator and positive electrode oxygen concentration [mol.m-3]",
        )
        self.variables[
            "Separator and positive electrode oxygen concentration [mol.m-3]"
        ] = c_ox
        self.variables.update(
            self._get_standard_concentration_variables(c_ox_n, c_ox_s, c_ox_p)
        )

        def coupled(name, domain):
            return self.coupled_variable(
                name,
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
            )

        def concatenate(name_s, name_p):
            return pybamm.concatenation(
                coupled(name_s, "separator"), coupled(name_p, "positive electrode")
            )

        tor = concatenate(
            "Separator electrolyte transport efficiency",
            "Positive electrolyte transport efficiency",
        )
        eps = concatenate("Separator porosity", "Positive electrode porosity")
        deps_dt = concatenate(
            "Separator porosity change [s-1]",
            "Positive electrode porosity change [s-1]",
        )
        a_j_ox = coupled(
            "Positive electrode oxygen volumetric interfacial current density [A.m-3]",
            "positive electrode",
        )

        # TODO: allow charge and convection?
        v_box = pybamm.Scalar(0)
        N_ox = -tor * self.param.D_ox * pybamm.grad(c_ox) + c_ox * v_box
        # Flux in the negative electrode is zero
        self.variables.update(
            self._get_standard_flux_variables(
                pybamm.concatenation(
                    pybamm.FullBroadcast(0, "negative electrode", "current collector"),
                    N_ox,
                )
            )
        )

        source_terms = pybamm.concatenation(
            pybamm.FullBroadcast(0, "separator", "current collector"),
            self.param.s_ox_Ox * a_j_ox,
        )
        self.rhs[c_ox] = (1 / eps) * (
            -pybamm.div(N_ox) + source_terms / self.param.F - c_ox * deps_dt
        )
        self.boundary_conditions[c_ox] = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_ox] = self.param.c_ox_init
