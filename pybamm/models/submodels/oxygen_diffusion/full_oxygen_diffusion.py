#
# Class for oxygen diffusion
#
import pybamm

from .base_oxygen_diffusion import BaseModel


class Full(BaseModel):
    """Class for conservation of mass of oxygen. (Full refers to unreduced by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.oxygen.stefan_maxwell.diffusion.BaseModel`
    """

    def __init__(self, param, reactions):
        super().__init__(param, reactions)

    def get_fundamental_variables(self):
        c_ox = pybamm.standard_variables.c_ox

        return self._get_standard_concentration_variables(c_ox)

    def get_coupled_variables(self, variables):

        eps = variables["Porosity"]
        c_ox = variables["Oxygen concentration"]
        # i_ox = variables["Oxygen current density"]
        v_box = variables["Volume-averaged velocity"]

        param = self.param

        N_ox_diffusion = -(eps ** param.b) * param.D_ox(c_ox) * pybamm.grad(c_ox)
        # N_ox_migration = (param.C_ox * param.t_plus) / param.gamma_ox * i_ox
        # N_ox_convection = c_ox * v_box

        # N_ox = N_ox_diffusion + N_ox_migration + N_ox_convection

        N_ox = N_ox_diffusion + c_ox * v_box

        variables.update(self._get_standard_flux_variables(N_ox))

        return variables

    def set_rhs(self, variables):

        param = self.param

        eps = variables["Porosity"]
        deps_dt = variables["Porosity change"]
        c_ox = variables["Oxygen concentration"]
        N_ox = variables["Oxygen flux"]

        source_terms = sum(
            pybamm.Concatenation(
                pybamm.Broadcast(0, "separator"),
                reaction["Positive"]["s_ox"] * variables[reaction["Positive"]["aj"]],
            )
            for reaction in self.reactions.values()
        )

        self.rhs = {
            c_ox: (1 / eps)
            * (-pybamm.div(N_ox) / param.C_ox + source_terms - c_ox * deps_dt)
        }

    def set_boundary_conditions(self, variables):

        c_ox = variables["Oxygen concentration"]

        self.boundary_conditions = {
            c_ox: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }

    def set_initial_conditions(self, variables):

        c_ox = variables["Oxygen concentration"]

        self.initial_conditions = {c_ox: self.param.c_ox_init}
