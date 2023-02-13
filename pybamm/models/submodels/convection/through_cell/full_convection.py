#
# Submodel for pressure driven convection
#
import pybamm
from .base_through_cell_convection import BaseThroughCellModel


class Full(BaseThroughCellModel):
    """Submodel for the full model of pressure-driven convection

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.convection.through_cell.BaseThroughCellModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        variables = {}
        for domain in self.options.whole_cell_domains:
            if domain != "separator":
                Domain = domain.capitalize()
                # Electrolyte pressure
                p_k = pybamm.Variable(
                    f"{Domain} pressure",
                    domain=domain,
                    auxiliary_domains={"secondary": "current collector"},
                )
                # TODO: add permeability and viscosity, and other terms
                v_mass_k = -pybamm.grad(p_k)
                v_box_k = v_mass_k

                div_v_box_k = pybamm.div(v_box_k)

                variables.update(
                    self._get_standard_convection_variables(
                        domain, v_box_k, div_v_box_k, p_k
                    )
                )

        return variables

    def get_coupled_variables(self, variables):
        # Set up
        param = self.param
        l_n = param.n.l
        x_s = pybamm.standard_spatial_vars.x_s

        # Transverse velocity in the separator determines through-cell velocity
        div_Vbox_s = variables[
            "X-averaged separator transverse volume-averaged acceleration"
        ]
        i_boundary_cc = variables["Current collector current density"]
        v_box_n_right = param.n.beta * i_boundary_cc
        div_v_box_s_av = -div_Vbox_s
        div_v_box_s = pybamm.PrimaryBroadcast(div_v_box_s_av, "separator")

        # Simple formula for velocity in the separator
        v_box_s = div_v_box_s_av * (x_s - l_n) + v_box_n_right

        variables.update(
            self._get_standard_sep_velocity_variables(v_box_s, div_v_box_s)
        )
        variables.update(self._get_standard_whole_cell_velocity_variables(variables))
        variables.update(
            self._get_standard_whole_cell_acceleration_variables(variables)
        )
        variables.update(self._get_standard_whole_cell_pressure_variables(variables))

        return variables

    def set_algebraic(self, variables):
        p_n = variables["Negative electrode pressure"]
        p_p = variables["Positive electrode pressure"]

        j_n = variables["Negative electrode interfacial current density"]
        j_p = variables["Positive electrode interfacial current density"]

        v_box_n = variables["Negative electrode volume-averaged velocity"]
        v_box_p = variables["Positive electrode volume-averaged velocity"]

        # Problems in the x-direction for p_n and p_p
        self.algebraic = {
            p_n: pybamm.div(v_box_n) - self.param.n.beta * j_n,
            p_p: pybamm.div(v_box_p) - self.param.p.beta * j_p,
        }

    def set_boundary_conditions(self, variables):
        p_n = variables["Negative electrode pressure"]
        p_s = variables["X-averaged separator pressure"]
        p_p = variables["Positive electrode pressure"]

        # Boundary conditions in x-direction for p_n and p_p
        self.boundary_conditions = {
            p_n: {"left": (pybamm.Scalar(0), "Neumann"), "right": (p_s, "Dirichlet")},
            p_p: {"left": (p_s, "Dirichlet"), "right": (pybamm.Scalar(0), "Neumann")},
        }

    def set_initial_conditions(self, variables):
        p_n = variables["Negative electrode pressure"]
        p_p = variables["Positive electrode pressure"]

        self.initial_conditions = {p_n: pybamm.Scalar(0), p_p: pybamm.Scalar(0)}
