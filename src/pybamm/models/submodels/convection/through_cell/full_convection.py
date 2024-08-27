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
                    f"{Domain} pressure [Pa]",
                    domain=domain,
                    auxiliary_domains={"secondary": "current collector"},
                )
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
        L_n = param.n.L
        x_s = pybamm.standard_spatial_vars.x_s

        # Transverse velocity in the separator determines through-cell velocity
        div_Vbox_s = variables[
            "X-averaged separator transverse volume-averaged acceleration [m.s-2]"
        ]
        i_boundary_cc = variables["Current collector current density [A.m-2]"]
        v_box_n_right = -param.n.DeltaV * i_boundary_cc / self.param.F
        div_v_box_s_av = -div_Vbox_s
        div_v_box_s = pybamm.PrimaryBroadcast(div_v_box_s_av, "separator")

        # Simple formula for velocity in the separator
        v_box_s = div_v_box_s_av * (x_s - L_n) + v_box_n_right

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
        p_n = variables["Negative electrode pressure [Pa]"]
        p_p = variables["Positive electrode pressure [Pa]"]

        a_j_n = variables[
            "Negative electrode volumetric interfacial current density [A.m-3]"
        ]
        a_j_p = variables[
            "Positive electrode volumetric interfacial current density [A.m-3]"
        ]

        v_box_n = variables["Negative electrode volume-averaged velocity [m.s-1]"]
        v_box_p = variables["Positive electrode volume-averaged velocity [m.s-1]"]

        # Problems in the x-direction for p_n and p_p
        # multiply by Lx**2 to improve conditioning
        self.algebraic = {
            p_n: self.param.L_x**2
            * (pybamm.div(v_box_n) + self.param.n.DeltaV * a_j_n / self.param.F),
            p_p: self.param.L_x**2
            * (pybamm.div(v_box_p) + self.param.p.DeltaV * a_j_p / self.param.F),
        }

    def set_boundary_conditions(self, variables):
        p_n = variables["Negative electrode pressure [Pa]"]
        p_s = variables["X-averaged separator pressure [Pa]"]
        p_p = variables["Positive electrode pressure [Pa]"]

        # Boundary conditions in x-direction for p_n and p_p
        self.boundary_conditions = {
            p_n: {"left": (pybamm.Scalar(0), "Neumann"), "right": (p_s, "Dirichlet")},
            p_p: {"left": (p_s, "Dirichlet"), "right": (pybamm.Scalar(0), "Neumann")},
        }

    def set_initial_conditions(self, variables):
        p_n = variables["Negative electrode pressure [Pa]"]
        p_p = variables["Positive electrode pressure [Pa]"]

        self.initial_conditions = {p_n: pybamm.Scalar(0), p_p: pybamm.Scalar(0)}
