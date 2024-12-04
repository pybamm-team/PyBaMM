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

    def build(self, submodels):
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
        L_n = self.param.n.L
        x_s = pybamm.standard_spatial_vars.x_s

        # Transverse velocity in the separator determines through-cell velocity
        div_Vbox_s = pybamm.CoupledVariable(
            "X-averaged separator transverse volume-averaged acceleration [m.s-2]",
            domain="current collector",
        )
        self.coupled_variables.update({div_Vbox_s.name: div_Vbox_s})
        i_boundary_cc = pybamm.CoupledVariable(
            "Current collector current density [A.m-2]",
            domain="current collector",
        )
        self.coupled_variables.update({i_boundary_cc.name: i_boundary_cc})
        v_box_n_right = -self.param.n.DeltaV * i_boundary_cc / self.param.F
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

        p_n = variables["Negative electrode pressure [Pa]"]
        p_p = variables["Positive electrode pressure [Pa]"]
        a_j_n = pybamm.CoupledVariable(
            "Negative electrode volumetric interfacial current density [A.m-3]",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({a_j_n.name: a_j_n})
        a_j_p = pybamm.CoupledVariable(
            "Positive electrode volumetric interfacial current density [A.m-3]",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({a_j_p.name: a_j_p})

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
        p_s = pybamm.CoupledVariable(
            "X-averaged separator pressure [Pa]",
            domain="current collector",
        )
        self.coupled_variables.update({p_s.name: p_s})

        # Boundary conditions in x-direction for p_n and p_p
        self.boundary_conditions = {
            p_n: {"left": (pybamm.Scalar(0), "Neumann"), "right": (p_s, "Dirichlet")},
            p_p: {"left": (p_s, "Dirichlet"), "right": (pybamm.Scalar(0), "Neumann")},
        }
        self.variables.update(variables)
        self.initial_conditions = {p_n: pybamm.Scalar(0), p_p: pybamm.Scalar(0)}
