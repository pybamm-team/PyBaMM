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

    def build(self):
        v_box_dict, div_v_box_dict, p_dict = {}, {}, {}
        for domain in self.options.whole_cell_domains:
            if domain == "separator":
                continue
            Domain = domain.capitalize()
            # Electrolyte pressure
            p_k = pybamm.Variable(
                f"{Domain} pressure [Pa]",
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
            )
            v_box_k = -pybamm.grad(p_k)
            div_v_box_k = pybamm.div(v_box_k)
            v_box_dict[domain], div_v_box_dict[domain], p_dict[domain] = (
                v_box_k,
                div_v_box_k,
                p_k,
            )
            self.variables.update(
                self._get_standard_convection_variables(
                    domain, v_box_k, div_v_box_k, p_k
                )
            )

            # Problem in the x-direction for p_k, multiplied by Lx**2 to improve
            # conditioning
            a_j_k = self.coupled_variable(
                f"{Domain} volumetric interfacial current density [A.m-3]",
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
            )
            DeltaV_k = self.param.domain_params[domain.split()[0]].DeltaV
            self.algebraic[p_k] = self.param.L_x**2 * (
                pybamm.div(v_box_k) + DeltaV_k * a_j_k / self.param.F
            )
            self.initial_conditions[p_k] = pybamm.Scalar(0)

        v_box_s, div_v_box_s = self._get_separator_velocity()
        self.variables.update(
            self._get_separator_and_whole_cell_variables(
                v_box_dict, div_v_box_dict, p_dict, v_box_s, div_v_box_s
            )
        )

        # Boundary conditions in x-direction for p_n and p_p
        p_s = self.coupled_variable(
            "X-averaged separator pressure [Pa]", domain="current collector"
        )
        self.boundary_conditions = {
            p_dict["negative electrode"]: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (p_s, "Dirichlet"),
            },
            p_dict["positive electrode"]: {
                "left": (p_s, "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            },
        }
