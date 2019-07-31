#
# Submodel for pressure driven convection
#
import pybamm
from .base_convection import BaseModel


class BaseFull(BaseModel):
    """Submodel for the full model of pressure-driven convection

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.convection.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        # Electrolyte pressure
        p_n = pybamm.Variable(
            "Negative electrode pressure",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        p_p = pybamm.Variable(
            "Positive electrode pressure",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        p_s = self._get_separator_pressure()
        p = pybamm.Concatenation(p_n, pybamm.PrimaryBroadcast(p_s, "separator"), p_p)

        v_mass_n = -pybamm.grad(p_n)
        v_mass_p = -pybamm.grad(p_p)
        v_box_n = v_mass_n
        v_box_s = v_mass_s
        v_box_p = v_mass_p

        variables = self._get_standard_pressure_variables(p)
        variables.update(self._get_standard_velocity_variables(v_box))

        return variables

    def get_coupled_variables(self, variables):

        v_box_s, div_Vbox_s = self._separator_velocity(variables)

        variables.update(self._get_standard_vertical_velocity_variables(div_Vbox_s))

        return variables

    def set_algebraic(self, variables):
        p_n = variables["Negative electrode pressure"]
        p_s = variables["X-averaged separator pressure"]
        p_p = variables["Positive electrode pressure"]

        j_n = variables["Negative electrode interfacial current density"]
        j_p = variables["Positive electrode interfacial current density"]

        v_box_n = variables["Negative electrode volume-averaged velocity"]
        v_box_p = variables["Positive electrode volume-averaged velocity"]
        div_Vbox_s = variables["Transverse separator volume-averaged acceleration"]
        Vbox_s = variables["X-averaged transverse separator volume-averaged velocity"]

        # Problems in the x-direction for p_n and p_p
        self.algebraic = {
            p_n: pybamm.div(v_box_n) - self.param.beta_n * j_n,
            p_p: pybamm.div(v_box_p) - self.param.beta_p * j_p,
        }
        # Problem in the z-direction for p_s
        self.algebraic.update(
            self._get_separator_transient_velocity_algebraic(variables)
        )

    def set_boundary_conditions(self, variables):
        p_n = variables["Negative electrode pressure"]
        p_s = variables["X-averaged separator pressure"]
        p_p = variables["Positive electrode pressure"]

        # Boundary conditions in x-direction for p_n and p_p
        self.boundary_conditions = {
            p_n: {"left": (pybamm.Scalar(0), "Neumann"), "right": (p_s, "Dirichlet")},
            p_p: {"left": (p_s, "Dirichlet"), "right": (pybamm.Scalar(0), "Neumann")},
        }
        # Boundary conditions in z-direction for p_s (left=bottom, right=top)
        self.boundary_conditions[p_s] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Dirichlet"),
        }

    def set_initial_conditions(self, variables):
        p_n = variables["Negative electrode pressure"]
        p_s = variables["X-averaged separator pressure"]
        p_p = variables["Positive electrode pressure"]

        self.initial_conditions = {
            p_n: pybamm.Scalar(0),
            p_s: pybamm.Scalar(0),
            p_p: pybamm.Scalar(0),
        }


class FullVerticallyVarying(BaseFull):
    """
    Submodel for convection in the case where we allow variations in the transient
    directions
    """

    def __init__(self, param):
        super().__init__(self, param)

    def _get_separator_pressure(self):
        "Separator pressure as a variable"
        return pybamm.Variable("Separator pressure", domain="current collector")

    def _get_separator_transient_velocity(self, variables):
        "Separator velocity is the gradient of pressure"
        return -pybamm.grad(p_s)

    def _get_separator_transient_velocity_algebraic(self, variables):
        "Algebraic equation for the pressure / transient velocity in the separator"
        # div_Vbox_s is known in advance
        return {p_s: pybamm.div(Vbox_s) + div_Vbox_s}


class FullVerticallyUniform(BaseFull):
    """
    Submodel for convection in the case where everything is uniform in transient
    directions
    """

    def __init__(self, param):
        super().__init__(self, param)

    def _get_separator_pressure(self):
        "Constant separator pressure "
        return pybamm.PrimaryBroadcast(0, domain="current collector")

    def _get_separator_transient_velocity(self, variables):
        "Transient velocity in the separator is linear in z"
        return pybamm.IndefiniteIntegral(div_Vbox_s, z)

    def _get_separator_transient_velocity_algebraic(self, variables):
        "No algebraic equation for the pressure / transient velocity in the separator"
        return {}
