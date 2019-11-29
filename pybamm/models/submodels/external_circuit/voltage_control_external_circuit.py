#
# External circuit with voltage control
#
import pybamm
from .base_external_circuit import BaseModel


class VoltageControl(BaseModel):
    """
    External circuit with voltage control, implemented directly.
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        # Voltage is given as a function of time
        V_dim = pybamm.FunctionParameter("Voltage function", pybamm.t)

        param = self.param
        V = (V_dim - (param.U_p_ref - param.U_n_ref)) / param.potential_scale

        variables = {"Terminal voltage [V]": V_dim, "Terminal voltage": V}

        return variables

    def get_coupled_variables(self, variables):
        # Update currrent
        phi_s_p = variables["Positive electrode potential"]
        tor = variables["Positive electrode tortuosity"]

        param = self.param
        i_boundary_cc = (
            -param.sigma_p
            * pybamm.BoundaryValue(tor, "right")
            * pybamm.BoundaryGradient(phi_s_p, "right")
        )
        i_cell = pybamm.BoundaryValue(i_boundary_cc, "right")
        I = i_cell * abs(param.I_typ)
        i_cell_dim = I / (param.n_electrodes_parallel * param.A_cc)

        variables = {
            "Total current density": i_cell,
            "Total current density [A.m-2]": i_cell_dim,
            "Current [A]": I,
        }

        return variables
