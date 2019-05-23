#
# Vertical models
#
# NOTE: this can eventually be merged with the current collector
import pybamm


class Vertical(pybamm.SubModel):
    """
    Vertical submodel

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)
        param = set_of_parameters

        # Define conductivity
        scaled_cond_n = param.l_n * param.sigma_n * param.delta ** 2
        scaled_cond_p = param.l_p * param.sigma_p * param.delta ** 2
        self.vertical_conductivity = (scaled_cond_n * scaled_cond_p) / (
            scaled_cond_n + scaled_cond_p
        )

    def set_leading_order_vertical_current(self, delta_phi_n, delta_phi_p):
        param = self.set_of_parameters
        delta_phi_difference = pybamm.Variable(
            "delta_phi_difference", domain="current collector"
        )

        # Simple model: read off vertical current (no extra equation)
        delta_phi_n_right = pybamm.boundary_value(delta_phi_n, "right")
        delta_phi_p_left = pybamm.boundary_value(delta_phi_p, "left")
        I_s_perp = self.vertical_conductivity * pybamm.grad(delta_phi_difference)
        i_curr_coll = pybamm.div(I_s_perp)
        self.algebraic = {
            delta_phi_difference: delta_phi_difference
            - (delta_phi_n_right - delta_phi_p_left)
        }

        # Set initial conditions
        delta_phi_n_right_init = param.U_n(param.c_e_init)
        delta_phi_p_left_init = param.U_p(param.c_e_init)
        self.initial_conditions = {
            delta_phi_difference: (delta_phi_n_right_init - delta_phi_p_left_init)
        }

        # Set boundary conditions at top ("right") and bottom ("left")
        i_cell = param.current_with_time
        self.boundary_conditions = {
            delta_phi_difference: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (i_cell / self.vertical_conductivity, "Neumann"),
            }
        }
        self.variables = {"Current collector current": i_curr_coll}
