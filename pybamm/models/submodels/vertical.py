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

    def set_algebraic_vertical_current(self, bc_variables):
        """ Set the system that gives the current in the current collectors.

        Parameters
        ----------
        bc_variables : dict of :class:`pybamm.Symbol`
            Dictionary of variables in the current collector
        """
        param = self.set_of_parameters
        i_cell = param.current_with_time

        # Set up variables
        c_e = bc_variables["c_e"]
        delta_phi_n = bc_variables["delta_phi_n"]
        delta_phi_p = bc_variables["delta_phi_p"]
        i_curr_coll = pybamm.Variable(
            "Current collector current", domain="current collector"
        )
        electrolyte_terms = delta_phi_n - delta_phi_p

        # Algebraic equation for the current in the current collector
        I_s_perp = self.vertical_conductivity * pybamm.grad(
            i_curr_coll - electrolyte_terms
        )
        self.algebraic[i_curr_coll] = pybamm.div(I_s_perp) - i_curr_coll
        self.initial_conditions[i_curr_coll] = i_cell

        # Set boundary conditions at top ("right") and bottom ("left")
        self.boundary_conditions[i_curr_coll] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (i_cell / self.vertical_conductivity, "Neumann"),
        }
        # Trivial boundary conditions for the electrolyte terms
        self.boundary_conditions[electrolyte_terms] = {
            "left": (pybamm.BoundaryFlux(electrolyte_terms, "left"), "Neumann"),
            "right": (pybamm.BoundaryFlux(electrolyte_terms, "right"), "Neumann"),
        }

        self.variables = {"Current collector current": i_curr_coll}

    def set_leading_order_vertical_current(self, bc_variables):
        """ Set the system that gives the leading-order current in the current
        collectors.

        Parameters
        ----------
        bc_variables : dict of :class:`pybamm.Symbol`
            Dictionary of variables in the current collector
        """
        param = self.set_of_parameters
        delta_phi_n = bc_variables["delta_phi_n"]
        delta_phi_p = bc_variables["delta_phi_p"]
        delta_phi_difference = delta_phi_n - delta_phi_p

        # Simple model: read off vertical current (no extra equation)
        I_s_perp = self.vertical_conductivity * pybamm.grad(delta_phi_difference)
        i_current_collector = pybamm.div(I_s_perp)

        # Set boundary conditions at top ("right") and bottom ("left")
        i_cell = param.current_with_time
        self.boundary_conditions = {
            delta_phi_difference: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (i_cell / self.vertical_conductivity, "Neumann"),
            }
        }
        self.variables = {"Current collector current": i_current_collector}
