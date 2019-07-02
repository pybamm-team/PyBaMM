#
# Equation classes for the current collector
#
import pybamm


class OldOhmTwoDimensional(pybamm.OldSubModel):
    """
    Ohm's law + conservation of current for the current in the current collectors.

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_uniform_current(self, bc_variables):
        """
        PDE system for current in the current collectors, using Ohm's law

        Parameters
        ----------
        bc_variables : dict of :class:`pybamm.Symbol`
            Dictionary of variables in the current collector
        """
        param = self.set_of_parameters
        i_boundary_cc = bc_variables["Current collector current density"]

        # set uniform current density (can be useful for testing)
        applied_current = param.current_with_time
        self.algebraic = {
            i_boundary_cc: i_boundary_cc - applied_current / param.l_y / param.l_z
        }
        self.initial_conditions = {
            i_boundary_cc: applied_current / param.l_y / param.l_z
        }

    def set_algebraic_system_spm(self, bc_variables):
        """
        PDE system for current in the current collectors, using Ohm's law

        Parameters
        ----------
        bc_variables : dict of :class:`pybamm.Symbol`
            Dictionary of variables in the current collector

        """
        param = self.set_of_parameters

        # variables
        v_boundary_cc = pybamm.Variable(
            "Current collector voltage", domain="current collector"
        )
        i_boundary_cc = bc_variables["Current collector current density"]

        # get average ocv and reaction overpotentials
        ocv_av = bc_variables["Average open circuit voltage"]
        eta_r_av = bc_variables["Average reaction overpotential"]

        # Poisson problem in the current collector with SPM current-voltage relation.
        applied_current = param.current_with_time
        self.algebraic = {
            v_boundary_cc: pybamm.laplacian(v_boundary_cc)
            + param.alpha * pybamm.source(i_boundary_cc, v_boundary_cc),
            i_boundary_cc: v_boundary_cc - (ocv_av - eta_r_av),
        }
        self.initial_conditions = {
            v_boundary_cc: param.U_p(param.c_p_init) - param.U_n(param.c_n_init),
            i_boundary_cc: applied_current / param.l_y / param.l_z,
        }

        # Set boundary conditions at positive tab ("right") and negative tab ("left")
        neg_tab_bc = -applied_current / (
            param.sigma_cn * (param.L_x / param.L_z) ** 2 * param.l_tab_n * param.l_cn
        )
        pos_tab_bc = -applied_current / (
            param.sigma_cp * (param.L_x / param.L_z) ** 2 * param.l_tab_p * param.l_cp
        )
        self.boundary_conditions = {
            v_boundary_cc: {
                "left": (neg_tab_bc, "Neumann"),
                "right": (pos_tab_bc, "Neumann"),
            }
        }
        self.variables = {"Current collector voltage": v_boundary_cc}
