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

    def set_potential_pair_spm(self, bc_variables):
        """
        PDE system for current in the current collectors, using Ohm's law and
        the voltage-current relationship for the SPM(e).

        Parameters
        ----------
        bc_variables : dict of :class:`pybamm.Symbol`
            Dictionary of variables in the current collector

        """
        param = self.set_of_parameters

        # variables
        phi_s_cn = pybamm.Variable(
            "Negative current collector potential", domain="current collector"
        )
        phi_s_cp = pybamm.Variable(
            "Positive current collector potential", domain="current collector"
        )
        i_boundary_cc = bc_variables["Current collector current density"]

        # get local voltage expression for SPM(e)
        ocv_av = bc_variables["Average open circuit voltage"]
        eta_r_av = bc_variables["Average reaction overpotential"]
        # NOTE: eta_e_av = eta_c_av + delta_phi_e_av
        eta_e_av = bc_variables["Average electrolyte overpotential"]
        delta_phi_s_av = bc_variables["Average solid phase ohmic losses"]
        local_voltage_expression = ocv_av + eta_r_av + eta_e_av + delta_phi_s_av

        # Poisson problem in the current collectors with SPM current-voltage relation.
        v_boundary_cc = phi_s_cp - phi_s_cn
        applied_current = param.current_with_time
        self.algebraic = {
            phi_s_cn: pybamm.laplacian(phi_s_cn)
            - (param.sigma_cn * param.delta ** 2 / param.l_cn)
            * pybamm.source(i_boundary_cc, phi_s_cn),
            phi_s_cp: pybamm.laplacian(phi_s_cp)
            + (param.sigma_cp * param.delta ** 2 / param.l_cp)
            * pybamm.source(i_boundary_cc, phi_s_cp),
            i_boundary_cc: v_boundary_cc - local_voltage_expression,
        }
        self.initial_conditions = {
            phi_s_cn: pybamm.Scalar(0),
            phi_s_cp: param.U_p(param.c_p_init) - param.U_n(param.c_n_init),
            i_boundary_cc: applied_current / param.l_y / param.l_z,
        }

        # Set boundary conditions at positive tab ("right") and negative tab ("left")
        pos_tab_bc = -applied_current / (
            param.sigma_cp * param.delta ** 2 * param.l_tab_p * param.l_cp
        )
        self.boundary_conditions = {
            phi_s_cn: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            },
            phi_s_cp: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pos_tab_bc, "Neumann"),
            },
        }
        self.variables = {
            "Current collector voltage": v_boundary_cc,
            "Negative current collector potential": phi_s_cn,
            "Positive current collector potential": phi_s_cp,
        }

    def set_potential_difference_spm(self, bc_variables):
        """
        PDE system for current in the current collectors, using Ohm's law and
        the voltage-current relationship for the SPM(e). Note this only solves
        for the local potential difference, NOT the potentials on each current
        collector.

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

        # get local voltage expression for SPM(e)
        ocv_av = bc_variables["Average open circuit voltage"]
        eta_r_av = bc_variables["Average reaction overpotential"]
        # NOTE: eta_e_av = eta_c_av + delta_phi_e_av
        eta_e_av = bc_variables["Average electrolyte overpotential"]
        delta_phi_s_av = bc_variables["Average solid phase ohmic losses"]
        local_voltage_expression = ocv_av + eta_r_av + eta_e_av + delta_phi_s_av
        # Poisson problem in the current collector with SPM current-voltage relation.
        applied_current = param.current_with_time
        self.algebraic = {
            v_boundary_cc: pybamm.laplacian(v_boundary_cc)
            + param.alpha * pybamm.source(i_boundary_cc, v_boundary_cc),
            i_boundary_cc: v_boundary_cc - local_voltage_expression,
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
