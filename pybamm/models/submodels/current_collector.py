#
# Equation classes for the current collector
#
import pybamm

import importlib

dolfin_spec = importlib.util.find_spec("dolfin")
if dolfin_spec is not None:
    dolfin = importlib.util.module_from_spec(dolfin_spec)
    dolfin_spec.loader.exec_module(dolfin)


class OhmTwoDimensional(pybamm.SubModel):
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
        i_boundary_cc = bc_variables["Current collector current density"]
        y = pybamm.standard_spatial_vars.y
        z = pybamm.standard_spatial_vars.z

        v_boundary_cc = pybamm.Variable(
            "Current collector voltage", domain="current collector"
        )

        # get average ocv and reaction overpotentials
        ocv_av = bc_variables["Average open circuit voltage"]
        eta_r_av = bc_variables["Average reaction overpotential"]

        # Poisson problem in the current collector with SPM current-voltage relation
        # We add a dummy variable to account for the constraint that the through-cell
        # current must integrate over the current collector domain to give the applied
        # current.
        applied_current = param.current_with_time
        constraint_var = pybamm.Variable("Current conservation constraint", domain=[])
        self.algebraic = {
            v_boundary_cc: pybamm.laplacian(v_boundary_cc)
            + param.alpha * pybamm.source(i_boundary_cc, v_boundary_cc),
            i_boundary_cc: v_boundary_cc - (ocv_av - eta_r_av),
            constraint_var: pybamm.Integral(i_boundary_cc, [y, z]) - applied_current,
        }
        self.initial_conditions = {
            v_boundary_cc: param.U_p(param.c_p_init) - param.U_n(param.c_n_init),
            i_boundary_cc: applied_current / param.l_y / param.l_z,
            constraint_var: 0,
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
        self.variables = {
            "Current collector voltage": v_boundary_cc,
            "Current conservation constraint": constraint_var,
        }
