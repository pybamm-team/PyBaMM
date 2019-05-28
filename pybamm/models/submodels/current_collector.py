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

    def set_algebraic_system(self, v_local, i_local):
        """
        PDE system for current in the current collectors, using Ohm's law

        Parameters
        ----------
        v_local : :class:`pybamm.Variable`
            Local cell voltage
        i_local : :class:`pybamm.Variable`
            Local through-cell current density

        """
        param = self.set_of_parameters
        y = pybamm.standard_spatial_vars.y
        z = pybamm.standard_spatial_vars.z

        # algebraic equations
        applied_current = param.current_with_time
        self.algebraic = {
            v_local: pybamm.laplacian(v_local) + param.alpha * pybamm.source(i_local),
            i_local: pybamm.Integral(i_local, [y, z]) - applied_current,
        }
        self.initial_conditions = {
            v_local: param.U_p(param.c_p_init) - param.U_n(param.c_n_init),
            i_local: applied_current / param.l_y / param.l_z,
        }
        # left for negative tab, right for positive tab
        neg_tab_bc = -applied_current / (
            param.sigma_cn * (param.L_x / param.L_z) ** 2 * param.l_tab_n * param.l_cn
        )
        pos_tab_bc = -applied_current / (
            param.sigma_cp * (param.L_x / param.L_z) ** 2 * param.l_tab_p * param.l_cp
        )
        self.boundary_conditions = {
            v_local: {"left": (neg_tab_bc, "Neumann"), "right": (pos_tab_bc, "Neumann")}
        }
        self.variables = {
            "Local cell voltage": v_local,
            "Local through-cell current density": i_local,
        }
