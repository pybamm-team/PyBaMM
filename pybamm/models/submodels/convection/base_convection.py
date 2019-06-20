#
# Base class for convection
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Base class for convection

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def _get_standard_velocity_variables(self, v_box):

        vel_scale = self.param.velocity_scale

        # add more to this (x-averages etc)
        variables = {
            "Volume-averaged velocity": v_box,
            "Volume-averaged velocity [m.s-1]": vel_scale * v_box,
        }

        return variables

    def _get_standard_pressure_variables(self, p):

        # add more to this (x-averages etc)
        variables = {"Electrolyte pressure": p}

        return variables

    def _get_standard_vertical_velocity_variables(self, dVbox_dz):

        vel_scale = self.set_of_parameters.velocity_scale
        L_z = self.set_of_parameters.L_z

        variables = {
            "Vertical volume-averaged acceleration": dVbox_dz,
            "Vertical volume-averaged acceleration [m.s-2]": vel_scale / L_z * dVbox_dz,
        }

        return variables

    def _separator_velocity(self, variables):
        """
        Calculate x- and z-components of velocity in the separator

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model

        Returns
        -------
        v_box_s : :class:`pybamm.Symbol`
            The x-component of velocity in the separator
        dVbox_dz : :class:`pybamm.Symbol`
            The z-component of velocity in the separator
        """
        # Set up
        param = self.param
        l_n = pybamm.geometric_parameters.l_n
        l_s = pybamm.geometric_parameters.l_s
        x_s = pybamm.standard_spatial_vars.x_s

        # Difference in negative and positive electrode velocities determines the
        # velocity in the separator
        i_boundary_cc = variables["Current collector current density"]
        v_box_n_right = param.beta_n * i_boundary_cc
        v_box_p_left = param.beta_p * i_boundary_cc
        d_vbox_s__dx = (v_box_p_left - v_box_n_right) / l_s

        # Simple formula for velocity in the separator
        dVbox_dz = pybamm.Concatenation(
            pybamm.Broadcast(0, "negative electrode"),
            pybamm.Broadcast(-d_vbox_s__dx, "separator"),
            pybamm.Broadcast(0, "positive electrode"),
        )
        v_box_s = d_vbox_s__dx * (x_s - l_n) + v_box_n_right

        return v_box_s, dVbox_dz
