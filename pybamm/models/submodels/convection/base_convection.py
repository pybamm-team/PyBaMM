#
# Base class for convection submodels
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Base class for convection submodels.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def _get_standard_whole_cell_velocity_variables(self, variables):
        """
        A private function to obtain the standard variables which
        can be derived from the fluid velocity.

        Parameters
        ----------
        variables : dict
            The existing variables in the model

        Returns
        -------
        variables : dict
            The variables which can be derived from the volume-averaged
            velocity.
        """

        vel_scale = self.param.velocity_scale

        v_box_n = variables["Negative electrode volume-averaged velocity"]
        v_box_s = variables["Separator volume-averaged velocity"]
        v_box_p = variables["Positive electrode volume-averaged velocity"]

        v_box = pybamm.Concatenation(v_box_n, v_box_s, v_box_p)

        variables = {
            "Volume-averaged velocity": v_box,
            "Volume-averaged velocity [m.s-1]": vel_scale * v_box,
        }

        return variables

    def _get_standard_whole_cell_acceleration_variables(self, variables):
        """
        A private function to obtain the standard variables which
        can be derived from the fluid velocity.

        Parameters
        ----------
        variables : dict
            The existing variables in the model

        Returns
        -------
        variables : dict
            The variables which can be derived from the volume-averaged
            velocity.
        """

        acc_scale = self.param.velocity_scale / self.param.L_x

        div_v_box_n = variables["Negative electrode volume-averaged acceleration"]
        div_v_box_s = variables["Separator volume-averaged acceleration"]
        div_v_box_p = variables["Positive electrode volume-averaged acceleration"]

        div_v_box = pybamm.Concatenation(div_v_box_n, div_v_box_s, div_v_box_p)
        div_v_box_av = pybamm.x_average(div_v_box)

        variables = {
            "Volume-averaged acceleration": div_v_box,
            "X-averaged volume-averaged acceleration": div_v_box_av,
            "Volume-averaged acceleration [m.s-1]": acc_scale * div_v_box,
            "X-averaged volume-averaged acceleration [m.s-1]": acc_scale * div_v_box_av,
        }

        return variables

    def _get_standard_whole_cell_pressure_variables(self, variables):
        """
        A private function to obtain the standard variables which
        can be derived from the pressure in the fluid.

        Parameters
        ----------
        variables : dict
            The existing variables in the model

        Returns
        -------
        variables : dict
            The variables which can be derived from the pressure.
        """
        p_n = variables["Negative electrode pressure"]
        p_s = variables["Separator pressure"]
        p_p = variables["Positive electrode pressure"]

        p = pybamm.Concatenation(p_n, p_s, p_p)

        variables = {"Pressure": p}

        return variables

    def _separator_velocity(self, variables):
        """
        A private method to calculate x- and z-components of velocity in the separator
        Parameters
        ----------
        variables : dict
            Dictionary of variables in the whole model.

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
            pybamm.FullBroadcast(
                0,
                "negative electrode",
                auxiliary_domains={"secondary": "current collector"},
            ),
            pybamm.PrimaryBroadcast(-d_vbox_s__dx, "separator"),
            pybamm.FullBroadcast(
                0,
                "positive electrode",
                auxiliary_domains={"secondary": "current collector"},
            ),
        )
        v_box_s = d_vbox_s__dx * (x_s - l_n) + v_box_n_right

        return v_box_s, dVbox_dz
