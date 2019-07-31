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

    def _get_standard_velocity_variables(self, v_box):
        """
        A private function to obtain the standard variables which
        can be derived from the fluid velocity.

        Parameters
        ----------
        v_box : :class:`pybamm.Symbol`
            The volume-averaged fluid velocity

        Returns
        -------
        variables : dict
            The variables which can be derived from the volume-averaged
            velocity.
        """

        vel_scale = self.param.velocity_scale

        v_box_av = pybamm.x_average(v_box)
        # add more to this (x-averages etc)
        variables = {
            "Volume-averaged velocity": v_box,
            "Volume-averaged velocity [m.s-1]": vel_scale * v_box,
            # "X-averaged volume-averaged velocity": v_box_av,
            # "X-averaged volume-averaged velocity [m.s-1]": vel_scale * v_box_av,
        }

        return variables

    def _get_standard_pressure_variables(self, p):
        """
        A private function to obtain the standard variables which
        can be derived from the pressure in the fluid.

        Parameters
        ----------
        p : :class:`pybamm.Symbol`
            The fluid pressure

        Returns
        -------
        variables : dict
            The variables which can be derived from the pressure.
        """

        p_n, p_s, p_p = p.orphans

        variables = {
            "Negative electrode pressure": p_n,
            "Separator pressure": p_n,
            "Positive electrode pressure": p_n,
            "Pressure": p,
            "X-averaged pressure": pybamm.x_average(p),
            "X-averaged negative electrode pressure": pybamm.x_average(p_n),
            "X-averaged separator pressure": pybamm.x_average(p_n),
            "X-averaged positive electrode pressure": pybamm.x_average(p_n),
        }

        return variables

    def _get_standard_vertical_velocity_variables(self, div_Vbox_s):
        """
        A private function to obtain the standard variables which
        can be derived from the vertical velocity of the fluid.

        Parameters
        ----------
        div_Vbox_s : :class:`pybamm.Symbol`
            The vertical acceleration of the fluid

        Returns
        -------
        variables : dict
            The variables which can be derived from the vertical velocity.
        """
        vel_scale = self.param.velocity_scale
        L_z = self.param.L_z

        variables = {
            "Vertical volume-averaged acceleration": div_Vbox_s,
            "Vertical volume-averaged acceleration [m.s-2]": vel_scale
            / L_z
            * div_Vbox_s,
        }

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
        v_box_s = pybamm.outer(d_vbox_s__dx, (x_s - l_n)) + pybamm.PrimaryBroadcast(
            v_box_n_right, "separator"
        )

        return v_box_s, dVbox_dz
