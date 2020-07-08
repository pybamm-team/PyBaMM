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
