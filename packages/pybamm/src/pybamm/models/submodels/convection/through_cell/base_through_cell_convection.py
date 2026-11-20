#
# Base class for convection submodels in the through-cell direction
#
import pybamm
from pybamm.models.submodels.convection.base_convection import BaseModel


class BaseThroughCellModel(BaseModel):
    """Base class for convection submodels in the through-cell direction.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def _get_standard_convection_variables(self, domain, v_box_k, div_v_box_k, p_k):
        Domain = domain.capitalize()

        div_v_box_k_av = pybamm.x_average(div_v_box_k)

        variables = {
            f"{Domain} volume-averaged velocity [m.s-1]": v_box_k,
            f"{Domain} volume-averaged acceleration [m.s-2]": div_v_box_k,
            f"X-averaged {domain} volume-averaged acceleration [m.s-2]": div_v_box_k_av,
            f"{Domain} pressure [Pa]": p_k,
            f"X-averaged {domain} pressure [Pa]": pybamm.x_average(p_k),
        }
        return variables

    def _get_standard_sep_velocity_variables(self, v_box_s, div_v_box_s):
        """Volume-averaged velocity in the separator"""

        div_v_box_s_av = pybamm.x_average(div_v_box_s)

        variables = {
            "Separator volume-averaged velocity [m.s-1]": v_box_s,
            "Separator volume-averaged acceleration [m.s-2]": div_v_box_s,
            "X-averaged separator volume-averaged acceleration [m.s-2]": div_v_box_s_av,
        }

        return variables

    def _get_separator_and_whole_cell_variables(
        self, v_box_dict, div_v_box_dict, p_dict, v_box_s, div_v_box_s
    ):
        """
        Separator and whole-cell variables, given the electrode velocity,
        acceleration and pressure in each electrode and the separator velocity and
        acceleration. The separator pressure is set by the transverse submodel.
        """
        p_dict = {
            **p_dict,
            "separator": self.coupled_variable(
                "Separator pressure [Pa]",
                domain="separator",
                auxiliary_domains={"secondary": "current collector"},
            ),
        }
        v_box_dict = {**v_box_dict, "separator": v_box_s}
        div_v_box_dict = {**div_v_box_dict, "separator": div_v_box_s}
        variables = self._get_standard_sep_velocity_variables(v_box_s, div_v_box_s)
        variables.update(
            self._get_standard_whole_cell_variables(v_box_dict, div_v_box_dict, p_dict)
        )
        return variables

    def _get_separator_velocity(self):
        """
        Velocity and acceleration in the separator, set by the transverse
        acceleration and the velocity at the negative electrode boundary.
        """
        div_Vbox_s = self.coupled_variable(
            "X-averaged separator transverse volume-averaged acceleration [m.s-2]",
            domain="current collector",
        )
        i_boundary_cc = self.coupled_variable(
            "Current collector current density [A.m-2]", domain="current collector"
        )
        v_box_n_right = -self.param.n.DeltaV * i_boundary_cc / self.param.F
        div_v_box_s_av = -div_Vbox_s
        div_v_box_s = pybamm.PrimaryBroadcast(div_v_box_s_av, "separator")

        # Simple formula for velocity in the separator
        x_s = pybamm.standard_spatial_vars.x_s
        v_box_s = div_v_box_s_av * (x_s - self.param.n.L) + v_box_n_right
        return v_box_s, div_v_box_s
