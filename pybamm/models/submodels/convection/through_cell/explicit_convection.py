#
# Class for leading-order pressure driven convection
#
import pybamm
from .base_through_cell_convection import BaseThroughCellModel


class Explicit(BaseThroughCellModel):
    """A submodel for the leading-order approximation of pressure-driven convection

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.convection.through_cell.BaseThroughCellModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_coupled_variables(self, variables):
        # Set up
        param = self.param
        p_s = variables["X-averaged separator pressure [Pa]"]
        for domain in self.options.whole_cell_domains:
            if domain == "separator":
                continue
            a_j_k_av = variables[
                f"X-averaged {domain} volumetric interfacial current density [A.m-3]"
            ]
            if domain == "negative electrode":
                x_n = pybamm.standard_spatial_vars.x_n
                DeltaV_k = param.n.DeltaV
                p_k = (
                    -DeltaV_k * a_j_k_av / param.F * (-(x_n**2) + param.n.L**2) / 2
                    + p_s
                )
                v_box_k = -DeltaV_k * a_j_k_av / param.F * x_n
            elif domain == "positive electrode":
                x_p = pybamm.standard_spatial_vars.x_p
                DeltaV_k = param.p.DeltaV
                p_k = (
                    -DeltaV_k
                    * a_j_k_av
                    / param.F
                    * ((x_p - 1) ** 2 - param.p.L**2)
                    / 2
                    + p_s
                )
                v_box_k = -DeltaV_k * a_j_k_av / param.F * (x_p - param.L_x)
            div_v_box_k = pybamm.PrimaryBroadcast(
                -DeltaV_k * a_j_k_av / param.F, domain
            )

            variables.update(
                self._get_standard_convection_variables(
                    domain, v_box_k, div_v_box_k, p_k
                )
            )

        # Transverse velocity in the separator determines through-cell velocity
        div_Vbox_s = variables[
            "X-averaged separator transverse volume-averaged acceleration [m.s-2]"
        ]
        i_boundary_cc = variables["Current collector current density [A.m-2]"]
        v_box_n_right = -param.n.DeltaV * i_boundary_cc / param.F
        div_v_box_s_av = -div_Vbox_s
        div_v_box_s = pybamm.PrimaryBroadcast(div_v_box_s_av, "separator")

        # Simple formula for velocity in the separator
        x_s = pybamm.standard_spatial_vars.x_s
        v_box_s = div_v_box_s_av * (x_s - param.n.L) + v_box_n_right

        variables.update(
            self._get_standard_sep_velocity_variables(v_box_s, div_v_box_s)
        )
        variables.update(self._get_standard_whole_cell_velocity_variables(variables))
        variables.update(
            self._get_standard_whole_cell_acceleration_variables(variables)
        )
        variables.update(self._get_standard_whole_cell_pressure_variables(variables))

        return variables
