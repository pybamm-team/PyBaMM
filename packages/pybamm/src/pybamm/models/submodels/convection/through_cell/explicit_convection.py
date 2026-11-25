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
    """

    def __init__(self, param):
        super().__init__(param)

    def build(self):
        p_s = self.coupled_variable(
            "X-averaged separator pressure [Pa]", domain="current collector"
        )
        v_box_dict, div_v_box_dict, p_dict = {}, {}, {}
        for domain in self.options.whole_cell_domains:
            if domain == "separator":
                continue
            a_j_k_av = self.coupled_variable(
                f"X-averaged {domain} volumetric interfacial current density [A.m-3]",
                domain="current collector",
            )
            if domain == "negative electrode":
                x_n = pybamm.standard_spatial_vars.x_n
                DeltaV_k = self.param.n.DeltaV
                p_k = (
                    -DeltaV_k
                    * a_j_k_av
                    / self.param.F
                    * (-(x_n**2) + self.param.n.L**2)
                    / 2
                    + p_s
                )
                v_box_k = -DeltaV_k * a_j_k_av / self.param.F * x_n
            elif domain == "positive electrode":
                x_p = pybamm.standard_spatial_vars.x_p
                DeltaV_k = self.param.p.DeltaV
                p_k = (
                    -DeltaV_k
                    * a_j_k_av
                    / self.param.F
                    * ((x_p - 1) ** 2 - self.param.p.L**2)
                    / 2
                    + p_s
                )
                v_box_k = -DeltaV_k * a_j_k_av / self.param.F * (x_p - self.param.L_x)
            div_v_box_k = pybamm.PrimaryBroadcast(
                -DeltaV_k * a_j_k_av / self.param.F, domain
            )
            v_box_dict[domain], div_v_box_dict[domain], p_dict[domain] = (
                v_box_k,
                div_v_box_k,
                p_k,
            )
            self.variables.update(
                self._get_standard_convection_variables(
                    domain, v_box_k, div_v_box_k, p_k
                )
            )

        v_box_s, div_v_box_s = self._get_separator_velocity()
        self.variables.update(
            self._get_separator_and_whole_cell_variables(
                v_box_dict, div_v_box_dict, p_dict, v_box_s, div_v_box_s
            )
        )
