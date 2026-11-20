#
# No convection
#
import pybamm

from .base_through_cell_convection import BaseThroughCellModel


class NoConvection(BaseThroughCellModel):
    """A submodel for case where there is no convection.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def build(self):
        v_box_dict, div_v_box_dict, p_dict = {}, {}, {}
        for domain in self.options.whole_cell_domains:
            if domain == "separator":
                continue
            v_box_dict[domain] = pybamm.FullBroadcast(0, domain, "current collector")
            div_v_box_dict[domain] = pybamm.FullBroadcast(
                0, domain, "current collector"
            )
            p_dict[domain] = pybamm.FullBroadcast(0, domain, "current collector")
            self.variables.update(
                self._get_standard_convection_variables(
                    domain, v_box_dict[domain], div_v_box_dict[domain], p_dict[domain]
                )
            )

        v_box_s = pybamm.FullBroadcast(0, "separator", "current collector")
        div_v_box_s = pybamm.FullBroadcast(0, "separator", "current collector")
        self.variables.update(
            self._get_separator_and_whole_cell_variables(
                v_box_dict, div_v_box_dict, p_dict, v_box_s, div_v_box_s
            )
        )
