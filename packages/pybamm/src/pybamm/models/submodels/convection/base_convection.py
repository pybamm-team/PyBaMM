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
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def _get_standard_whole_cell_variables(self, v_box_dict, div_v_box_dict, p_dict):
        """
        Whole-cell velocity, acceleration and pressure variables.

        Parameters
        ----------
        v_box_dict : dict
            Volume-averaged velocity in each domain of the cell.
        div_v_box_dict : dict
            Volume-averaged acceleration in each domain of the cell.
        p_dict : dict
            Pressure in each domain of the cell.

        Returns
        -------
        variables : dict
            The whole-cell variables.
        """
        domains = self.options.whole_cell_domains
        v_box = pybamm.concatenation(*[v_box_dict[domain] for domain in domains])
        div_v_box = pybamm.concatenation(
            *[div_v_box_dict[domain] for domain in domains]
        )
        p = pybamm.concatenation(*[p_dict[domain] for domain in domains])
        return {
            "Volume-averaged velocity [m.s-1]": v_box,
            "Volume-averaged acceleration [m.s-1]": div_v_box,
            "X-averaged volume-averaged acceleration [m.s-1]": pybamm.x_average(
                div_v_box
            ),
            "Pressure [Pa]": p,
        }
