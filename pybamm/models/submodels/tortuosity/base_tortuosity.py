#
# Base class for tortuosity
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Base class for tortuosity

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    phase : str
        The material for the model ('electrolyte' or 'electrode').

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, phase):
        super().__init__(param)
        self.phase = phase

    def _get_standard_tortuosity_variables(self, tor, set_leading_order=False):
        tor_n, tor_s, tor_p = tor.orphans

        variables = {
            self.phase + " tortuosity": tor,
            "Negative " + self.phase.lower() + " tortuosity": tor_n,
            "Positive " + self.phase.lower() + " tortuosity": tor_p,
            "X-averaged negative "
            + self.phase.lower()
            + " tortuosity": pybamm.x_average(tor_n),
            "X-averaged positive "
            + self.phase.lower()
            + " tortuosity": pybamm.x_average(tor_p),
        }
        if self.phase == "Electrolyte":
            variables.update(
                {
                    "Separator tortuosity": tor_s,
                    "X-averaged separator tortuosity": pybamm.x_average(tor_s),
                }
            )

        if set_leading_order is True:
            leading_order_variables = {
                "Leading-order " + name.lower(): var for name, var in variables.items()
            }
            variables.update(leading_order_variables)

        return variables
