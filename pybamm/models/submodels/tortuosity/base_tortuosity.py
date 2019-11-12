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

    def _get_standard_tortuosity_variables(self, tor):
        tor_n, tor_s, tor_p = tor.orphans

        variables = {
            self.phase + " tortuosity": tor,
            "Negative " + self.phase.lower() + " tortuosity": tor_n,
            "Separator " + self.phase.lower() + " tortuosity": tor_s,
            "Positive " + self.phase.lower() + " tortuosity": tor_p,
            "X-averaged negative "
            + self.phase.lower()
            + " tortuosity": pybamm.x_average(tor_n),
            "X-averaged separator "
            + self.phase.lower()
            + " tortuosity": pybamm.x_average(tor_s),
            "X-averaged positive "
            + self.phase.lower()
            + " tortuosity": pybamm.x_average(tor_p),
        }

        return variables
