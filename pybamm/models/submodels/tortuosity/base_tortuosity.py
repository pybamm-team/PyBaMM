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
    domain : str
        The subdomain for the model ('Negative electrode', 'Negative electrolyte',
        'Separator electrolyte', 'Positive electrode' or 'Positive electrolyte'). Note
        that there is no 'Separator electrode' as the separator is insulating (no solid
        problem there).

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_standard_tortuosity_variables(self, tor):
        variables = {
            self.domain + " tortuosity": tor,
            "X-averaged " + self.domain.lower() + " tortuosity": pybamm.x_average(tor),
        }

        return variables
