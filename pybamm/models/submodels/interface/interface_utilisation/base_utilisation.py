#
# Base class for interface utilisation
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Base class for interface utilisation

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'Negative' or 'Positive'
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain, options):
        super().__init__(param, domain, options=options)

    def _get_standard_interface_utilisation_variables(self, u):
        u_av = pybamm.x_average(u)
        if self.half_cell and self.domain == "Negative":
            variables = {"Lithium metal interface utilisation": u_av}
        else:
            variables = {
                self.domain + " electrode interface utilisation": u,
                "X-averaged "
                + self.domain.lower()
                + " electrode interface utilisation": u_av,
            }

        return variables
