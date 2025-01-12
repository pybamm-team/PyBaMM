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
        Either 'negative' or 'positive'
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, domain, options):
        super().__init__(param, domain, options=options)

    def _get_standard_interface_utilisation_variables(self, u_var):
        domain, Domain = self.domain_Domain

        u = pybamm.maximum(u_var, 1e-8)
        u_var_av = pybamm.x_average(u_var)
        u_av = pybamm.maximum(u_var_av, 1e-8)
        if self.options.electrode_types[self.domain] == "planar":
            variables = {
                "Lithium metal interface utilisation variable": u_var_av,
                "Lithium metal interface utilisation": u_av,
            }
        else:
            variables = {
                f"{Domain} electrode interface utilisation variable": u_var,
                f"X-averaged {domain} electrode interface utilisation "
                "variable": u_var_av,
                f"{Domain} electrode interface utilisation": u,
                f"X-averaged {domain} electrode interface utilisation": u_av,
            }

        return variables
