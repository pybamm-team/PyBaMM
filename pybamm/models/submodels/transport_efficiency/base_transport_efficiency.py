#
# Base class for transport_efficiency
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Base class for transport_efficiency

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    component : str
        The material for the model ('electrolyte' or 'electrode').
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, component, options=None):
        super().__init__(param, options=options)
        self.component = component

    def _get_standard_transport_efficiency_variables(self, tor_dict):
        component = self.component.lower()

        tor = pybamm.concatenation(*tor_dict.values())

        variables = {f"{self.component} transport efficiency": tor}

        for domain, tor_k in tor_dict.items():
            domain = domain.split()[0]
            Domain = domain.capitalize()
            tor_k_av = pybamm.x_average(tor_k)

            variables.update(
                {
                    f"{Domain} {component} transport efficiency": tor_k,
                    f"X-averaged {domain} {component} transport efficiency": tor_k_av,
                }
            )

        # Override print_name
        tor.print_name = r"\epsilon^{b_e}"

        return variables
