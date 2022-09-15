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

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, component, options=None):
        super().__init__(param, options=options)
        self.component = component

    def _get_standard_transport_efficiency_variables(self, tor_dict):
        component = self.component.lower()

        tor = pybamm.concatenation(*tor_dict.values())

        variables = {f"{self.component} transport efficiency": tor}

        for domain, tor_k in tor_dict.items():
            if not (domain == "separator" and self.component == "Electrode"):
                Domain = domain.split()[0]
                domain = Domain.lower()

                variables.update(
                    {
                        f"{Domain} {component} transport efficiency": tor_k,
                        f"X-averaged {domain} {component} "
                        "transport efficiency": pybamm.x_average(tor_k),
                    }
                )

        if self.set_leading_order is True:
            leading_order_variables = {
                "Leading-order " + name.lower(): var for name, var in variables.items()
            }
            variables.update(leading_order_variables)

        # Override print_name
        tor.print_name = r"\epsilon^{b_e}"

        return variables
