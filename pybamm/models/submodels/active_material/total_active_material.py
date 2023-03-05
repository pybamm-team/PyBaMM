#
# Class for total active material volume fraction, for models with multiple phases
#
import pybamm


class Total(pybamm.BaseSubModel):
    """Class for total active material volume fraction, for models with multiple phases

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    options : dict
        Additional options to pass to the model

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain, options):
        super().__init__(param, domain, options=options)

    def get_coupled_variables(self, variables):
        # Creates "total" active material volume fraction and capacity variables
        # by summing up all the phases
        domain, Domain = self.domain_Domain

        phases = self.options.phases[domain]
        # For each of the variables, the variable name without the phase name
        # is constructed by summing all of the variable names with the phases
        for variable_template in [
            f"{Domain} electrode {{}}active material volume fraction",
            f"X-averaged {domain} electrode {{}}active material volume fraction",
            f"{Domain} electrode {{}}active material volume fraction change [s-1]",
            f"X-averaged {domain} electrode {{}}active material "
            "volume fraction change [s-1]",
        ]:
            sumvar = sum(
                variables[variable_template.format(phase + " ")] for phase in phases
            )
            variables[variable_template.format("")] = sumvar

        if self.options["particle shape"] != "no particles":
            # capacity doesn't fit the template so needs to be done separately
            C = sum(
                variables[f"{Domain} electrode {phase} phase capacity [A.h]"]
                for phase in phases
            )
            variables.update({f"{Domain} electrode capacity [A.h]": C})

        return variables
