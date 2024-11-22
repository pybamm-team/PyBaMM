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
    """

    def __init__(self, param, domain, options):
        super().__init__(param, domain, options=options)

    def build(self):
        domain, Domain = self.domain_Domain
        phases = self.options.phases[domain]
        zero = pybamm.Scalar(0)
        for variable_template in [
            f"{Domain} electrode {{}}active material volume fraction",
            f"X-averaged {domain} electrode {{}}active material volume fraction",
            f"{Domain} electrode {{}}active material volume fraction change [s-1]",
            f"X-averaged {domain} electrode {{}}active material "
            "volume fraction change [s-1]",
            f"Loss of lithium due to loss of {{}}active material in {domain} electrode [mol]",
        ]:
            sumvar = zero
            for phase in phases:
                # It would be nice if we had a systematic way to set the domain for these things.
                if "X-averaged" in variable_template:
                    var = pybamm.CoupledVariable(
                        variable_template.format(phase + " "),
                        domain="current collector",
                    )
                elif "of lithium" in variable_template:
                    var = pybamm.CoupledVariable(variable_template.format(phase + " "))
                else:
                    var = pybamm.CoupledVariable(
                        variable_template.format(phase + " "),
                        domain=f"{domain} electrode",
                        auxiliary_domains={"secondary": "current collector"},
                    )
                self.coupled_variables.update({var.name: var})
                sumvar += var
            self.variables.update({variable_template.format(""): sumvar})

        if self.options["particle shape"] != "no particles":
            C = zero
            for phase in phases:
                var = pybamm.CoupledVariable(
                    f"{Domain} electrode {phase} phase capacity [A.h]"
                )
                self.coupled_variables.update({var.name: var})
                C += var
            self.variables.update({f"{Domain} electrode capacity [A.h]": C})
