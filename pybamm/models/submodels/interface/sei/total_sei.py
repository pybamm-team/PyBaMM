#
# Class summing up contributions to the SEI reaction
# for cases with primary, secondary, ... reactions e.g. silicon-graphite
#
import pybamm


class TotalSEI(pybamm.BaseSubModel):
    """
    Class summing up contributions to the SEI reaction
    for cases with primary, secondary, ... reactions e.g. silicon-graphite

    Parameters
    ----------
    param :
        model parameters
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    """

    def __init__(self, param, domain, options, cracks=False):
        if cracks is True:
            self.reaction = "SEI on cracks"
        else:
            self.reaction = "SEI"
        super().__init__(param, domain, options=options)

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phases = self.options.phases[domain]
        # For each of the variables, the variable name without the phase name
        # is constructed by summing all of the variable names with the phases
        for variable_template in [
            f"{Domain} electrode {{}}{self.reaction} volumetric "
            "interfacial current density [A.m-3]",
            f"X-averaged {domain} electrode {{}}{self.reaction} volumetric "
            "interfacial current density [A.m-3]",
            f"Loss of lithium to {domain} {{}}{self.reaction} [mol]",
            f"Loss of capacity to {domain} {{}}{self.reaction} [A.h]",
        ]:
            sumvar = sum(
                variables[variable_template.format(phase + " ")] for phase in phases
            )
            variables[variable_template.format("")] = sumvar

        return variables
