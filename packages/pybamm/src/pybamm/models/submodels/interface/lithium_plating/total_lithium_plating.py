#
# Class summing up contributions to the lithium plating reaction
# for cases with primary, secondary, ... reactions e.g. silicon-graphite
#
import pybamm


class TotalLithiumPlating(pybamm.BaseSubModel):
    """
    Class summing up contributions to the lithium plating reaction
    for cases with primary, secondary, ... reactions e.g. silicon-graphite

    Parameters
    ----------
    param :
        model parameters
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    """

    def build(self):
        domain, Domain = self.domain_Domain
        phases = self.options.phases[domain]
        electrode = {"primary": f"{domain} electrode", "secondary": "current collector"}
        x_averaged = {"primary": "current collector"}
        volumetric = "volumetric interfacial current density [A.m-3]"
        # For each of the variables, the variable name without the phase name
        # is constructed by summing all of the variable names with the phases
        for variable_template, domains in [
            (f"{Domain} electrode {{}}lithium plating {volumetric}", electrode),
            (
                f"X-averaged {domain} electrode {{}}lithium plating {volumetric}",
                x_averaged,
            ),
            (f"Loss of lithium to {domain} {{}}lithium plating [mol]", {}),
            (f"Loss of capacity to {domain} {{}}lithium plating [A.h]", {}),
        ]:
            self.variables[variable_template.format("")] = sum(
                self.coupled_variable(
                    variable_template.format(phase + " "), domains=domains
                )
                for phase in phases
            )
