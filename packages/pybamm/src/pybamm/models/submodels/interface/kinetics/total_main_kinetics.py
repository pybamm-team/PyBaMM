#
# Class summing up contributions to the main (e.g. intercalation) reaction
# for cases with primary, secondary, ... reactions e.g. silicon-graphite
#
import pybamm


class TotalMainKinetics(pybamm.BaseSubModel):
    """
    Class summing up contributions to the main (e.g. intercalation) reaction
    for cases with primary, secondary, ... reactions e.g. silicon-graphite

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    """

    def __init__(self, param, domain, reaction, options):
        super().__init__(param, domain, name=reaction, options=options)

    def build(self):
        domain, Domain = self.domain_Domain
        electrode_domains = {
            "primary": f"{domain} electrode",
            "secondary": "current collector",
        }
        # The variable name without the phase name is the sum over the phases
        sumvar = sum(
            self.coupled_variable(
                f"{Domain} electrode {phase} volumetric "
                "interfacial current density [A.m-3]",
                domains=electrode_domains,
            )
            for phase in self.options.phases[domain]
        )
        self.variables.update(
            {
                f"{Domain} electrode volumetric "
                "interfacial current density [A.m-3]": sumvar,
                f"X-averaged {domain} electrode volumetric "
                "interfacial current density [A.m-3]": pybamm.x_average(sumvar),
            }
        )
