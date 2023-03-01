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

    **Extends:** :class:`pybamm.interface.BaseInterface`
    """

    def __init__(self, param, domain, reaction, options):
        super().__init__(param, domain, reaction, options=options)

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain

        phases = self.options.phases[domain]
        # The variable name without the phase name
        # is constructed by summing all of the variable names with the phases
        sumvar = sum(
            variables[
                f"{Domain} electrode {phase} volumetric "
                "interfacial current density [A.m-3]"
            ]
            for phase in phases
        )
        variables.update(
            {
                f"{Domain} electrode volumetric "
                "interfacial current density [A.m-3]": sumvar,
                f"X-averaged {domain} electrode volumetric "
                "interfacial current density [A.m-3]": pybamm.x_average(sumvar),
            }
        )

        return variables
