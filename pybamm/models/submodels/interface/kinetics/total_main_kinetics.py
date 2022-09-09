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
        Domain = self.domain
        domain = Domain.lower()

        if Domain == "Negative" and self.half_cell is True:
            return variables

        phases = self.options.phases[domain]
        if len(phases) > 1:
            a_j = sum(
                variables[
                    f"{Domain} electrode {phase} volumetric interfacial current density"
                ]
                for phase in phases
            )
            a_j_dim = sum(
                variables[
                    f"{Domain} electrode {phase} volumetric "
                    "interfacial current density [A.m-3]"
                ]
                for phase in phases
            )
            a_j_av = sum(
                variables[
                    f"X-averaged {domain} electrode {phase} volumetric "
                    "interfacial current density"
                ]
                for phase in phases
            )
            variables.update(
                {
                    f"{Domain} electrode volumetric interfacial current density": a_j,
                    f"{Domain} electrode volumetric "
                    "interfacial current density [A.m-2]": a_j_dim,
                    f"X-averaged {domain} electrode volumetric "
                    "interfacial current density": a_j_av,
                }
            )

        return variables
