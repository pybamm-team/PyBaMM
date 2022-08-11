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
        # Creates "total" active material volume fraction and capacity variables
        # by summing up all the phases
        Domain = self.domain
        domain = Domain.lower()

        phases = phases = self.options.phase_number_to_names(
            getattr(self.options, domain)["particle phases"]
        )

        j = sum(
            variables[f"{Domain} electrode {phase} interfacial current density"]
            for phase in phases
        )
        j_dim = sum(
            variables[f"{Domain} electrode {phase} interfacial current density [A.m-2]"]
            for phase in phases
        )
        j_av = sum(
            variables[
                f"X-averaged {domain} electrode {phase} interfacial current density"
            ]
            for phase in phases
        )
        variables.update(
            {
                f"{Domain} electrode interfacial current density": j,
                f"{Domain} electrode interfacial current density [A.m-2]": j_dim,
                f"X-averaged {domain} electrode interfacial current density": j_av,
            }
        )

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

        j0 = sum(
            variables[f"{Domain} electrode {phase} exchange current density"]
            for phase in phases
        )
        j0_dim = sum(
            variables[f"{Domain} electrode {phase} exchange current density [A.m-2]"]
            for phase in phases
        )
        j0_av = sum(
            variables[f"X-averaged {domain} electrode {phase} exchange current density"]
            for phase in phases
        )
        variables.update(
            {
                f"{Domain} electrode exchange current density": j0,
                f"{Domain} electrode exchange current density [A.m-2]": j0_dim,
                f"X-averaged {domain} electrode exchange current density": j0_av,
            }
        )

        return variables
