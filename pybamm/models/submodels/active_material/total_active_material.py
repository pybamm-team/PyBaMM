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
        Domain = self.domain
        domain = Domain.lower()

        phases = self.options.phase_number_to_names(
            getattr(self.options, domain)["particle phases"]
        )
        eps_solid = sum(
            variables[f"{Domain} electrode {phase} active material volume fraction"]
            for phase in phases
        )
        eps_solid_av = sum(
            variables[
                f"X-averaged {domain} electrode {phase} active material volume fraction"
            ]
            for phase in phases
        )
        variables.update(
            {
                f"{Domain} electrode active material volume fraction": eps_solid,
                f"X-averaged {domain} electrode active material volume fraction"
                "": eps_solid_av,
            }
        )

        if self.options["particle shape"] != "no particles":
            C = sum(
                variables[f"{Domain} electrode {phase} phase capacity [A.h]"]
                for phase in phases
            )
            variables.update({f"{Domain} electrode capacity [A.h]": C})

        deps_solid_dt = sum(
            variables[
                f"{Domain} electrode {phase} active material volume fraction change"
            ]
            for phase in phases
        )
        deps_solid_dt_av = sum(
            variables[
                f"X-averaged {domain} electrode {phase} active material "
                "volume fraction change"
            ]
            for phase in phases
        )
        variables.update(
            {
                f"{Domain} electrode active material volume fraction change"
                "": deps_solid_dt,
                f"X-averaged {domain} electrode active material volume fraction change"
                "": deps_solid_dt_av,
            }
        )

        return variables
