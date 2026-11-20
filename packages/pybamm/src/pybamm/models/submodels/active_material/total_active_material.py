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
        electrode_domains = {
            "primary": f"{domain} electrode",
            "secondary": "current collector",
        }
        current_collector = {"primary": "current collector"}

        def sum_over_phases(template, domains):
            return sum(
                self.coupled_variable(template.format(phase + " "), domains=domains)
                for phase in phases
            )

        # Each total is named by the template without a phase name
        volume_fraction = "{}active material volume fraction"
        lithium_loss = (
            f"Loss of lithium due to loss of {{}}active material in {domain} "
            "electrode [mol]"
        )
        for template, domains in [
            (f"{Domain} electrode {volume_fraction}", electrode_domains),
            (f"X-averaged {domain} electrode {volume_fraction}", current_collector),
            (f"{Domain} electrode {volume_fraction} change [s-1]", electrode_domains),
            (
                f"X-averaged {domain} electrode {volume_fraction} change [s-1]",
                current_collector,
            ),
            (lithium_loss, {}),
        ]:
            self.variables[template.format("")] = sum_over_phases(template, domains)

        if self.options["particle shape"] != "no particles":
            self.variables[f"{Domain} electrode capacity [A.h]"] = sum_over_phases(
                f"{Domain} electrode {{}}phase capacity [A.h]", {}
            )

        if self.options["particle shape"] == "spherical":
            # every phase's particle surface lies in the same electrode volume
            a = sum_over_phases(
                f"{Domain} electrode {{}}surface area to volume ratio [m-1]",
                electrode_domains,
            )
            self.variables.update(
                {
                    f"{Domain} electrode surface area to volume ratio [m-1]": a,
                    f"X-averaged {domain} electrode surface area "
                    "to volume ratio [m-1]": pybamm.x_average(a),
                }
            )
