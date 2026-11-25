#
# Total interfacial current class, summing up contributions from all reactions
#
import pybamm


class TotalInterfacialCurrent(pybamm.BaseSubModel):
    """
    Total interfacial current, summing up contributions from all reactions

    Parameters
    ----------
    param :
        model parameters
    chemistry : str
        The name of the battery chemistry whose reactions need to be summed up
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    """

    def __init__(self, param, chemistry, options):
        super().__init__(param, options=options)
        self.chemistry = chemistry

    def build(self):
        """
        Get variables associated with interfacial current over the whole cell domain
        This function also creates the "total source term" variables by summing all
        the reactions.
        """
        variables = {}
        for domain in self.options.whole_cell_domains:
            if domain != "separator":
                variables.update(self._get_variables_by_domain(domain.split()[0]))
        variables.update(self._get_whole_cell_variables(variables))
        self.variables.update(variables)

    def _get_variables_by_domain(self, domain):
        phase_names = [""]
        num_phases = int(getattr(self.options, domain)["particle phases"])
        if num_phases > 1:
            phase_names += ["primary ", "secondary "]

        variables = {}
        for phase_name in phase_names:
            variables.update(
                self._get_variables_by_phase_and_domain(domain, phase_name)
            )
        return variables

    def _get_reactions(self, domain, phase_name):
        """Names of the reactions and their electrolyte source coefficients."""
        if self.chemistry == "lithium-ion":
            # All reactions contribute fully to the electrolyte reaction current
            reaction_names = [""]
            if phase_name == "":
                reaction_names += ["SEI "]
                if self.options.electrode_types[domain] == "porous":
                    # separate plating reaction only if the electrode is porous,
                    # since plating is the main reaction otherwise.
                    # Likewise, SEI on cracks only in a porous electrode
                    reaction_names.extend(["lithium plating ", "SEI on cracks "])
            return {reaction_name: 1 for reaction_name in reaction_names}
        else:  # "lead-acid"
            return {
                "": self.param.domain_params[domain].prim.s_plus_S,
                "oxygen ": self.param.s_plus_Ox,
            }

    def _get_variables_by_phase_and_domain(self, domain, phase_name):
        Domain = domain.capitalize()
        electrode_domains = {
            "primary": f"{domain} electrode",
            "secondary": "current collector",
        }

        source_terms = 0
        source_terms_av = 0
        a_j_sum = 0
        a_j_sum_av = 0
        for reaction_name, s_k in self._get_reactions(domain, phase_name).items():
            a_j_k = self.coupled_variable(
                f"{Domain} electrode {reaction_name}{phase_name}volumetric "
                "interfacial current density [A.m-3]",
                domains=electrode_domains,
            )
            source_terms += s_k * a_j_k
            source_terms_av += pybamm.x_average(s_k * a_j_k)
            a_j_sum += a_j_k
            a_j_sum_av += pybamm.x_average(a_j_k)

        return {
            f"Sum of {domain} electrode {phase_name}"
            "electrolyte reaction source terms [A.m-3]": source_terms,
            f"Sum of x-averaged {domain} electrode {phase_name}"
            "electrolyte reaction source terms [A.m-3]": source_terms_av,
            f"Sum of {domain} electrode {phase_name}"
            "volumetric interfacial current densities [A.m-3]": a_j_sum,
            f"Sum of x-averaged {domain} electrode {phase_name}"
            "volumetric interfacial current densities [A.m-3]": a_j_sum_av,
        }

    def _get_whole_cell_variables(self, variables):
        # Interfacial current density and exchange-current density for the main reaction
        zero_s = pybamm.FullBroadcast(0, "separator", "current collector")
        whole_cell_variables = {}

        if (
            self.options.negative["particle phases"] == "1"
            and self.options.positive["particle phases"] == "1"
        ):
            for variable_template in [
                "{}interfacial current density [A.m-2]",
                "{}exchange current density [A.m-2]",
            ]:
                var_dict = {}
                for domain in self.options.whole_cell_domains:
                    if domain == "separator":
                        var_dict[domain] = zero_s
                    else:
                        var_dict[domain] = self.coupled_variable(
                            variable_template.format(domain.capitalize() + " "),
                            domain=domain,
                            auxiliary_domains={"secondary": "current collector"},
                        )
                var = pybamm.concatenation(*var_dict.values())
                var_name = variable_template.format("")
                var_name = var_name[0].upper() + var_name[1:]  # capitalise first letter
                whole_cell_variables[var_name] = var

        # Sum variables
        for variable_template in [
            "Sum of {}volumetric interfacial current densities [A.m-3]",
            "Sum of {}electrolyte reaction source terms [A.m-3]",
        ]:
            var_dict = {}
            for domain in self.options.whole_cell_domains:
                if domain == "separator":
                    var_dict[domain] = zero_s
                else:
                    var_dict[domain] = variables[variable_template.format(domain + " ")]
            var = pybamm.concatenation(*var_dict.values())
            whole_cell_variables[variable_template.format("")] = var

        return whole_cell_variables
