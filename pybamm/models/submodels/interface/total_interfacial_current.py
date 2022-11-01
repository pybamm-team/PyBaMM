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

    **Extends:** :class:`pybamm.interface.BaseInterface`
    """

    def __init__(self, param, chemistry, options):
        super().__init__(param, options=options)
        self.chemistry = chemistry

    def get_coupled_variables(self, variables):
        """
        Get variables associated with interfacial current over the whole cell domain
        This function also creates the "total source term" variables by summing all
        the reactions.
        """
        for domain in self.options.whole_cell_domains:
            if domain != "separator":
                variables.update(
                    self._get_coupled_variables_by_domain(variables, domain.split()[0])
                )

        variables.update(self._get_whole_cell_coupled_variables(variables))

        return variables

    def _get_coupled_variables_by_domain(self, variables, domain):
        phase_names = [""]

        num_phases = int(getattr(self.options, domain)["particle phases"])
        if num_phases > 1:
            phase_names += ["primary ", "secondary "]

        for phase_name in phase_names:
            variables.update(
                self._get_coupled_variables_by_phase_and_domain(
                    variables, domain, phase_name
                )
            )

        return variables

    def _get_coupled_variables_by_phase_and_domain(self, variables, domain, phase_name):
        Domain = domain.capitalize()

        if self.chemistry == "lithium-ion":
            reaction_names = [""]
            if phase_name == "":
                reaction_names += ["SEI "]
                if self.options.electrode_types["negative"] == "porous":
                    # separate plating reaction only if the negative electrode is
                    # porous, since plating is the main reaction
                    # SEI on cracks only in a porous negative electrode
                    reaction_names.extend(["lithium plating ", "SEI on cracks "])
        elif self.chemistry == "lead-acid":
            reaction_names = ["", "oxygen "]

        # Create separate 'new_variables' so that variables only get updated once
        # everything is computed
        new_variables = variables.copy()

        # Initialize "total reaction" variables
        # These will get populated by each reaction, and then used
        # later by "set_rhs" or "set_algebraic", which ensures that we always have
        # added all the necessary variables by the time the sum is used
        new_variables.update(
            {
                f"Sum of {domain} electrode {phase_name}"
                "electrolyte reaction source terms [A.m-3]": 0,
                f"Sum of x-averaged {domain} electrode {phase_name}"
                "electrolyte reaction source terms [A.m-3]": 0,
                f"Sum of {domain} electrode {phase_name}"
                "volumetric interfacial current densities [A.m-3]": 0,
                f"Sum of x-averaged {domain} electrode {phase_name}"
                "volumetric interfacial current densities [A.m-3]": 0,
            }
        )
        for reaction_name in reaction_names:
            # Sum variables
            a_j_k = new_variables[
                f"{Domain} electrode {reaction_name}{phase_name}volumetric "
                "interfacial current density [A.m-3]"
            ]

            if self.chemistry == "lithium-ion":
                # Both the main reaction current contribute to the electrolyte reaction
                # current
                s_k = 1
            elif self.chemistry == "lead-acid":
                if reaction_name == "":  # main reaction
                    s_k = self.param.domain_params[domain].prim.s_plus_S
                elif reaction_name == "oxygen ":
                    s_k = self.param.s_plus_Ox

            new_variables[
                f"Sum of {domain} electrode {phase_name}"
                "electrolyte reaction source terms [A.m-3]"
            ] += (s_k * a_j_k)
            new_variables[
                f"Sum of x-averaged {domain} electrode {phase_name}"
                "electrolyte reaction source terms [A.m-3]"
            ] += pybamm.x_average(s_k * a_j_k)

            new_variables[
                f"Sum of {domain} electrode {phase_name}volumetric "
                "interfacial current densities [A.m-3]"
            ] += a_j_k
            new_variables[
                f"Sum of x-averaged {domain} electrode {phase_name}"
                "volumetric interfacial current densities [A.m-3]"
            ] += pybamm.x_average(a_j_k)

        variables.update(new_variables)

        return variables

    def _get_whole_cell_coupled_variables(self, variables):
        # Interfacial current density and exchange-current density for the main reaction
        zero_s = pybamm.FullBroadcast(0, "separator", "current collector")

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
                        Domain = domain.capitalize()
                        var_dict[domain] = variables[
                            variable_template.format(Domain + " ")
                        ]
                var = pybamm.concatenation(*var_dict.values())
                var_name = variable_template.format("")
                var_name = var_name[0].upper() + var_name[1:]  # capitalise first letter
                variables.update({var_name: var})

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
            variables.update({variable_template.format(""): var})

        return variables
