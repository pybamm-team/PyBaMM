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
        if self.half_cell:
            domains = ["positive"]
        else:
            domains = ["negative", "positive"]
        for domain in domains:
            variables.update(self._get_coupled_variables_by_domain(variables, domain))

        variables.update(self._get_whole_cell_coupled_variables(variables))

        return variables

    def _get_coupled_variables_by_domain(self, variables, domain):
        phase_names = [""]

        num_phases = int(getattr(self.options, domain.lower())["particle phases"])
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
                if not self.half_cell:
                    # no separate plating reaction in a half-cell,
                    # since plating is the main reaction
                    # no SEI on cracks with half-cell model
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
                "electrolyte reaction source terms": 0,
                f"Sum of x-averaged {domain} electrode {phase_name}"
                "electrolyte reaction source terms": 0,
                f"Sum of {domain} electrode {phase_name}"
                "volumetric interfacial current densities": 0,
                f"Sum of x-averaged {domain} electrode {phase_name}"
                "volumetric interfacial current densities": 0,
            }
        )
        for reaction_name in reaction_names:
            # Sum variables
            a_j_k = new_variables[
                f"{Domain} electrode {reaction_name}{phase_name}volumetric "
                "interfacial current density"
            ]

            if self.chemistry == "lithium-ion":
                # Both the main reaction current contribute to the electrolyte reaction
                # current
                s_k = 1
            elif self.chemistry == "lead-acid":
                if reaction_name == "":  # main reaction
                    if domain == "negative":
                        s_k = self.param.n.prim.s_plus_S
                    elif domain == "positive":
                        s_k = self.param.p.prim.s_plus_S
                elif reaction_name == "oxygen ":
                    s_k = self.param.s_plus_Ox

            new_variables[
                f"Sum of {domain} electrode {phase_name}"
                "electrolyte reaction source terms"
            ] += (s_k * a_j_k)
            new_variables[
                f"Sum of x-averaged {domain} electrode {phase_name}"
                "electrolyte reaction source terms"
            ] += pybamm.x_average(s_k * a_j_k)

            new_variables[
                f"Sum of {domain} electrode {phase_name}volumetric "
                "interfacial current densities"
            ] += a_j_k
            new_variables[
                f"Sum of x-averaged {domain} electrode {phase_name}"
                "volumetric interfacial current densities"
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
            j_p = variables["Positive electrode interfacial current density"]
            j_p_dim = variables[
                "Positive electrode interfacial current density [A.m-2]"
            ]
            j_p.print_name = "j_p"
            if self.half_cell:
                j = pybamm.concatenation(zero_s, j_p)
                j_dim = pybamm.concatenation(zero_s, j_p_dim)
            else:
                j_n = variables["Negative electrode interfacial current density"]
                j_n.print_name = "j_n"
                j_n_dim = variables[
                    "Negative electrode interfacial current density [A.m-2]"
                ]
                j = pybamm.concatenation(j_n, zero_s, j_p)
                j_dim = pybamm.concatenation(j_n_dim, zero_s, j_p_dim)

            j0_p = variables["Positive electrode exchange current density"]
            j0_p_dim = variables["Positive electrode exchange current density [A.m-2]"]

            if self.half_cell:
                j0 = pybamm.concatenation(zero_s, j0_p)
                j0_dim = pybamm.concatenation(zero_s, j0_p_dim)
            else:
                j0_n = variables["Negative electrode exchange current density"]
                j0_n_dim = variables[
                    "Negative electrode exchange current density [A.m-2]"
                ]
                j0 = pybamm.concatenation(j0_n, zero_s, j0_p)
                j0_dim = pybamm.concatenation(j0_n_dim, zero_s, j0_p_dim)
            variables.update(
                {
                    "Interfacial current density": j,
                    "Interfacial current density [A.m-2]": j_dim,
                    "Exchange current density": j0,
                    "Exchange current density [A.m-2]": j0_dim,
                }
            )

        # Sum variables
        a_j_p = variables[
            "Sum of positive electrode volumetric interfacial current densities"
        ]
        s_a_j_p = variables[
            "Sum of positive electrode electrolyte reaction source terms"
        ]

        if self.half_cell:
            a_j = pybamm.concatenation(zero_s, a_j_p)
            s_a_j = pybamm.concatenation(zero_s, s_a_j_p)
        else:
            a_j_n = variables[
                "Sum of negative electrode volumetric interfacial current densities"
            ]
            s_a_j_n = variables[
                "Sum of negative electrode electrolyte reaction source terms"
            ]
            a_j = pybamm.concatenation(a_j_n, zero_s, a_j_p)
            s_a_j = pybamm.concatenation(s_a_j_n, zero_s, s_a_j_p)

        # Override print_name
        a_j.print_name = "aj"

        variables["Sum of electrolyte reaction source terms"] = s_a_j
        variables["Sum of volumetric interfacial current densities"] = a_j

        return variables
