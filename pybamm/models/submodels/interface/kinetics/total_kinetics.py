#
# Total kinetics class, summing up contributions from all reactions
#
import pybamm


class TotalKinetics(pybamm.BaseSubModel):
    """
    Total kinetics class, summing up contributions from all reactions

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
        the reactions
        """
        if self.chemistry == "lithium-ion":
            reaction_names = ["", "SEI "]
            if not self.half_cell:
                # no separate plating reaction in a half-cell,
                # since plating is the main reaction
                reaction_names.append("lithium plating ")
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
                "Sum of electrolyte reaction source terms": 0,
                "Sum of positive electrode electrolyte reaction source terms": 0,
                "Sum of x-averaged positive electrode "
                "electrolyte reaction source terms": 0,
                "Sum of interfacial current densities": 0,
                "Sum of volumetric interfacial current densities": 0,
                "Sum of positive electrode interfacial current densities": 0,
                "Sum of x-averaged positive electrode interfacial current densities": 0,
                "Sum of positive electrode volumetric interfacial current densities": 0,
                "Sum of x-averaged positive electrode volumetric "
                "interfacial current densities": 0,
            }
        )
        if not self.half_cell:
            new_variables.update(
                {
                    "Sum of negative electrode electrolyte reaction source terms": 0,
                    "Sum of x-averaged negative electrode "
                    "electrolyte reaction source terms": 0,
                    "Sum of negative electrode interfacial current densities": 0,
                    "Sum of x-averaged negative electrode "
                    "interfacial current densities": 0,
                    "Sum of negative electrode volumetric "
                    "interfacial current densities": 0,
                    "Sum of x-averaged negative electrode volumetric "
                    "interfacial current densities": 0,
                }
            )
        for reaction_name in reaction_names:
            j_p_av = variables[
                f"X-averaged positive electrode {reaction_name}"
                "interfacial current density"
            ]

            zero_s = pybamm.FullBroadcast(0, "separator", "current collector")
            j_p = variables[
                f"Positive electrode {reaction_name}interfacial current density"
            ]
            j_p_dim = variables[
                f"Positive electrode {reaction_name}interfacial current density [A.m-2]"
            ]

            if self.half_cell:
                j = pybamm.concatenation(zero_s, j_p)
                j_dim = pybamm.concatenation(zero_s, j_p_dim)
            else:
                j_n_av = variables[
                    f"X-averaged negative electrode {reaction_name}"
                    "interfacial current density"
                ]
                j_n = variables[
                    f"Negative electrode {reaction_name}interfacial current density"
                ]
                j_n_dim = variables[
                    f"Negative electrode {reaction_name}"
                    "interfacial current density [A.m-2]"
                ]
                j = pybamm.concatenation(j_n, zero_s, j_p)
                j_dim = pybamm.concatenation(j_n_dim, zero_s, j_p_dim)

            if reaction_name not in ["SEI ", "lithium plating "]:
                j0_p = variables[
                    f"Positive electrode {reaction_name}exchange current density"
                ]
                j0_p_dim = variables[
                    f"Positive electrode {reaction_name}"
                    "exchange current density [A.m-2]"
                ]

                if self.half_cell:
                    j0 = pybamm.concatenation(zero_s, j0_p)
                    j0_dim = pybamm.concatenation(zero_s, j0_p_dim)
                else:
                    j0_n = variables[
                        f"Negative electrode {reaction_name}exchange current density"
                    ]
                    j0_n_dim = variables[
                        f"Negative electrode {reaction_name}"
                        "exchange current density [A.m-2]"
                    ]
                    j0 = pybamm.concatenation(j0_n, zero_s, j0_p)
                    j0_dim = pybamm.concatenation(j0_n_dim, zero_s, j0_p_dim)
                new_variables.update(
                    {
                        f"{reaction_name}interfacial ".capitalize()
                        + "current density": j,
                        f"{reaction_name}interfacial ".capitalize()
                        + "current density [A.m-2]": j_dim,
                        f"{reaction_name}exchange ".capitalize()
                        + "current density": j0,
                        f"{reaction_name}exchange ".capitalize()
                        + "current density [A.m-2]": j0_dim,
                    }
                )

            # Sum variables
            a_j_p = new_variables[
                f"Positive electrode {reaction_name}volumetric "
                "interfacial current density"
            ]

            if self.chemistry == "lithium-ion":
                # Both the main reaction current contribute to the electrolyte reaction
                # current
                s_n, s_p = 1, 1
            elif self.chemistry == "lead-acid":
                if reaction_name == "":  # main reaction
                    s_n, s_p = self.param.n.s_plus_S, self.param.p.s_plus_S
                elif reaction_name == "oxygen ":
                    s_n, s_p = self.param.s_plus_Ox, self.param.s_plus_Ox
            if self.half_cell:
                a_j = pybamm.concatenation(zero_s, a_j_p)
                s = pybamm.concatenation(
                    zero_s,
                    pybamm.FullBroadcast(
                        s_p, "positive electrode", "current collector"
                    ),
                )
            else:
                a_j_n = new_variables[
                    f"Negative electrode {reaction_name}volumetric "
                    "interfacial current density"
                ]
                a_j = pybamm.concatenation(a_j_n, zero_s, a_j_p)
                s = pybamm.concatenation(
                    pybamm.FullBroadcast(
                        s_n, "negative electrode", "current collector"
                    ),
                    zero_s,
                    pybamm.FullBroadcast(
                        s_p, "positive electrode", "current collector"
                    ),
                )

            # Override print_name
            j.print_name = "J"
            a_j.print_name = "aj"
            j_p.print_name = "j_p"

            new_variables["Sum of electrolyte reaction source terms"] += s * a_j
            new_variables[
                "Sum of positive electrode electrolyte reaction source terms"
            ] += (s_p * a_j_p)
            new_variables[
                "Sum of x-averaged positive electrode electrolyte reaction source terms"
            ] += pybamm.x_average(s_p * a_j_p)

            new_variables["Sum of interfacial current densities"] += j
            new_variables["Sum of volumetric interfacial current densities"] += a_j
            new_variables[
                "Sum of positive electrode interfacial current densities"
            ] += j_p
            new_variables[
                "Sum of x-averaged positive electrode interfacial current densities"
            ] += j_p_av
            new_variables[
                "Sum of positive electrode volumetric interfacial current densities"
            ] += a_j_p
            new_variables[
                "Sum of x-averaged positive electrode volumetric "
                "interfacial current densities"
            ] += pybamm.x_average(a_j_p)

            if not self.half_cell:
                j_n.print_name = "j_n"
                new_variables[
                    "Sum of negative electrode electrolyte reaction source terms"
                ] += (s_n * a_j_n)
                new_variables[
                    "Sum of x-averaged negative electrode electrolyte "
                    "reaction source terms"
                ] += pybamm.x_average(s_n * a_j_n)
                new_variables[
                    "Sum of negative electrode interfacial current densities"
                ] += j_n
                new_variables[
                    "Sum of x-averaged negative electrode interfacial current densities"
                ] += j_n_av
                new_variables[
                    "Sum of negative electrode volumetric "
                    "interfacial current densities"
                ] += a_j_n
                new_variables[
                    "Sum of x-averaged negative electrode volumetric "
                    "interfacial current densities"
                ] += pybamm.x_average(a_j_n)

        variables.update(new_variables)

        return variables
